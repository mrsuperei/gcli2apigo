package transformers

import (
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"
	"time"

	"gcli2apigo/internal/config"
	"gcli2apigo/internal/models"

	"github.com/google/uuid"
)

// OpenAIRequestToGemini transforms an OpenAI chat completion request to Gemini format
func OpenAIRequestToGemini(req *models.OpenAIChatCompletionRequest) map[string]interface{} {
	var tools []map[string]interface{}
	if len(req.Tools) > 0 {
		functionDeclarations := make([]map[string]interface{}, 0)
		for _, t := range req.Tools {
			if t.Type == "function" {
				functionDeclarations = append(functionDeclarations, map[string]interface{}{
					"name":        t.Function.Name,
					"description": t.Function.Description,
					"parameters":  t.Function.Parameters,
				})
			}
		}
		// Gemini verwacht tools gewikkeld in "function_declarations"
		tools = append(tools, map[string]interface{}{
			"function_declarations": functionDeclarations,
		})
	}
	contents := make([]map[string]interface{}, 0)

	// Process each message in the conversation
	for _, message := range req.Messages {
		role := message.Role

		// Map OpenAI roles to Gemini roles
		if role == "assistant" {
			role = "model"
		} else if role == "system" {
			role = "user"
		}

		parts := make([]map[string]interface{}, 0)
		// Specifieke handling voor Tool Responses (OpenAI: role="tool" -> Gemini: role="function")
		if role == "tool" {
			role = "function" // Gemini gebruikt soms 'user' context of specifieke 'function' role afhankelijk van versie, check docs voor v1beta

			// Gemini verwacht functionResponse structuur
			parts = append(parts, map[string]interface{}{
				"functionResponse": map[string]interface{}{
					"name": message.Name, // OpenAI stuurt dit soms niet mee in de 'tool' message, je moet dit mogelijk cachen of uit context halen
					"response": map[string]interface{}{
						"content": message.Content, // JSON object of string
					},
				},
			})
		} else if len(message.ToolCalls) > 0 {
			// Handling Assistant die een Tool Call doet (historie)
			role = "model"
			for _, tc := range message.ToolCalls {
				// Parse arguments string naar map, want Gemini wil een object, geen string
				var argsMap map[string]interface{}
				// Probeer de argument string te parsen. Als het faalt, stuur lege map.
				// tc.Function.Arguments is een string in OpenAI models
				if argsStr, ok := tc.Function.Arguments.(string); ok {
					_ = json.Unmarshal([]byte(argsStr), &argsMap)
				} else if argsMapRaw, ok := tc.Function.Arguments.(map[string]interface{}); ok {
					// Als het al een map is (intern gebruik)
					argsMap = argsMapRaw
				}

				parts = append(parts, map[string]interface{}{
					"functionCall": map[string]interface{}{
						"name": tc.Function.Name,
						"args": argsMap,
					},
				})
			}
		} else {
			// Handle different content types
			switch content := message.Content.(type) {
			case string:
				// Simple text content; extract Markdown images
				parts = extractMarkdownImages(content)

			case []interface{}:
				// List of content parts
				for _, part := range content {
					if partMap, ok := part.(map[string]interface{}); ok {
						if partType, _ := partMap["type"].(string); partType == "text" {
							if text, ok := partMap["text"].(string); ok {
								parts = append(parts, extractMarkdownImages(text)...)
							}
						} else if partType == "image_url" {
							if imageURL, ok := partMap["image_url"].(map[string]interface{}); ok {
								if url, ok := imageURL["url"].(string); ok {
									imagePart := parseDataURI(url)
									if imagePart != nil {
										parts = append(parts, imagePart)
									}
								}
							}
						}
					}
				}
			}

			if len(parts) == 0 {
				parts = append(parts, map[string]interface{}{"text": ""})
			}

			contents = append(contents, map[string]interface{}{
				"role":  role,
				"parts": parts,
			})
		}
	}
	// Map OpenAI generation parameters to Gemini format
	generationConfig := make(map[string]interface{})

	// Set minimum thinking budget for all models
	generationConfig["thinkingConfig"] = map[string]interface{}{
		"thinkingBudget": config.GetThinkingBudget(req.Model),
	}

	if req.Temperature != nil {
		generationConfig["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		generationConfig["topP"] = *req.TopP
	}
	if req.MaxTokens != nil {
		generationConfig["maxOutputTokens"] = *req.MaxTokens
		if config.IsDebugEnabled() {
			log.Printf("[DEBUG] Using client-specified maxOutputTokens: %d", *req.MaxTokens)
		}
	} else {
		// Set a high default to prevent truncation when client doesn't specify max_tokens
		// Gemini models support up to 65,535 tokens output
		generationConfig["maxOutputTokens"] = 65535
		if config.IsDebugEnabled() {
			log.Printf("[DEBUG] No max_tokens specified, using default maxOutputTokens: 65535")
		}
	}
	if req.Stop != nil {
		switch stop := req.Stop.(type) {
		case string:
			generationConfig["stopSequences"] = []string{stop}
		case []interface{}:
			stopSeqs := make([]string, 0)
			for _, s := range stop {
				if str, ok := s.(string); ok {
					stopSeqs = append(stopSeqs, str)
				}
			}
			generationConfig["stopSequences"] = stopSeqs
		}
	}
	if req.FrequencyPenalty != nil {
		generationConfig["frequencyPenalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		generationConfig["presencePenalty"] = *req.PresencePenalty
	}
	if req.N != nil {
		generationConfig["candidateCount"] = *req.N
	}
	if req.Seed != nil {
		generationConfig["seed"] = *req.Seed
	}
	if req.ResponseFormat != nil {
		if respType, ok := req.ResponseFormat["type"].(string); ok && respType == "json_object" {
			generationConfig["responseMimeType"] = "application/json"
		}
	}

	// Build the request payload
	requestPayload := map[string]interface{}{
		"contents":         contents,
		"generationConfig": generationConfig,
		"safetySettings":   config.DefaultSafetySettings,
		"model":            req.Model,
	}
	if len(tools) > 0 {
		requestPayload["tools"] = tools
	}

	return requestPayload
}

// GeminiResponseToOpenAI transforms a Gemini API response to OpenAI chat completion format
func GeminiResponseToOpenAI(geminiResp map[string]interface{}, model string) map[string]interface{} {
	if config.IsDebugEnabled() {
		respBytes, _ := json.MarshalIndent(geminiResp, "", "  ")
		log.Printf("[DEBUG] Raw Gemini Response: %s", string(respBytes))
	}
	choices := make([]map[string]interface{}, 0)

	candidates, _ := geminiResp["candidates"].([]interface{})
	for _, candidate := range candidates {
		candMap, _ := candidate.(map[string]interface{})
		content, _ := candMap["content"].(map[string]interface{})
		role, _ := content["role"].(string)

		// Map Gemini roles back to OpenAI roles
		if role == "model" {
			role = "assistant"
		}

		// Extract and separate thinking tokens from regular content and tool calls
		parts, _ := content["parts"].([]interface{})
		contentParts := make([]string, 0)
		toolCalls := make([]map[string]interface{}, 0)
		reasoningContent := ""

		for _, part := range parts {
			partMap, _ := part.(map[string]interface{})

			// Check for Function Call
			if fnCall, ok := partMap["functionCall"].(map[string]interface{}); ok {
				name, _ := fnCall["name"].(string)
				args, _ := fnCall["args"].(map[string]interface{})

				// OpenAI verwacht argumenten als JSON string, Gemini geeft een object.
				// We moeten het object terug converteren naar een JSON string.
				argsBytes, _ := json.Marshal(args)

				toolCalls = append(toolCalls, map[string]interface{}{
					"id":   "call_" + uuid.New().String(),
					"type": "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": string(argsBytes),
					},
				})
				continue
			}

			// Text parts (may include thinking tokens)
			if text, ok := partMap["text"].(string); ok {
				if thought, _ := partMap["thought"].(bool); thought {
					reasoningContent += text
				} else {
					contentParts = append(contentParts, text)
				}
				continue
			}

			// Inline image data -> embed as Markdown data URI
			if inlineData, ok := partMap["inlineData"].(map[string]interface{}); ok {
				if data, ok := inlineData["data"].(string); ok {
					mimeType, _ := inlineData["mimeType"].(string)
					if mimeType == "" {
						mimeType = "image/png"
					}
					if strings.HasPrefix(mimeType, "image/") {
						contentParts = append(contentParts, fmt.Sprintf("![image](data:%s;base64,%s)", mimeType, data))
					}
				}
			}
		}

		contentStr := strings.Join(contentParts, "")

		// Build message object
		message := map[string]interface{}{
			"role":    role,
			"content": contentStr,
		}

		// Add tool_calls if present
		if len(toolCalls) > 0 {
			message["tool_calls"] = toolCalls
			// Als er tool calls zijn, is content vaak null of leeg in OpenAI spec
			if contentStr == "" {
				message["content"] = nil
			}
		}

		// Add reasoning_content if there are thinking tokens
		if reasoningContent != "" {
			message["reasoning_content"] = reasoningContent
		}

		index, _ := candMap["index"].(float64)
		finishReason, _ := candMap["finishReason"].(string)

		// Override finish reason als we tool calls hebben
		finalFinishReason := mapFinishReason(finishReason)
		if len(toolCalls) > 0 {
			finalFinishReason = "tool_calls"
		}

		choices = append(choices, map[string]interface{}{
			"index":         int(index),
			"message":       message,
			"finish_reason": finalFinishReason,
		})
	}

	return map[string]interface{}{
		"id":      uuid.New().String(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": choices,
	}
}

// GeminiStreamChunkToOpenAI transforms a Gemini streaming response chunk to OpenAI streaming format
func GeminiStreamChunkToOpenAI(geminiChunk map[string]interface{}, model string, responseID string) map[string]interface{} {
	choices := make([]map[string]interface{}, 0)

	candidates, _ := geminiChunk["candidates"].([]interface{})
	for _, candidate := range candidates {
		candMap, _ := candidate.(map[string]interface{})
		content, _ := candMap["content"].(map[string]interface{})
		role, _ := content["role"].(string)

		// Map Gemini roles back to OpenAI roles
		if role == "model" {
			role = "assistant"
		}

		// Extract and separate thinking tokens from regular content
		parts, _ := content["parts"].([]interface{})
		contentParts := make([]string, 0)
		toolCalls := make([]map[string]interface{}, 0)
		reasoningContent := ""

		for _, part := range parts {
			partMap, _ := part.(map[string]interface{})

			// Check for Function Call
			if fnCall, ok := partMap["functionCall"].(map[string]interface{}); ok {
				name, _ := fnCall["name"].(string)
				args, _ := fnCall["args"].(map[string]interface{})

				argsBytes, _ := json.Marshal(args)

				toolCalls = append(toolCalls, map[string]interface{}{
					"index": 0, // In stream chunks, meestal index 0 voor de eerste tool call
					"id":    "call_" + responseID,
					"type":  "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": string(argsBytes),
					},
				})
				continue
			}

			// Text parts (may include thinking tokens)
			if text, ok := partMap["text"].(string); ok {
				if thought, _ := partMap["thought"].(bool); thought {
					reasoningContent += text
				} else {
					contentParts = append(contentParts, text)
				}
				continue
			}

			// Inline image data -> embed as Markdown data URI
			if inlineData, ok := partMap["inlineData"].(map[string]interface{}); ok {
				if data, ok := inlineData["data"].(string); ok {
					mimeType, _ := inlineData["mimeType"].(string)
					if mimeType == "" {
						mimeType = "image/png"
					}
					if strings.HasPrefix(mimeType, "image/") {
						contentParts = append(contentParts, fmt.Sprintf("![image](data:%s;base64,%s)", mimeType, data))
					}
				}
			}
		}

		contentStr := strings.Join(contentParts, "")

		// Build delta object
		delta := make(map[string]interface{})
		if contentStr != "" {
			delta["content"] = contentStr
		}
		if reasoningContent != "" {
			delta["reasoning_content"] = reasoningContent
		}
		if len(toolCalls) > 0 {
			delta["tool_calls"] = toolCalls
		}

		index, _ := candMap["index"].(float64)
		finishReason, _ := candMap["finishReason"].(string)

		finalFinishReason := mapFinishReason(finishReason)
		if len(toolCalls) > 0 {
			finalFinishReason = "tool_calls"
		}

		choices = append(choices, map[string]interface{}{
			"index":         int(index),
			"delta":         delta,
			"finish_reason": finalFinishReason,
		})
	}

	return map[string]interface{}{
		"id":      responseID,
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": choices,
	}
}

func MapFinishReason(geminiReason string) interface{} {
	switch geminiReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION":
		return "content_filter"
	default:
		return nil
	}
}

// Deprecated: use MapFinishReason instead
func mapFinishReason(geminiReason string) interface{} {
	return MapFinishReason(geminiReason)
}

func extractMarkdownImages(text string) []map[string]interface{} {
	parts := make([]map[string]interface{}, 0)
	pattern := regexp.MustCompile(`!\[[^\]]*\]\(([^)]+)\)`)
	matches := pattern.FindAllStringSubmatchIndex(text, -1)

	if len(matches) == 0 {
		parts = append(parts, map[string]interface{}{"text": text})
		return parts
	}

	lastIdx := 0
	for _, match := range matches {
		// match[0] = start of full match, match[1] = end of full match
		// match[2] = start of URL group, match[3] = end of URL group
		start, end := match[0], match[1]
		urlStart, urlEnd := match[2], match[3]

		// Emit text before the image
		if start > lastIdx {
			before := text[lastIdx:start]
			if before != "" {
				parts = append(parts, map[string]interface{}{"text": before})
			}
		}

		// Handle data URI images
		url := strings.TrimSpace(text[urlStart:urlEnd])
		url = strings.Trim(url, `"'`)

		if strings.HasPrefix(url, "data:") {
			imagePart := parseDataURI(url)
			if imagePart != nil {
				parts = append(parts, imagePart)
			} else {
				// Fallback: keep original markdown as text
				parts = append(parts, map[string]interface{}{"text": text[start:end]})
			}
		} else {
			// Non-data URIs: keep markdown as text
			parts = append(parts, map[string]interface{}{"text": text[start:end]})
		}

		lastIdx = end
	}

	// Tail text after last image
	if lastIdx < len(text) {
		tail := text[lastIdx:]
		if tail != "" {
			parts = append(parts, map[string]interface{}{"text": tail})
		}
	}

	return parts
}

func parseDataURI(url string) map[string]interface{} {
	if !strings.HasPrefix(url, "data:") {
		return nil
	}

	parts := strings.SplitN(url, ",", 2)
	if len(parts) != 2 {
		return nil
	}

	header := parts[0]
	base64Data := parts[1]

	// Extract MIME type from header (e.g., "data:image/png;base64")
	mimeType := "image/png"
	if strings.Contains(header, ":") {
		headerParts := strings.SplitN(header, ":", 2)
		if len(headerParts) == 2 {
			mimeTypePart := strings.Split(headerParts[1], ";")[0]
			if mimeTypePart != "" {
				mimeType = mimeTypePart
			}
		}
	}

	return map[string]interface{}{
		"inlineData": map[string]interface{}{
			"mimeType": mimeType,
			"data":     base64Data,
		},
	}
}

// AssembleCompleteResponse merges streaming chunks into a single complete response
// This is used for fake stream mode where chunks are collected internally but returned as a single response
func AssembleCompleteResponse(chunks []map[string]interface{}, model string) map[string]interface{} {
	if len(chunks) == 0 {
		return map[string]interface{}{
			"id":      uuid.New().String(),
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]interface{}{},
		}
	}

	// Accumulate content from all chunks by candidate index
	candidateMap := make(map[int]*candidateAccumulator)

	for _, chunk := range chunks {
		candidates, _ := chunk["candidates"].([]interface{})
		for _, candidate := range candidates {
			candMap, _ := candidate.(map[string]interface{})
			index, _ := candMap["index"].(float64)
			candIndex := int(index)

			// Initialize accumulator for this candidate if not exists
			if _, exists := candidateMap[candIndex]; !exists {
				candidateMap[candIndex] = &candidateAccumulator{
					index:            candIndex,
					contentParts:     make([]string, 0),
					toolCalls:        make([]map[string]interface{}, 0),
					reasoningContent: "",
					finishReason:     "",
					role:             "",
				}
			}

			acc := candidateMap[candIndex]

			// Extract content from this chunk
			content, _ := candMap["content"].(map[string]interface{})
			role, _ := content["role"].(string)
			if role != "" {
				acc.role = role
			}

			// Process parts
			parts, _ := content["parts"].([]interface{})
			for _, part := range parts {
				partMap, _ := part.(map[string]interface{})

				// Check for Function Call
				if fnCall, ok := partMap["functionCall"].(map[string]interface{}); ok {
					name, _ := fnCall["name"].(string)
					args, _ := fnCall["args"].(map[string]interface{})
					argsBytes, _ := json.Marshal(args)

					// Append to accumulated tool calls
					acc.toolCalls = append(acc.toolCalls, map[string]interface{}{
						"id":   "call_" + uuid.New().String(),
						"type": "function",
						"function": map[string]interface{}{
							"name":      name,
							"arguments": string(argsBytes),
						},
					})
					continue
				}

				// Text parts (may include thinking tokens)
				if text, ok := partMap["text"].(string); ok {
					if thought, _ := partMap["thought"].(bool); thought {
						acc.reasoningContent += text
					} else {
						acc.contentParts = append(acc.contentParts, text)
					}
					continue
				}

				// Inline image data -> embed as Markdown data URI
				if inlineData, ok := partMap["inlineData"].(map[string]interface{}); ok {
					if data, ok := inlineData["data"].(string); ok {
						mimeType, _ := inlineData["mimeType"].(string)
						if mimeType == "" {
							mimeType = "image/png"
						}
						if strings.HasPrefix(mimeType, "image/") {
							acc.contentParts = append(acc.contentParts, fmt.Sprintf("![image](data:%s;base64,%s)", mimeType, data))
						}
					}
				}
			}

			// Update finish reason (use the last one)
			if finishReason, ok := candMap["finishReason"].(string); ok && finishReason != "" {
				acc.finishReason = finishReason
			}
		}
	}

	// Build choices from accumulated candidates
	choices := make([]map[string]interface{}, 0)
	for i := 0; i < len(candidateMap); i++ {
		if acc, exists := candidateMap[i]; exists {
			role := acc.role
			// Map Gemini roles back to OpenAI roles
			if role == "model" {
				role = "assistant"
			}

			// Combine all content parts
			contentStr := strings.Join(acc.contentParts, "")

			// Build message object
			message := map[string]interface{}{
				"role":    role,
				"content": contentStr,
			}

			// Add tool_calls
			if len(acc.toolCalls) > 0 {
				message["tool_calls"] = acc.toolCalls
				if contentStr == "" {
					message["content"] = nil
				}
			}

			// Add reasoning_content if there are thinking tokens
			if acc.reasoningContent != "" {
				message["reasoning_content"] = acc.reasoningContent
			}

			finishReason := mapFinishReason(acc.finishReason)
			if len(acc.toolCalls) > 0 {
				finishReason = "tool_calls"
			}

			choices = append(choices, map[string]interface{}{
				"index":         acc.index,
				"message":       message,
				"finish_reason": finishReason,
			})
		}
	}

	return map[string]interface{}{
		"id":      uuid.New().String(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": choices,
	}
}

// candidateAccumulator holds accumulated content for a single candidate across chunks
type candidateAccumulator struct {
	index            int
	contentParts     []string
	toolCalls        []map[string]interface{}
	reasoningContent string
	finishReason     string
	role             string
}
