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
	// Extract system instruction from messages (Gemini CLI API format)
	var systemInstruction map[string]interface{}
	contents := make([]map[string]interface{}, 0)

	// Build tool call ID to name mapping for function responses
	toolCallMap := make(map[string]string) // tool_call_id -> function_name

	// First pass: collect tool call IDs and names
	for _, message := range req.Messages {
		if len(message.ToolCalls) > 0 {
			for _, tc := range message.ToolCalls {
				if tc.ID != "" && tc.Function.Name != "" {
					toolCallMap[tc.ID] = tc.Function.Name
				}
			}
		}
	}

	// Process messages and separate system instruction
	for i, message := range req.Messages {
		role := message.Role

		// Handle system messages - convert to systemInstruction
		if role == "system" {
			if i == 0 {
				// Only the first system message becomes systemInstruction
				parts := make([]map[string]interface{}, 0)
				switch content := message.Content.(type) {
				case string:
					parts = append(parts, map[string]interface{}{"text": content})
				case []interface{}:
					for _, part := range content {
						if partMap, ok := part.(map[string]interface{}); ok {
							if partType, _ := partMap["type"].(string); partType == "text" {
								if text, ok := partMap["text"].(string); ok {
									parts = append(parts, map[string]interface{}{"text": text})
								}
							}
						}
					}
				}

				if len(parts) > 0 {
					systemInstruction = map[string]interface{}{
						"parts": parts,
					}
				}
			} else {
				// Subsequent system messages become user messages
				role = "user"
			}
		}

		// Skip if this was the first system message (already processed)
		if role == "system" {
			continue
		}

		// Handle tool response messages
		if role == "tool" {
			// Debug: log tool response details
			if config.IsDebugEnabled() {
				log.Printf("[DEBUG] Processing tool response: tool_call_id=%s, name=%s, has_content=%v",
					message.ToolCallID, message.Name, message.Content != nil)
			}

			// Get function name from message or tool call map
			functionName := message.Name
			if functionName == "" && message.ToolCallID != "" {
				// Try to get name from tool call ID mapping
				if name, ok := toolCallMap[message.ToolCallID]; ok {
					functionName = name
					if config.IsDebugEnabled() {
						log.Printf("[DEBUG] Found function name '%s' from tool call ID '%s'", functionName, message.ToolCallID)
					}
				}
			}

			// Skip if we still don't have a function name
			if functionName == "" {
				log.Printf("Warning: Skipping tool response without function name (tool_call_id: %s)", message.ToolCallID)
				continue
			}

			// Parse tool response content
			var responseContent interface{}
			if contentStr, ok := message.Content.(string); ok {
				// Try to parse as JSON
				var jsonContent interface{}
				if err := json.Unmarshal([]byte(contentStr), &jsonContent); err == nil {
					responseContent = jsonContent
				} else {
					responseContent = contentStr
				}
			} else {
				responseContent = message.Content
			}

			// Gemini CLI format for function response
			contents = append(contents, map[string]interface{}{
				"role": "user",
				"parts": []map[string]interface{}{
					{
						"functionResponse": map[string]interface{}{
							"name": functionName,
							"response": map[string]interface{}{
								"name":    functionName,
								"content": responseContent,
							},
						},
					},
				},
			})
			continue
		}

		// Map assistant role to model
		if role == "assistant" {
			role = "model"
		}

		// Handle assistant messages with tool calls
		if len(message.ToolCalls) > 0 {
			parts := make([]map[string]interface{}, 0)

			// Add text content if present
			if contentStr, ok := message.Content.(string); ok && contentStr != "" {
				parts = append(parts, map[string]interface{}{"text": contentStr})
			}

			// Add function calls
			for _, tc := range message.ToolCalls {
				// Parse arguments
				var argsMap map[string]interface{}
				if argsStr, ok := tc.Function.Arguments.(string); ok && argsStr != "" {
					if err := json.Unmarshal([]byte(argsStr), &argsMap); err != nil {
						log.Printf("Warning: Failed to parse tool call arguments: %v", err)
						argsMap = make(map[string]interface{})
					}
				} else if argsMapRaw, ok := tc.Function.Arguments.(map[string]interface{}); ok {
					argsMap = argsMapRaw
				} else {
					argsMap = make(map[string]interface{})
				}

				parts = append(parts, map[string]interface{}{
					"functionCall": map[string]interface{}{
						"name": tc.Function.Name,
						"args": argsMap,
					},
				})
			}

			contents = append(contents, map[string]interface{}{
				"role":  "model",
				"parts": parts,
			})
			continue
		}

		// Handle regular messages (user/assistant)
		parts := make([]map[string]interface{}, 0)

		switch content := message.Content.(type) {
		case string:
			// Extract Markdown images and convert to inline data
			parts = extractMarkdownImages(content)

		case []interface{}:
			// Handle structured content parts
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

		// Ensure at least one part
		if len(parts) == 0 {
			parts = append(parts, map[string]interface{}{"text": ""})
		}

		contents = append(contents, map[string]interface{}{
			"role":  role,
			"parts": parts,
		})
	}

	// Build generation config
	generationConfig := make(map[string]interface{})

	// Temperature
	if req.Temperature != nil {
		generationConfig["temperature"] = *req.Temperature
	}

	// Top P
	if req.TopP != nil {
		generationConfig["topP"] = *req.TopP
	}

	// Max tokens
	if req.MaxTokens != nil {
		generationConfig["maxOutputTokens"] = *req.MaxTokens
	}

	// Stop sequences
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
			if len(stopSeqs) > 0 {
				generationConfig["stopSequences"] = stopSeqs
			}
		}
	}

	// Frequency penalty
	if req.FrequencyPenalty != nil {
		generationConfig["frequencyPenalty"] = *req.FrequencyPenalty
	}

	// Presence penalty
	if req.PresencePenalty != nil {
		generationConfig["presencePenalty"] = *req.PresencePenalty
	}

	// Candidate count
	if req.N != nil {
		generationConfig["candidateCount"] = *req.N
	}

	// Seed
	if req.Seed != nil {
		generationConfig["seed"] = *req.Seed
	}

	// Response format (JSON mode)
	if req.ResponseFormat != nil {
		if respType, ok := req.ResponseFormat["type"].(string); ok && respType == "json_object" {
			generationConfig["responseMimeType"] = "application/json"
		}
	}

	// Build request payload
	requestPayload := map[string]interface{}{
		"contents":         contents,
		"generationConfig": generationConfig,
		"safetySettings":   config.DefaultSafetySettings,
		"model":            req.Model,
	}

	// Add system instruction if present
	if systemInstruction != nil {
		requestPayload["systemInstruction"] = systemInstruction
	}

	// Add tools if present
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

		if len(functionDeclarations) > 0 {
			requestPayload["tools"] = []map[string]interface{}{
				{
					"function_declarations": functionDeclarations,
				},
			}

			// Add tool config if tool_choice is specified
			if req.ToolChoice != nil {
				toolConfig := map[string]interface{}{}

				switch tc := req.ToolChoice.(type) {
				case string:
					if tc == "auto" {
						toolConfig["functionCallingConfig"] = map[string]interface{}{
							"mode": "AUTO",
						}
					} else if tc == "none" {
						toolConfig["functionCallingConfig"] = map[string]interface{}{
							"mode": "NONE",
						}
					} else if tc == "required" {
						toolConfig["functionCallingConfig"] = map[string]interface{}{
							"mode": "ANY",
						}
					}
				case map[string]interface{}:
					// Specific function choice
					if tcType, ok := tc["type"].(string); ok && tcType == "function" {
						if fn, ok := tc["function"].(map[string]interface{}); ok {
							if name, ok := fn["name"].(string); ok {
								toolConfig["functionCallingConfig"] = map[string]interface{}{
									"mode":                 "ANY",
									"allowedFunctionNames": []string{name},
								}
							}
						}
					}
				}

				if len(toolConfig) > 0 {
					requestPayload["toolConfig"] = toolConfig
				}
			}
		}
	}

	// Debug log the final request payload
	if config.IsDebugEnabled() {
		payloadJSON, _ := json.MarshalIndent(requestPayload, "", "  ")
		log.Printf("[DEBUG] Gemini CLI API Request Payload:\n%s", string(payloadJSON))
	}

	return requestPayload
}

// GeminiResponseToOpenAI transforms a Gemini API response to OpenAI chat completion format
func GeminiResponseToOpenAI(geminiResp map[string]interface{}, model string) map[string]interface{} {
	choices := make([]map[string]interface{}, 0)

	candidates, _ := geminiResp["candidates"].([]interface{})
	for _, candidate := range candidates {
		candMap, _ := candidate.(map[string]interface{})
		content, _ := candMap["content"].(map[string]interface{})

		// Extract parts
		parts, _ := content["parts"].([]interface{})

		contentParts := make([]string, 0)
		toolCalls := make([]map[string]interface{}, 0)
		reasoningContent := ""

		for _, part := range parts {
			partMap, _ := part.(map[string]interface{})

			// Handle function calls
			if fnCall, ok := partMap["functionCall"].(map[string]interface{}); ok {
				name, _ := fnCall["name"].(string)
				args, _ := fnCall["args"].(map[string]interface{})
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

			// Handle text (with thinking tokens)
			if text, ok := partMap["text"].(string); ok {
				if thought, _ := partMap["thought"].(bool); thought {
					reasoningContent += text
				} else {
					contentParts = append(contentParts, text)
				}
				continue
			}

			// Handle inline images
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

		// Build message
		contentStr := strings.Join(contentParts, "")
		message := map[string]interface{}{
			"role":    "assistant",
			"content": contentStr,
		}

		// Add tool calls if present
		if len(toolCalls) > 0 {
			message["tool_calls"] = toolCalls
			if contentStr == "" {
				message["content"] = nil
			}
		}

		// Add reasoning content
		if reasoningContent != "" {
			message["reasoning_content"] = reasoningContent
		}

		// Map finish reason
		index, _ := candMap["index"].(float64)
		finishReason, _ := candMap["finishReason"].(string)

		mappedFinishReason := MapFinishReason(finishReason)
		if len(toolCalls) > 0 {
			mappedFinishReason = "tool_calls"
		}

		choices = append(choices, map[string]interface{}{
			"index":         int(index),
			"message":       message,
			"finish_reason": mappedFinishReason,
		})
	}

	return map[string]interface{}{
		"id":      "chatcmpl-" + uuid.New().String(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": choices,
	}
}

// MapFinishReason converts Gemini finish reasons to OpenAI format
func MapFinishReason(geminiReason string) interface{} {
	switch geminiReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION":
		return "content_filter"
	case "OTHER":
		return "stop"
	default:
		if geminiReason == "" {
			return nil
		}
		return "stop"
	}
}

// Helper functions

func extractMarkdownImages(text string) []map[string]interface{} {
	parts := make([]map[string]interface{}, 0)
	pattern := regexp.MustCompile(`!\[[^\]]*\]\(([^)]+)\)`)
	matches := pattern.FindAllStringSubmatchIndex(text, -1)

	if len(matches) == 0 {
		if text != "" {
			parts = append(parts, map[string]interface{}{"text": text})
		}
		return parts
	}

	lastIdx := 0
	for _, match := range matches {
		start, end := match[0], match[1]
		urlStart, urlEnd := match[2], match[3]

		// Text before image
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
				parts = append(parts, map[string]interface{}{"text": text[start:end]})
			}
		} else {
			// Keep non-data URIs as text
			parts = append(parts, map[string]interface{}{"text": text[start:end]})
		}

		lastIdx = end
	}

	// Remaining text
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

	// Extract MIME type
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
