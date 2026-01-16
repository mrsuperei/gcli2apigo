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

// ReasoningEffortToThinkingBudget maps OpenAI reasoning_effort to Gemini thinking budget
// low: 1024, medium: 4096, high: 8192
func ReasoningEffortToThinkingBudget(effort string) int {
	switch strings.ToLower(effort) {
	case "low":
		return 1024
	case "medium":
		return 4096
	case "high":
		return 8192
	default:
		return -1 // Gemini default
	}
}

// OpenAIRequestToGemini transforms an OpenAI chat completion request to Gemini format
// Now with full reasoning/thinking support
func OpenAIRequestToGemini(req *models.OpenAIChatCompletionRequest) map[string]interface{} {
	log.Printf("[DEBUG] ═══════════════════════════════════════════")
	log.Printf("[DEBUG] OpenAIRequestToGemini CALLED")
	log.Printf("[DEBUG] Model: %s", req.Model)
	log.Printf("[DEBUG] ReasoningEffort field: '%s'", req.ReasoningEffort)
	log.Printf("[DEBUG] ═══════════════════════════════════════════")

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

	// ===== REASONING/THINKING SUPPORT =====
	// Support: reasoning_effort (string), thinking_tokens (int), thinking_enabled (bool)
	var thinkingBudget int

	log.Printf("[DEBUG] ════════ REASONING EFFORT CHECK START ════════")
	log.Printf("[DEBUG] req.ReasoningEffort = '%s' (empty: %v)", req.ReasoningEffort, req.ReasoningEffort == "")
	log.Printf("[DEBUG] req.ThinkingTokens = %v", req.ThinkingTokens)
	log.Printf("[DEBUG] req.ThinkingEnabled = %v", req.ThinkingEnabled)

	// PRIORITY 0: Check nested GenerationConfig (trpc-agent-go format)
	if req.GenerationConfig != nil {
		genCfg := *req.GenerationConfig

		// Check thinking_tokens in nested config
		if tokens, ok := genCfg["thinking_tokens"].(float64); ok && tokens > 0 {
			thinkingBudget = int(tokens)
			log.Printf("[DEBUG] ✅ Using thinking_tokens from nested generation_config → %d", thinkingBudget)
		}

		// Check thinking_enabled in nested config
		if thinkingBudget == 0 {
			if enabled, ok := genCfg["thinking_enabled"].(bool); ok && enabled {
				if req.MaxTokens != nil && *req.MaxTokens > 0 {
					thinkingBudget = *req.MaxTokens
				} else {
					thinkingBudget = 4096 // Default medium budget
				}
				log.Printf("[DEBUG] ✅ Using thinking_enabled from nested generation_config → %d", thinkingBudget)
			}
		}

		// Check max_tokens in nested config
		if thinkingBudget == 0 {
			if tokens, ok := genCfg["max_tokens"].(float64); ok && tokens > 0 {
				thinkingBudget = int(tokens)
				log.Printf("[DEBUG] ✅ Using max_tokens from nested generation_config → %d", thinkingBudget)
			}
		}
	}

	// PRIORITY 1: Direct token count (thinking_tokens)
	if req.ThinkingTokens != nil && *req.ThinkingTokens > 0 {
		thinkingBudget = *req.ThinkingTokens
		log.Printf("[DEBUG] ✅ Using direct thinking_tokens: %d", thinkingBudget)
	} else if req.ThinkingEnabled != nil && *req.ThinkingEnabled {
		// PRIORITY 2: Boolean flag (thinking_enabled)
		// Use MaxTokens if available, otherwise use default budget
		if req.MaxTokens != nil && *req.MaxTokens > 0 {
			thinkingBudget = *req.MaxTokens
			log.Printf("[DEBUG] ✅ Using thinking_enabled flag with MaxTokens: %d", thinkingBudget)
		} else {
			thinkingBudget = 4096 // Default medium budget
			log.Printf("[DEBUG] ✅ Using thinking_enabled flag → default budget: %d", thinkingBudget)
		}
	} else if req.ReasoningEffort != "" {
		// PRIORITY 3: Reasoning effort (low/medium/high)
		thinkingBudget = ReasoningEffortToThinkingBudget(req.ReasoningEffort)
		log.Printf("[DEBUG] ✅ Using reasoning_effort '%s' → %d tokens", req.ReasoningEffort, thinkingBudget)
	} else if req.ResponseFormat != nil {
		// PRIORITY 4: Fallback to response_format (deprecated)
		if effort, ok := req.ResponseFormat["reasoning_effort"].(string); ok {
			thinkingBudget = ReasoningEffortToThinkingBudget(effort)
			log.Printf("[DEBUG] ⚠️  Using reasoning_effort from response_format (deprecated): '%s' → %d tokens", effort, thinkingBudget)
		} else if tokens, ok := req.ResponseFormat["thinking_tokens"].(float64); ok {
			thinkingBudget = int(tokens)
			log.Printf("[DEBUG] ⚠️  Using thinking_tokens from response_format (deprecated): %d", thinkingBudget)
		} else if enabled, ok := req.ResponseFormat["thinking_enabled"].(bool); ok && enabled {
			thinkingBudget = 4096
			log.Printf("[DEBUG] ⚠️  Using thinking_enabled from response_format (deprecated): %d", thinkingBudget)
		}
	}

	log.Printf("[DEBUG] Final thinkingBudget: %d", thinkingBudget)

	// HELIXRUN COMPATIBILITY: Check nested generation_config
	// Helixrun sends: {"generation_config": {"thinking_enabled": true}}
	// We need to check this BEFORE applying thinking budget
	if thinkingBudget == 0 && req.ResponseFormat != nil {
		if genCfg, ok := req.ResponseFormat["generation_config"].(map[string]interface{}); ok {
			log.Printf("[DEBUG] 🔧 Found nested generation_config (helixrun style): %+v", genCfg)

			// Check thinking_enabled in nested config
			if enabled, ok := genCfg["thinking_enabled"].(bool); ok && enabled {
				thinkingBudget = 4096 // Default medium budget
				log.Printf("[DEBUG] ✅ Using thinking_enabled from nested generation_config → %d", thinkingBudget)
			}

			// Check thinking_tokens in nested config
			if tokens, ok := genCfg["thinking_tokens"].(float64); ok && tokens > 0 {
				thinkingBudget = int(tokens)
				log.Printf("[DEBUG] ✅ Using thinking_tokens from nested generation_config → %d", thinkingBudget)
			}
		}
	}

	// Apply thinking budget if set
	if thinkingBudget > 0 {
		log.Printf("[DEBUG] Setting up thinking config...")

		// ✅ CORRECT: includeThoughts BINNEN thinkingConfig!
		generationConfig["thinkingConfig"] = map[string]interface{}{
			"thinkingBudget":  thinkingBudget,
			"includeThoughts": true, // ← BINNEN thinkingConfig!
		}

		log.Printf("[DEBUG] ✅ Set thinkingBudget=%d, includeThoughts=true", thinkingBudget)
		log.Printf("[DEBUG] generationConfig now: %+v", generationConfig)
	} else {
		log.Printf("[DEBUG] ⚠️  No thinking enabled (budget=0)")
	}

	log.Printf("[DEBUG] ════════ REASONING EFFORT CHECK END ════════")
	log.Printf("[DEBUG] Final generationConfig: %+v", generationConfig)

	// Handle JSON mode
	if req.ResponseFormat != nil {
		if respType, ok := req.ResponseFormat["type"].(string); ok && respType == "json_object" {
			generationConfig["responseMimeType"] = "application/json"

			// Handle JSON schema if provided
			if schema, ok := req.ResponseFormat["json_schema"].(map[string]interface{}); ok {
				// Gemini expects the schema in a specific format
				if schemaObj, ok := schema["schema"].(map[string]interface{}); ok {
					generationConfig["responseSchema"] = schemaObj
					log.Printf("[DEBUG] Added JSON schema for structured output")
				}
			}
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
// Now with enhanced reasoning content handling and usage metadata
func GeminiResponseToOpenAI(geminiResp map[string]interface{}, model string) map[string]interface{} {
	choices := make([]map[string]interface{}, 0)

	candidates, _ := geminiResp["candidates"].([]interface{})

	// Count reasoning tokens for usage metadata
	reasoningTokenCount := 0

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
					// Estimate reasoning tokens (rough approximation: ~4 chars per token)
					reasoningTokenCount += len(text) / 4
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

		// Add reasoning content (OpenAI format)
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

	// Extract usage from Gemini response
	usage := ExtractUsageFromGeminiResponse(geminiResp, reasoningTokenCount)

	response := map[string]interface{}{
		"id":      "chatcmpl-" + uuid.New().String(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": choices,
	}

	// Add usage if available
	if usage != nil {
		response["usage"] = usage
	}

	return response
}

// ExtractUsageFromGeminiResponse extracts token usage from Gemini response
// Returns nil if usageMetadata is not present in the response
func ExtractUsageFromGeminiResponse(geminiResp map[string]interface{}, reasoningTokenCount int) *models.Usage {
	// Check for usageMetadata in Gemini response
	usageMetadata, hasMetadata := geminiResp["usageMetadata"].(map[string]interface{})

	if !hasMetadata {
		return nil
	}

	usage := &models.Usage{}

	// Extract prompt tokens
	if promptTokens, ok := usageMetadata["promptTokenCount"].(float64); ok {
		usage.PromptTokens = int(promptTokens)
	}

	// Extract completion tokens
	if completionTokens, ok := usageMetadata["candidatesTokenCount"].(float64); ok {
		usage.CompletionTokens = int(completionTokens)
	}

	// Extract total tokens
	if totalTokens, ok := usageMetadata["totalTokenCount"].(float64); ok {
		usage.TotalTokens = int(totalTokens)
	}

	// Add completion_tokens_details if we have reasoning tokens
	if reasoningTokenCount > 0 {
		usage.CompletionTokensDetails = &models.CompletionTokensDetails{
			ReasoningTokens: reasoningTokenCount,
		}
	}

	return usage
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
