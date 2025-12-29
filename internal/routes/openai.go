package routes

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"gcli2apigo/internal/auth"
	"gcli2apigo/internal/client"
	"gcli2apigo/internal/config"
	"gcli2apigo/internal/models"
	"gcli2apigo/internal/transformers"

	"github.com/google/uuid"
)

// isFakeStreamingAllowed checks if a model supports fake streaming
func isFakeStreamingAllowed(modelName string) bool {
	modelName = strings.TrimPrefix(modelName, "models/")

	if strings.HasPrefix(modelName, "gemini-2.5-pro") {
		return true
	}

	if strings.Contains(modelName, "gemini-flash") || strings.Contains(modelName, "gemini-2.5-flash") {
		if strings.Contains(modelName, "flash-image") {
			return false
		}
		return true
	}

	return false
}

// HandleChatCompletions handles OpenAI-compatible chat completions endpoint
func HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if _, err := auth.AuthenticateUser(r); err != nil {
		http.Error(w, `{"error":{"message":"Invalid authentication credentials","type":"invalid_request_error","code":401}}`, http.StatusUnauthorized)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, `{"error":{"message":"Method not allowed","type":"invalid_request_error","code":405}}`, http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":{"message":"Failed to read request body","type":"invalid_request_error","code":400}}`, http.StatusBadRequest)
		return
	}

	var request models.OpenAIChatCompletionRequest
	if err := json.Unmarshal(body, &request); err != nil {
		http.Error(w, `{"error":{"message":"Invalid JSON in request body","type":"invalid_request_error","code":400}}`, http.StatusBadRequest)
		return
	}

	// DEBUG: Log volledige request
	log.Printf("═══════════════════════════════════════════════════")
	log.Printf("📥 INCOMING REQUEST")
	log.Printf("Model: %s", request.Model)
	log.Printf("Stream: %v", request.Stream)

	// Check BOTH locations for reasoning_effort
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("🧠 REASONING/THINKING CHECK:")

	// Check thinking_tokens (direct token count)
	if request.ThinkingTokens != nil {
		log.Printf("  ✅ thinking_tokens: %d", *request.ThinkingTokens)
	} else {
		log.Printf("  ❌ thinking_tokens: not set")
	}

	// Check reasoning_effort (low/medium/high)
	if request.ReasoningEffort != "" {
		log.Printf("  ✅ reasoning_effort: '%s'", request.ReasoningEffort)
	} else {
		log.Printf("  ❌ reasoning_effort: not set")
	}

	// Check response_format fallbacks
	if request.ResponseFormat != nil {
		if effort, ok := request.ResponseFormat["reasoning_effort"].(string); ok {
			log.Printf("  ⚠️  response_format.reasoning_effort: '%s' (deprecated)", effort)
		}
		if tokens, ok := request.ResponseFormat["thinking_tokens"].(float64); ok {
			log.Printf("  ⚠️  response_format.thinking_tokens: %d (deprecated)", int(tokens))
		}
		if len(request.ResponseFormat) > 0 {
			log.Printf("  Response Format: %+v", request.ResponseFormat)
		}
	} else {
		log.Printf("  Response Format: nil")
	}
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("═══════════════════════════════════════════════════")

	// Detect fake stream mode - ALLEEN als expliciet gevraagd via model naam
	modelName := request.Model
	isFakeStream := false

	if strings.HasSuffix(modelName, "-fake") {
		isFakeStream = true
		modelName = strings.TrimSuffix(modelName, "-fake")
		log.Printf("🔄 Fake stream mode DETECTED via model suffix: %s", modelName)
	} else if strings.HasPrefix(modelName, "假流式/") {
		isFakeStream = true
		modelName = strings.TrimPrefix(modelName, "假流式/")
		log.Printf("🔄 Fake stream mode DETECTED via prefix: %s", modelName)
	} else {
		log.Printf("✅ Normal mode - NO fake stream detection")
	}

	if isFakeStream {
		request.Model = modelName
		if !isFakeStreamingAllowed(modelName) {
			errorData := map[string]interface{}{
				"error": map[string]interface{}{
					"message": fmt.Sprintf("Fake streaming not supported for model: %s", modelName),
					"type":    "invalid_request_error",
					"code":    400,
				},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(errorData)
			return
		}
	}

	// Transform OpenAI request to Gemini format
	log.Printf("🔄 Transforming OpenAI → Gemini...")
	geminiRequestData := transformers.OpenAIRequestToGemini(&request)

	// ✅ INLINE PAYLOAD BUILDING - Behoud generationConfig!
	// BuildGeminiPayloadFromOpenAI overschrijft mogelijk de config, dus doen we het inline
	safetySettings := config.DefaultSafetySettings
	if ss, ok := geminiRequestData["safetySettings"]; ok && ss != nil {
		if ssSlice, ok := ss.([]config.SafetySetting); ok {
			safetySettings = ssSlice
		}
	}

	// BELANGRIJK: Haal generationConfig uit transform (bevat thinkingConfig!)
	generationConfig, hasGenConfig := geminiRequestData["generationConfig"].(map[string]interface{})
	if !hasGenConfig || generationConfig == nil {
		generationConfig = make(map[string]interface{})
		log.Printf("[WARN] No generationConfig from transformer!")
	}

	log.Printf("[DEBUG] generationConfig from transformer: %+v", generationConfig)

	requestData := map[string]interface{}{
		"contents":         geminiRequestData["contents"],
		"safetySettings":   safetySettings,
		"generationConfig": generationConfig, // ✅ Bevat thinkingConfig!
	}

	if systemInstruction, ok := geminiRequestData["systemInstruction"]; ok && systemInstruction != nil {
		requestData["systemInstruction"] = systemInstruction
	}
	if cachedContent, ok := geminiRequestData["cachedContent"]; ok && cachedContent != nil {
		requestData["cachedContent"] = cachedContent
	}
	if tools, ok := geminiRequestData["tools"]; ok && tools != nil {
		requestData["tools"] = tools
	}
	if toolConfig, ok := geminiRequestData["toolConfig"]; ok && toolConfig != nil {
		requestData["toolConfig"] = toolConfig
	}

	geminiPayload := map[string]interface{}{
		"model":   geminiRequestData["model"],
		"request": requestData,
	}

	// DEBUG: Check wat er in geminiPayload zit
	if reqData, ok := geminiPayload["request"].(map[string]interface{}); ok {
		if genConfig, ok := reqData["generationConfig"].(map[string]interface{}); ok {
			log.Printf("📋 Generation Config: %+v", genConfig)
			if thinkingConfig, ok := genConfig["thinkingConfig"].(map[string]interface{}); ok {
				log.Printf("🧠 THINKING CONFIG FOUND: %+v", thinkingConfig)
			} else {
				log.Printf("⚠️  NO THINKING CONFIG in generation config")
			}
		}
	}

	// Route to appropriate handler
	if isFakeStream {
		log.Printf("📤 Routing to: FAKE STREAM handler")
		handleFakeStreamChatCompletion(w, r, &request, geminiPayload)
	} else if request.Stream {
		log.Printf("📤 Routing to: TRUE LIVE STREAM handler")
		handleTrueLiveStreamingChatCompletion(w, r, &request, geminiPayload)
	} else {
		log.Printf("📤 Routing to: NON-STREAMING handler")
		handleNonStreamingChatCompletion(w, r, &request, geminiPayload)
	}
}

// handleTrueLiveStreamingChatCompletion implements TRUE LIVE STREAMING with ZERO buffering
func handleTrueLiveStreamingChatCompletion(w http.ResponseWriter, r *http.Request, request *models.OpenAIChatCompletionRequest, geminiPayload map[string]interface{}) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error":{"message":"Streaming not supported","type":"api_error","code":500}}`, http.StatusInternalServerError)
		return
	}

	log.Printf("🚀 Starting TRUE LIVE STREAMING")

	// Send request to Gemini with streaming enabled
	result, err := client.SendGeminiRequest(geminiPayload, true)
	if err != nil {
		log.Printf("❌ Gemini request failed: %v", err)
		errorChunk := map[string]interface{}{
			"error": map[string]interface{}{
				"message": fmt.Sprintf("Request failed: %v", err),
				"type":    "api_error",
				"code":    500,
			},
		}
		jsonData, _ := json.Marshal(errorChunk)
		fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	streamChan, ok := result.(chan string)
	if !ok {
		log.Printf("❌ Invalid stream channel type")
		errorChunk := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "Streaming request failed",
				"type":    "api_error",
				"code":    500,
			},
		}
		jsonData, _ := json.Marshal(errorChunk)
		fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	responseID := "chatcmpl-" + uuid.New().String()
	createdTime := time.Now().Unix()

	log.Printf("✅ Stream channel received, response ID: %s", responseID)

	// Send initial chunk with role
	initialChunk := map[string]interface{}{
		"id":      responseID,
		"object":  "chat.completion.chunk",
		"created": createdTime,
		"model":   request.Model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"delta": map[string]interface{}{
					"role": "assistant",
				},
				"finish_reason": nil,
			},
		},
	}
	jsonData, _ := json.Marshal(initialChunk)
	fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
	flusher.Flush()
	log.Printf("📤 Sent initial chunk with role")

	// Process stream chunks with ZERO buffering
	chunkCount := 0
	var lastFinishReason string
	toolCallsBuffer := make([]map[string]interface{}, 0)
	toolCallsSent := false
	reasoningChunks := 0
	contentChunks := 0

	for geminiChunkStr := range streamChan {
		chunkCount++

		var geminiChunk map[string]interface{}
		if err := json.Unmarshal([]byte(geminiChunkStr), &geminiChunk); err != nil {
			log.Printf("⚠️  Chunk %d: Parse error: %v", chunkCount, err)
			continue
		}

		// DEBUG: Log eerste paar chunks
		if chunkCount <= 3 {
			log.Printf("📦 Chunk %d received: %s", chunkCount, geminiChunkStr[:min(200, len(geminiChunkStr))])
		}

		// Check for errors
		if errObj, ok := geminiChunk["error"]; ok {
			log.Printf("❌ Error in chunk %d: %+v", chunkCount, errObj)
			errorChunk := map[string]interface{}{
				"id":      responseID,
				"object":  "chat.completion.chunk",
				"created": createdTime,
				"model":   request.Model,
				"choices": []map[string]interface{}{
					{
						"index":         0,
						"delta":         map[string]interface{}{},
						"finish_reason": "error",
					},
				},
				"error": errObj,
			}
			jsonData, _ := json.Marshal(errorChunk)
			fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
			flusher.Flush()
			break
		}

		candidates, _ := geminiChunk["candidates"].([]interface{})

		for _, candidate := range candidates {
			candMap, _ := candidate.(map[string]interface{})
			content, _ := candMap["content"].(map[string]interface{})
			parts, _ := content["parts"].([]interface{})

			// Process each part
			for partIdx, part := range parts {
				partMap, _ := part.(map[string]interface{})

				// DEBUG: Log part details in eerste chunks
				if chunkCount <= 3 {
					log.Printf("  Part %d: %+v", partIdx, partMap)
				}

				// Handle tool calls - BUFFER them
				if fnCall, ok := partMap["functionCall"].(map[string]interface{}); ok {
					name, _ := fnCall["name"].(string)
					args, _ := fnCall["args"].(map[string]interface{})
					argsJSON, _ := json.Marshal(args)

					toolCall := map[string]interface{}{
						"index": len(toolCallsBuffer),
						"id":    "call_" + uuid.New().String(),
						"type":  "function",
						"function": map[string]interface{}{
							"name":      name,
							"arguments": string(argsJSON),
						},
					}
					toolCallsBuffer = append(toolCallsBuffer, toolCall)

					log.Printf("🔧 Buffered tool call %d: %s", len(toolCallsBuffer), name)
					continue
				}

				// Handle text content - IMMEDIATE STREAMING
				if text, ok := partMap["text"].(string); ok {
					if text == "" {
						continue
					}

					// Check if this is reasoning/thinking content
					isThought, _ := partMap["thought"].(bool)

					delta := map[string]interface{}{}
					if isThought {
						delta["reasoning_content"] = text
						reasoningChunks++
						log.Printf("🧠 REASONING chunk %d: %s", reasoningChunks, text[:min(50, len(text))])
					} else {
						delta["content"] = text
						contentChunks++
						if contentChunks <= 5 {
							log.Printf("💬 Content chunk %d: %s", contentChunks, text[:min(50, len(text))])
						}
					}

					chunk := map[string]interface{}{
						"id":      responseID,
						"object":  "chat.completion.chunk",
						"created": createdTime,
						"model":   request.Model,
						"choices": []map[string]interface{}{
							{
								"index":         0,
								"delta":         delta,
								"finish_reason": nil,
							},
						},
					}

					jsonData, _ := json.Marshal(chunk)
					fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
					flusher.Flush() // CRITICAL: Flush immediately!
				}
			}

			// Handle finish reason
			if finishReason, ok := candMap["finishReason"].(string); ok && finishReason != "" {
				lastFinishReason = finishReason
				log.Printf("🏁 Finish reason detected: %s", finishReason)
			}
		}
	}

	log.Printf("📊 Stream completed:")
	log.Printf("  - Total chunks: %d", chunkCount)
	log.Printf("  - Content chunks: %d", contentChunks)
	log.Printf("  - Reasoning chunks: %d", reasoningChunks)
	log.Printf("  - Tool calls: %d", len(toolCallsBuffer))

	// Send buffered tool calls if any
	if len(toolCallsBuffer) > 0 && !toolCallsSent {
		toolCallChunk := map[string]interface{}{
			"id":      responseID,
			"object":  "chat.completion.chunk",
			"created": createdTime,
			"model":   request.Model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": toolCallsBuffer,
					},
					"finish_reason": nil,
				},
			},
		}
		jsonData, _ := json.Marshal(toolCallChunk)
		fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		flusher.Flush()
		log.Printf("🔧 Sent %d buffered tool calls", len(toolCallsBuffer))
	}

	// Send finish reason
	if lastFinishReason != "" {
		mappedFinishReason := transformers.MapFinishReason(lastFinishReason)
		if len(toolCallsBuffer) > 0 {
			mappedFinishReason = "tool_calls"
		}

		finishChunk := map[string]interface{}{
			"id":      responseID,
			"object":  "chat.completion.chunk",
			"created": createdTime,
			"model":   request.Model,
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"delta":         map[string]interface{}{},
					"finish_reason": mappedFinishReason,
				},
			},
		}
		jsonData, _ := json.Marshal(finishChunk)
		fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		flusher.Flush()
		log.Printf("🏁 Sent finish_reason: %v", mappedFinishReason)
	}

	// Send [DONE]
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()

	log.Printf("✅ Stream COMPLETE: %s", responseID)
}

// handleFakeStreamChatCompletion handles fake streaming (buffer then stream complete response)
func handleFakeStreamChatCompletion(w http.ResponseWriter, r *http.Request, request *models.OpenAIChatCompletionRequest, geminiPayload map[string]interface{}) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error":{"message":"Streaming not supported","type":"api_error","code":500}}`, http.StatusInternalServerError)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	result, err := client.SendGeminiRequest(geminiPayload, true)
	if err != nil {
		errorData := map[string]interface{}{
			"error": map[string]interface{}{
				"message": fmt.Sprintf("Request failed: %v", err),
				"type":    "api_error",
				"code":    500,
			},
		}
		jsonData, _ := json.Marshal(errorData)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write(jsonData)
		return
	}

	streamChan, ok := result.(chan string)
	if !ok {
		errorData := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "Streaming request failed",
				"type":    "api_error",
				"code":    500,
			},
		}
		jsonData, _ := json.Marshal(errorData)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write(jsonData)
		return
	}

	responseID := "chatcmpl-" + uuid.New().String()

	// Heartbeat
	heartbeatDone := make(chan struct{})
	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				heartbeat := map[string]interface{}{
					"id":      responseID,
					"object":  "chat.completion.chunk",
					"created": time.Now().Unix(),
					"model":   request.Model,
					"choices": []map[string]interface{}{
						{
							"index": 0,
							"delta": map[string]interface{}{
								"role":    "assistant",
								"content": "",
							},
							"finish_reason": nil,
						},
					},
				}
				jsonData, _ := json.Marshal(heartbeat)
				fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
				flusher.Flush()
			case <-heartbeatDone:
				return
			}
		}
	}()
	defer close(heartbeatDone)

	log.Printf("Starting fake stream collection for model: %s", request.Model)

	// Collect all chunks
	allChunks := make([]map[string]interface{}, 0)
	for chunk := range streamChan {
		select {
		case <-ctx.Done():
			return
		default:
		}

		var geminiChunk map[string]interface{}
		if err := json.Unmarshal([]byte(chunk), &geminiChunk); err != nil {
			continue
		}

		if errObj, ok := geminiChunk["error"]; ok {
			errorData := map[string]interface{}{"error": errObj}
			jsonData, _ := json.Marshal(errorData)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			w.Write(jsonData)
			return
		}

		allChunks = append(allChunks, geminiChunk)
	}

	// Merge chunks
	completeResponse := mergeGeminiChunks(allChunks)
	if completeResponse == nil {
		errorData := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "No response data collected",
				"type":    "api_error",
				"code":    500,
			},
		}
		jsonData, _ := json.Marshal(errorData)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write(jsonData)
		return
	}

	// Transform to OpenAI format
	openaiResponse := transformers.GeminiResponseToOpenAI(completeResponse, request.Model)

	// Extract choices
	var choices []map[string]interface{}
	if choicesRaw, ok := openaiResponse["choices"].([]map[string]interface{}); ok {
		choices = choicesRaw
	}

	// Build streaming chunk
	streamingChoices := make([]map[string]interface{}, 0)
	for _, choiceMap := range choices {
		message, _ := choiceMap["message"].(map[string]interface{})
		index, _ := choiceMap["index"].(int)
		finishReason := choiceMap["finish_reason"]

		delta := make(map[string]interface{})
		if content, ok := message["content"].(string); ok {
			delta["content"] = content
		}
		if reasoningContent, ok := message["reasoning_content"].(string); ok {
			delta["reasoning_content"] = reasoningContent
		}

		streamingChoices = append(streamingChoices, map[string]interface{}{
			"index":         index,
			"delta":         delta,
			"finish_reason": finishReason,
		})
	}

	streamChunk := map[string]interface{}{
		"id":      responseID,
		"object":  "chat.completion.chunk",
		"created": openaiResponse["created"],
		"model":   request.Model,
		"choices": streamingChoices,
	}

	jsonData, _ := json.Marshal(streamChunk)
	fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
	flusher.Flush()

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func mergeGeminiChunks(chunks []map[string]interface{}) map[string]interface{} {
	if len(chunks) == 0 {
		return nil
	}

	merged := make(map[string]interface{})
	allParts := make([]interface{}, 0)
	var lastFinishReason string

	for _, chunk := range chunks {
		candidates, _ := chunk["candidates"].([]interface{})
		for _, candidate := range candidates {
			candMap, _ := candidate.(map[string]interface{})
			content, _ := candMap["content"].(map[string]interface{})
			parts, _ := content["parts"].([]interface{})
			allParts = append(allParts, parts...)

			if fr, ok := candMap["finishReason"].(string); ok && fr != "" {
				lastFinishReason = fr
			}
		}
	}

	merged["candidates"] = []interface{}{
		map[string]interface{}{
			"index": 0,
			"content": map[string]interface{}{
				"role":  "model",
				"parts": allParts,
			},
			"finishReason": lastFinishReason,
		},
	}

	return merged
}

func handleNonStreamingChatCompletion(w http.ResponseWriter, r *http.Request, request *models.OpenAIChatCompletionRequest, geminiPayload map[string]interface{}) {
	log.Printf("🔄 Sending non-streaming request to Gemini...")

	result, err := client.SendGeminiRequest(geminiPayload, false)
	if err != nil {
		log.Printf("❌ Gemini request failed: %v", err)
		errorData := map[string]interface{}{
			"error": map[string]interface{}{
				"message": fmt.Sprintf("Request failed: %v", err),
				"type":    "api_error",
				"code":    500,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(errorData)
		return
	}

	geminiResponse, ok := result.(map[string]interface{})
	if !ok {
		log.Printf("❌ Invalid response type from Gemini")
		errorData := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "Invalid response from API",
				"type":    "api_error",
				"code":    500,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(errorData)
		return
	}

	// DEBUG: Log Gemini response
	if candidates, ok := geminiResponse["candidates"].([]interface{}); ok {
		log.Printf("📥 Gemini returned %d candidates", len(candidates))
		if len(candidates) > 0 {
			if cand, ok := candidates[0].(map[string]interface{}); ok {
				if content, ok := cand["content"].(map[string]interface{}); ok {
					if parts, ok := content["parts"].([]interface{}); ok {
						log.Printf("📦 Response has %d parts", len(parts))
						for i, part := range parts {
							if partMap, ok := part.(map[string]interface{}); ok {
								isThought, _ := partMap["thought"].(bool)
								hasText, _ := partMap["text"].(string)
								log.Printf("  Part %d: thought=%v, hasText=%v", i, isThought, hasText != "")
							}
						}
					}
				}
			}
		}
	}

	if errObj, ok := geminiResponse["error"]; ok {
		log.Printf("❌ Gemini returned error: %+v", errObj)
		w.Header().Set("Content-Type", "application/json")
		if errMap, ok := errObj.(map[string]interface{}); ok {
			if code, ok := errMap["code"].(float64); ok {
				w.WriteHeader(int(code))
			} else {
				w.WriteHeader(http.StatusInternalServerError)
			}
		} else {
			w.WriteHeader(http.StatusInternalServerError)
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"error": errObj})
		return
	}

	log.Printf("🔄 Transforming Gemini → OpenAI...")
	openaiResponse := transformers.GeminiResponseToOpenAI(geminiResponse, request.Model)

	// DEBUG: Log OpenAI response
	if choices, ok := openaiResponse["choices"].([]map[string]interface{}); ok && len(choices) > 0 {
		message, _ := choices[0]["message"].(map[string]interface{})
		hasContent := message["content"] != nil
		hasReasoning := message["reasoning_content"] != nil
		log.Printf("✅ OpenAI response: hasContent=%v, hasReasoning=%v", hasContent, hasReasoning)
		if hasReasoning {
			reasoningPreview := message["reasoning_content"].(string)
			log.Printf("🧠 REASONING CONTENT PREVIEW: %s", reasoningPreview[:min(100, len(reasoningPreview))])
		}
	}

	log.Printf("✅ Successfully processed non-streaming response for model: %s", request.Model)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(openaiResponse)
}

// HandleListModels handles OpenAI-compatible models endpoint
func HandleListModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, `{"error":{"message":"Method not allowed","type":"invalid_request_error","code":405}}`, http.StatusMethodNotAllowed)
		return
	}

	log.Println("OpenAI models list requested")

	openaiModels := make([]map[string]interface{}, 0)
	for _, model := range config.SupportedModels {
		modelID := strings.TrimPrefix(model.Name, "models/")

		openaiModels = append(openaiModels, map[string]interface{}{
			"id":       modelID,
			"object":   "model",
			"created":  1677610602,
			"owned_by": "google",
			"permission": []map[string]interface{}{
				{
					"id":                   "modelperm-" + strings.ReplaceAll(modelID, "/", "-"),
					"object":               "model_permission",
					"created":              1677610602,
					"allow_create_engine":  false,
					"allow_sampling":       true,
					"allow_logprobs":       false,
					"allow_search_indices": false,
					"allow_view":           true,
					"allow_fine_tuning":    false,
					"organization":         "*",
					"group":                nil,
					"is_blocking":          false,
				},
			},
			"root":   modelID,
			"parent": nil,
		})

		if isFakeStreamingAllowed(modelID) {
			fakeModelID := config.GetFakeModelName(modelID)
			openaiModels = append(openaiModels, map[string]interface{}{
				"id":       fakeModelID,
				"object":   "model",
				"created":  1677610602,
				"owned_by": "google",
				"permission": []map[string]interface{}{
					{
						"id":                   "modelperm-" + strings.ReplaceAll(fakeModelID, "/", "-"),
						"object":               "model_permission",
						"created":              1677610602,
						"allow_create_engine":  false,
						"allow_sampling":       true,
						"allow_logprobs":       false,
						"allow_search_indices": false,
						"allow_view":           true,
						"allow_fine_tuning":    false,
						"organization":         "*",
						"group":                nil,
						"is_blocking":          false,
					},
				},
				"root":   fakeModelID,
				"parent": nil,
			})
		}
	}

	log.Printf("Returning %d models (including -fake variants)", len(openaiModels))

	response := map[string]interface{}{
		"object": "list",
		"data":   openaiModels,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
