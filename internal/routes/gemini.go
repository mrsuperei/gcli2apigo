package routes

import (
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
	"gcli2apigo/internal/transformers"

	"github.com/google/uuid"
)

// HandleGeminiListModels handles native Gemini models endpoint
func HandleGeminiListModels(w http.ResponseWriter, r *http.Request) {
	// Authenticate user
	if _, err := auth.AuthenticateUser(r); err != nil {
		http.Error(w, `{"error":{"message":"Invalid authentication credentials","code":401}}`, http.StatusUnauthorized)
		return
	}

	if r.Method != http.MethodGet {
		http.Error(w, `{"error":{"message":"Method not allowed","code":405}}`, http.StatusMethodNotAllowed)
		return
	}

	log.Println("Gemini models list requested")

	// Build models list including fake streaming variants
	allModels := make([]config.Model, 0, len(config.SupportedModels)*2)

	// Add base models
	allModels = append(allModels, config.SupportedModels...)

	// Add fake streaming variants for supported models
	for _, model := range config.SupportedModels {
		modelID := strings.TrimPrefix(model.Name, "models/")
		if isFakeStreamingAllowed(modelID) {
			fakeModelName := config.GetFakeModelName(modelID)
			fakeModel := model
			fakeModel.Name = "models/" + fakeModelName
			fakeModel.DisplayName = fakeModel.DisplayName + " (Fake Streaming)"
			fakeModel.Description = fakeModel.Description + " - Fake streaming variant"
			allModels = append(allModels, fakeModel)
		}
	}

	modelsResponse := map[string]interface{}{
		"models": allModels,
	}

	log.Printf("Returning %d Gemini models (including fake streaming variants)", len(allModels))

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(modelsResponse)
}

// HandleGeminiListModelsV1 handles alternative models endpoint for v1 API version
func HandleGeminiListModelsV1(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/v1/models" {
		HandleGeminiListModels(w, r)
	} else {
		http.NotFound(w, r)
	}
}

// HandleGeminiProxy handles native Gemini API proxy endpoint
func HandleGeminiProxy(w http.ResponseWriter, r *http.Request) {
	// Skip if this is a known route
	if r.URL.Path == "/" || r.URL.Path == "/health" ||
		strings.HasPrefix(r.URL.Path, "/v1/chat/completions") ||
		(r.URL.Path == "/v1/models" && r.Method == http.MethodGet) ||
		(r.URL.Path == "/v1beta/models" && r.Method == http.MethodGet) {
		http.NotFound(w, r)
		return
	}

	// Only handle Gemini API paths
	if !strings.HasPrefix(r.URL.Path, "/v1beta/") && !strings.HasPrefix(r.URL.Path, "/v1/") {
		http.NotFound(w, r)
		return
	}

	// Authenticate user
	if _, err := auth.AuthenticateUser(r); err != nil {
		http.Error(w, `{"error":{"message":"Invalid authentication credentials","code":401}}`, http.StatusUnauthorized)
		return
	}

	// Get the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":{"message":"Failed to read request body","code":400}}`, http.StatusBadRequest)
		return
	}

	// Determine if this is a streaming request
	isStreaming := strings.Contains(strings.ToLower(r.URL.Path), "stream")

	// Extract model name from the path
	modelName := extractModelFromPath(r.URL.Path)

	log.Printf("Gemini proxy request: path=%s, model=%s, stream=%v", r.URL.Path, modelName, isStreaming)

	if modelName == "" {
		log.Printf("Could not extract model name from path: %s", r.URL.Path)
		http.Error(w, fmt.Sprintf(`{"error":{"message":"Could not extract model name from path: %s","code":400}}`, r.URL.Path), http.StatusBadRequest)
		return
	}

	// Detect and handle fake stream mode based on language setting
	isFakeStream := false

	// Check for English format: modelID-fake
	if strings.HasSuffix(modelName, "-fake") {
		isFakeStream = true
		modelName = strings.TrimSuffix(modelName, "-fake")
	} else if strings.HasPrefix(modelName, "假流式/") {
		// Check for Chinese format: 假流式/modelID
		isFakeStream = true
		modelName = strings.TrimPrefix(modelName, "假流式/")
	}

	if isFakeStream {
		log.Printf("Detected fake stream mode in Gemini proxy, stripped model name: %s", modelName)

		// Validate that fake streaming is only allowed for specific models
		if !isFakeStreamingAllowed(modelName) {
			errorData := map[string]interface{}{
				"error": map[string]interface{}{
					"message": fmt.Sprintf("Fake streaming is not supported for model: %s. Only gemini-2.5-pro (and preview models) and gemini flash models (excluding gemini-flash-image) support fake streaming.", modelName),
					"code":    400,
				},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(errorData)
			return
		}

		// Force streaming mode for fake stream
		isStreaming = true
	}

	// Parse the incoming request
	var incomingRequest map[string]interface{}
	if len(body) > 0 {
		if err := json.Unmarshal(body, &incomingRequest); err != nil {
			log.Printf("Invalid JSON in request body: %v", err)
			http.Error(w, `{"error":{"message":"Invalid JSON in request body","code":400}}`, http.StatusBadRequest)
			return
		}
	} else {
		incomingRequest = make(map[string]interface{})
	}

	// Build the payload for Google API
	geminiPayload := client.BuildGeminiPayloadFromNative(incomingRequest, modelName)

	// Send the request to Google API
	result, err := client.SendGeminiRequest(geminiPayload, isStreaming)
	if err != nil {
		log.Printf("Gemini proxy error: %v", err)
		errorData := map[string]interface{}{
			"error": map[string]interface{}{
				"message": fmt.Sprintf("Proxy error: %v", err),
				"code":    500,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(errorData)
		return
	}

	if isStreaming {
		handleGeminiStreamingResponse(w, result)
	} else {
		handleGeminiNonStreamingResponse(w, result, modelName)
	}
}

// handleGeminiStreamingResponse - FIXED: Transform Gemini chunks to OpenAI format with ZERO BUFFERING
func handleGeminiStreamingResponse(w http.ResponseWriter, result interface{}) {
	streamChan, ok := result.(chan string)
	if !ok {
		errorData := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "Streaming request failed",
				"code":    500,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(errorData)
		return
	}

	// Set headers for SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Content-Disposition", "attachment")
	w.Header().Set("Vary", "Origin, X-Origin, Referer")
	w.Header().Set("X-XSS-Protection", "0")
	w.Header().Set("X-Frame-Options", "SAMEORIGIN")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("Server", "ESF")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error":{"message":"Streaming not supported","code":500}}`, http.StatusInternalServerError)
		return
	}

	log.Printf("🚀 [GEMINI-NATIVE] Starting Gemini native streaming with OpenAI format transformation")

	responseID := "chatcmpl-" + uuid.New().String()
	createdTime := time.Now().Unix()

	// Send initial chunk with role
	sendStreamChunk(w, flusher, map[string]interface{}{
		"id":      responseID,
		"object":  "chat.completion.chunk",
		"created": createdTime,
		"model":   "gemini-pro",
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"delta": map[string]interface{}{
					"role": "assistant",
				},
				"finish_reason": nil,
			},
		},
	})
	log.Printf("📤 [GEMINI-NATIVE] Sent initial chunk with role")

	chunkCount := 0
	reasoningChunks := 0
	contentChunks := 0
	toolCallsBuffer := make([]map[string]interface{}, 0)
	var lastFinishReason string

	// CRITICAL FIX: Transform each Gemini chunk to OpenAI format and send immediately
	for chunk := range streamChan {
		chunkCount++

		var geminiChunk map[string]interface{}
		if err := json.Unmarshal([]byte(chunk), &geminiChunk); err != nil {
			log.Printf("⚠️  [GEMINI-NATIVE] Chunk %d: Parse error: %v", chunkCount, err)
			continue
		}

		// DIAGNOSTIC: Log first 5 chunks in detail
		if chunkCount <= 5 {
			log.Printf("📦 [GEMINI-NATIVE] Chunk #%d received (raw): %s", chunkCount, chunk[:min(300, len(chunk))])

			// Analyze chunk structure
			if candidates, ok := geminiChunk["candidates"].([]interface{}); ok {
				log.Printf("  └─ Has 'candidates': YES (count: %d)", len(candidates))
				if len(candidates) > 0 {
					if cand, ok := candidates[0].(map[string]interface{}); ok {
						if content, ok := cand["content"].(map[string]interface{}); ok {
							if parts, ok := content["parts"].([]interface{}); ok {
								log.Printf("  └─ Parts count: %d", len(parts))
							}
						}
					}
				}
			}
		}

		// Check for errors
		if errObj, ok := geminiChunk["error"]; ok {
			log.Printf("❌ Error in chunk %d: %+v", chunkCount, errObj)
			sendStreamChunk(w, flusher, map[string]interface{}{
				"id":      responseID,
				"object":  "chat.completion.chunk",
				"created": createdTime,
				"model":   "gemini-pro",
				"choices": []map[string]interface{}{
					{
						"index":         0,
						"delta":         map[string]interface{}{},
						"finish_reason": "error",
					},
				},
				"error": errObj,
			})
			break
		}

		candidates, _ := geminiChunk["candidates"].([]interface{})

		for _, candidate := range candidates {
			candMap, _ := candidate.(map[string]interface{})
			content, _ := candMap["content"].(map[string]interface{})
			parts, _ := content["parts"].([]interface{})

			// Process each part IMMEDIATELY
			for partIdx, part := range parts {
				partMap, _ := part.(map[string]interface{})

				// Log part details in first chunks
				if chunkCount <= 3 {
					log.Printf("  Part %d: %+v", partIdx, partMap)
				}

				// Handle tool calls - BUFFER them (required for valid JSON)
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

					log.Printf("🔧 [GEMINI-NATIVE] Buffered tool call %d: %s", len(toolCallsBuffer), name)
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
						if reasoningChunks <= 5 {
							log.Printf("🧠 [GEMINI-NATIVE] REASONING chunk %d: %s", reasoningChunks, text[:min(50, len(text))])
						}
					} else {
						delta["content"] = text
						contentChunks++
						if contentChunks <= 5 {
							log.Printf("💬 [GEMINI-NATIVE] Content chunk %d: %s", contentChunks, text[:min(50, len(text))])
						}
					}

					// DIAGNOSTIC: Log chunk being sent
					if reasoningChunks+contentChunks <= 10 {
						log.Printf("📤 [GEMINI-NATIVE] Sending OpenAI chunk #%d: type=%s, text='%s'",
							reasoningChunks+contentChunks,
							map[bool]string{true: "reasoning", false: "content"}[isThought],
							text[:min(50, len(text))])
					}

					// CRITICAL: Send chunk IMMEDIATELY and FLUSH
					sendStreamChunk(w, flusher, map[string]interface{}{
						"id":      responseID,
						"object":  "chat.completion.chunk",
						"created": createdTime,
						"model":   "gemini-pro",
						"choices": []map[string]interface{}{
							{
								"index":         0,
								"delta":         delta,
								"finish_reason": nil,
							},
						},
					})
				}
			}

			// Handle finish reason
			if finishReason, ok := candMap["finishReason"].(string); ok && finishReason != "" {
				lastFinishReason = finishReason
				log.Printf("🏁 [GEMINI-NATIVE] Finish reason detected: %s", finishReason)
			}
		}
	}

	log.Printf("📊 [GEMINI-NATIVE] Stream processing completed:")
	log.Printf("  - Total Gemini chunks received: %d", chunkCount)
	log.Printf("  - Content chunks sent: %d", contentChunks)
	log.Printf("  - Reasoning chunks sent: %d", reasoningChunks)
	log.Printf("  - Tool calls buffered: %d", len(toolCallsBuffer))
	log.Printf("  - Total OpenAI chunks sent: %d", contentChunks+reasoningChunks)

	// Send buffered tool calls if any
	if len(toolCallsBuffer) > 0 {
		sendStreamChunk(w, flusher, map[string]interface{}{
			"id":      responseID,
			"object":  "chat.completion.chunk",
			"created": createdTime,
			"model":   "gemini-pro",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": toolCallsBuffer,
					},
					"finish_reason": nil,
				},
			},
		})
		log.Printf("🔧 [GEMINI-NATIVE] Sent %d buffered tool calls", len(toolCallsBuffer))
	}

	// Send finish reason
	if lastFinishReason != "" {
		mappedFinishReason := transformers.MapFinishReason(lastFinishReason)
		if len(toolCallsBuffer) > 0 {
			mappedFinishReason = "tool_calls"
		}

		sendStreamChunk(w, flusher, map[string]interface{}{
			"id":      responseID,
			"object":  "chat.completion.chunk",
			"created": createdTime,
			"model":   "gemini-pro",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"delta":         map[string]interface{}{},
					"finish_reason": mappedFinishReason,
				},
			},
		})
		log.Printf("🏁 [GEMINI-NATIVE] Sent finish_reason: %v", mappedFinishReason)
	}

	// Send [DONE]
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()

	log.Printf("✅ [GEMINI-NATIVE] Stream COMPLETE: %s", responseID)
}

func handleGeminiNonStreamingResponse(w http.ResponseWriter, result interface{}, modelName string) {
	geminiResponse, ok := result.(map[string]interface{})
	if !ok {
		errorData := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "Invalid response from API",
				"code":    500,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(errorData)
		return
	}

	// Check for error in response
	if errObj, ok := geminiResponse["error"]; ok {
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

	log.Printf("Successfully processed Gemini request for model: %s", modelName)

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(geminiResponse)
}

func extractModelFromPath(path string) string {
	parts := strings.Split(path, "/")

	// Look for the pattern: .../models/{model_name}/...
	for i, part := range parts {
		if part == "models" && i+1 < len(parts) {
			modelName := parts[i+1]
			// Remove any action suffix like ":streamGenerateContent" or ":generateContent"
			if idx := strings.Index(modelName, ":"); idx != -1 {
				modelName = modelName[:idx]
			}
			return modelName
		}
	}

	return ""
}
