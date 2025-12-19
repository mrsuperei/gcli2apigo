package routes

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"gcli2apigo/internal/auth"
	"gcli2apigo/internal/client"
	"gcli2apigo/internal/config"
	"gcli2apigo/internal/models"
	"gcli2apigo/internal/transformers"

	"github.com/google/uuid"
)

// ChunkAccumulator accumulates streaming chunks with size checking
type ChunkAccumulator struct {
	chunks      []map[string]interface{}
	mu          sync.Mutex
	maxSize     int64
	currentSize int64
}

// NewChunkAccumulator creates a new ChunkAccumulator with the specified max size
func NewChunkAccumulator(maxSize int64) *ChunkAccumulator {
	return &ChunkAccumulator{
		chunks:      make([]map[string]interface{}, 0),
		maxSize:     maxSize,
		currentSize: 0,
	}
}

// Add adds a chunk to the accumulator with size checking
func (ca *ChunkAccumulator) Add(chunk map[string]interface{}) error {
	ca.mu.Lock()
	defer ca.mu.Unlock()

	// Estimate chunk size by marshaling to JSON
	chunkBytes, err := json.Marshal(chunk)
	if err != nil {
		return fmt.Errorf("failed to marshal chunk: %v", err)
	}

	chunkSize := int64(len(chunkBytes))

	// Check if adding this chunk would exceed the size limit
	if ca.currentSize+chunkSize > ca.maxSize {
		return fmt.Errorf("accumulated size would exceed limit: current=%d bytes, chunk=%d bytes, limit=%d bytes",
			ca.currentSize, chunkSize, ca.maxSize)
	}

	ca.chunks = append(ca.chunks, chunk)
	ca.currentSize += chunkSize

	return nil
}

// GetComplete merges all accumulated chunks into a complete response
func (ca *ChunkAccumulator) GetComplete() map[string]interface{} {
	ca.mu.Lock()
	defer ca.mu.Unlock()

	if len(ca.chunks) == 0 {
		return nil
	}

	return ca.mergeChunks()
}

// Size returns the current accumulated size in bytes
func (ca *ChunkAccumulator) Size() int64 {
	ca.mu.Lock()
	defer ca.mu.Unlock()
	return ca.currentSize
}

// mergeChunks merges all accumulated chunks into a single response
func (ca *ChunkAccumulator) mergeChunks() map[string]interface{} {
	if len(ca.chunks) == 0 {
		return nil
	}

	// Start with the first chunk as the base
	merged := make(map[string]interface{})

	// Copy top-level fields from the first chunk
	firstChunk := ca.chunks[0]
	for key, value := range firstChunk {
		merged[key] = value
	}

	// Merge all candidates from all chunks
	allCandidates := make([]map[string]interface{}, 0)

	for _, chunk := range ca.chunks {
		if candidates, ok := chunk["candidates"].([]interface{}); ok {
			for _, candidate := range candidates {
				if candMap, ok := candidate.(map[string]interface{}); ok {
					allCandidates = append(allCandidates, candMap)
				}
			}
		}
	}

	// Group candidates by index and merge their content
	candidatesByIndex := make(map[int][]map[string]interface{})

	for _, candidate := range allCandidates {
		index := 0
		if idx, ok := candidate["index"].(float64); ok {
			index = int(idx)
		}
		candidatesByIndex[index] = append(candidatesByIndex[index], candidate)
	}

	// Merge each candidate group
	mergedCandidates := make([]interface{}, 0)

	for index := 0; index < len(candidatesByIndex); index++ {
		candidates := candidatesByIndex[index]
		if len(candidates) == 0 {
			continue
		}

		// Merge content parts from all chunks for this candidate
		var contentParts []interface{}
		var reasoningParts []string
		var finalFinishReason string

		for _, candidate := range candidates {
			// Extract content parts
			if content, ok := candidate["content"].(map[string]interface{}); ok {
				if parts, ok := content["parts"].([]interface{}); ok {
					for _, part := range parts {
						if partMap, ok := part.(map[string]interface{}); ok {
							// Check if this is a thinking token
							if thought, ok := partMap["thought"].(bool); ok && thought {
								if text, ok := partMap["text"].(string); ok {
									reasoningParts = append(reasoningParts, text)
								}
							} else {
								// Regular content part
								contentParts = append(contentParts, part)
							}
						}
					}
				}
			}

			// Use the last non-empty finish reason
			if finishReason, ok := candidate["finishReason"].(string); ok && finishReason != "" {
				finalFinishReason = finishReason
			}
		}

		// Build merged candidate
		mergedCandidate := map[string]interface{}{
			"index": index,
			"content": map[string]interface{}{
				"role":  "model",
				"parts": contentParts,
			},
		}

		// Add reasoning parts if present
		if len(reasoningParts) > 0 {
			// Add reasoning as a separate part with thought flag
			reasoningText := strings.Join(reasoningParts, "")
			if reasoningText != "" {
				// Add to content parts with thought flag
				existingParts := mergedCandidate["content"].(map[string]interface{})["parts"].([]interface{})
				reasoningPart := map[string]interface{}{
					"text":    reasoningText,
					"thought": true,
				}
				existingParts = append(existingParts, reasoningPart)
				mergedCandidate["content"].(map[string]interface{})["parts"] = existingParts
			}
		}

		// Add finish reason if present
		if finalFinishReason != "" {
			mergedCandidate["finishReason"] = finalFinishReason
		}

		mergedCandidates = append(mergedCandidates, mergedCandidate)
	}

	// Set the merged candidates
	merged["candidates"] = mergedCandidates

	return merged
}

// isFakeStreamingAllowed checks if a model supports fake streaming
// Only gemini-2.5-pro (and its preview models) and gemini flash models (excluding gemini-flash-image) are allowed
func isFakeStreamingAllowed(modelName string) bool {
	// Remove "models/" prefix if present
	modelName = strings.TrimPrefix(modelName, "models/")

	// Allow gemini-2.5-pro and its preview models
	if strings.HasPrefix(modelName, "gemini-2.5-pro") {
		return true
	}

	// Allow gemini flash models, but exclude gemini-flash-image models
	if strings.Contains(modelName, "gemini-flash") || strings.Contains(modelName, "gemini-2.5-flash") {
		// Exclude gemini-flash-image models
		if strings.Contains(modelName, "flash-image") {
			return false
		}
		return true
	}

	return false
}

// HandleChatCompletions handles OpenAI-compatible chat completions endpoint
func HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	// Authenticate user
	if _, err := auth.AuthenticateUser(r); err != nil {
		http.Error(w, `{"error":{"message":"Invalid authentication credentials","type":"invalid_request_error","code":401}}`, http.StatusUnauthorized)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, `{"error":{"message":"Method not allowed","type":"invalid_request_error","code":405}}`, http.StatusMethodNotAllowed)
		return
	}

	// Parse request
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

	log.Printf("OpenAI chat completion request: model=%s, stream=%v", request.Model, request.Stream)

	// Detect and handle fake stream mode based on language setting
	modelName := request.Model
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
		request.Model = modelName
		log.Printf("Detected fake stream mode, stripped model name: %s", modelName)

		// Validate that fake streaming is only allowed for specific models
		if !isFakeStreamingAllowed(modelName) {
			errorData := map[string]interface{}{
				"error": map[string]interface{}{
					"message": fmt.Sprintf("Fake streaming is not supported for model: %s. Only gemini-2.5-pro (and preview models) and gemini flash models (excluding gemini-flash-image) support fake streaming.", modelName),
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
	geminiRequestData := transformers.OpenAIRequestToGemini(&request)

	// Build the payload for Google API
	geminiPayload := client.BuildGeminiPayloadFromOpenAI(geminiRequestData)

	// Route to appropriate handler
	if isFakeStream {
		// Force fake stream handler regardless of stream parameter
		handleFakeStreamChatCompletion(w, r, &request, geminiPayload)
	} else if request.Stream {
		handleStreamingChatCompletion(w, r, &request, geminiPayload)
	} else {
		handleNonStreamingChatCompletion(w, r, &request, geminiPayload)
	}
}

func handleFakeStreamChatCompletion(w http.ResponseWriter, r *http.Request, request *models.OpenAIChatCompletionRequest, geminiPayload map[string]interface{}) {
	// Set SSE headers since we'll return the response as streaming chunks
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error":{"message":"Streaming not supported","type":"api_error","code":500}}`, http.StatusInternalServerError)
		return
	}

	// Create context with timeout for chunk collection (5 minutes)
	const collectionTimeout = 5 * time.Minute
	ctx, cancel := context.WithTimeout(r.Context(), collectionTimeout)
	defer cancel()

	// Force streaming mode for internal API request
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

	// Receive streaming channel from client layer
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

	// Create chunk accumulator with 10 MB limit
	const maxSize = 10 * 1024 * 1024 // 10 MB
	accumulator := NewChunkAccumulator(maxSize)

	// Start heartbeat sender to keep connection alive during collection
	const heartbeatInterval = 3 * time.Second
	responseID := "chatcmpl-" + uuid.New().String()
	heartbeatDone := make(chan struct{})

	go func() {
		ticker := time.NewTicker(heartbeatInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				// Send heartbeat chunk with empty content
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

	// Loop through streaming channel and collect chunks
	for chunk := range streamChan {
		// Check for client disconnect or timeout
		select {
		case <-ctx.Done():
			if ctx.Err() == context.DeadlineExceeded {
				log.Printf("Timeout during fake stream collection after %v, cleaned up resources", collectionTimeout)
				errorData := map[string]interface{}{
					"error": map[string]interface{}{
						"message": fmt.Sprintf("Request timeout: chunk collection exceeded %v", collectionTimeout),
						"type":    "timeout_error",
						"code":    504,
					},
				}
				jsonData, _ := json.Marshal(errorData)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusGatewayTimeout)
				w.Write(jsonData)
			} else {
				log.Printf("Client disconnected during fake stream collection, cleaned up resources")
			}
			return
		default:
		}

		var geminiChunk map[string]interface{}
		if err := json.Unmarshal([]byte(chunk), &geminiChunk); err != nil {
			log.Printf("Failed to unmarshal chunk: %v", err)
			continue
		}

		// Check for error chunks and abort if found
		if errObj, ok := geminiChunk["error"]; ok {
			errorData := map[string]interface{}{
				"error": errObj,
			}
			jsonData, _ := json.Marshal(errorData)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			w.Write(jsonData)
			return
		}

		// Add chunk to accumulator
		if err := accumulator.Add(geminiChunk); err != nil {
			log.Printf("Failed to add chunk to accumulator: %v", err)
			errorData := map[string]interface{}{
				"error": map[string]interface{}{
					"message": fmt.Sprintf("Response too large: %v", err),
					"type":    "api_error",
					"code":    413,
				},
			}
			jsonData, _ := json.Marshal(errorData)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusRequestEntityTooLarge)
			w.Write(jsonData)
			return
		}
	}

	log.Printf("Completed fake stream collection, accumulated size: %d bytes", accumulator.Size())

	// Get complete response from accumulator
	completeResponse := accumulator.GetComplete()
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

	// Transform to OpenAI non-streaming format first
	openaiResponse := transformers.GeminiResponseToOpenAI(completeResponse, request.Model)

	log.Printf("Successfully processed fake stream response for model: %s", request.Model)

	// Convert the complete response to a single streaming chunk and send via SSE
	// Use the same responseID that was used for heartbeats

	// Extract choices from the complete response
	var choices []map[string]interface{}
	if choicesRaw, ok := openaiResponse["choices"].([]map[string]interface{}); ok {
		choices = choicesRaw
	} else if choicesInterface, ok := openaiResponse["choices"].([]interface{}); ok {
		// Convert []interface{} to []map[string]interface{}
		for _, c := range choicesInterface {
			if cMap, ok := c.(map[string]interface{}); ok {
				choices = append(choices, cMap)
			}
		}
	}

	// Build streaming choices with all content in deltas
	streamingChoices := make([]map[string]interface{}, 0)
	for _, choiceMap := range choices {
		message, _ := choiceMap["message"].(map[string]interface{})
		index, _ := choiceMap["index"].(int)
		finishReason := choiceMap["finish_reason"]

		// Build delta from message - this contains ALL the content
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

	// Create a single streaming chunk with all content
	streamChunk := map[string]interface{}{
		"id":      responseID,
		"object":  "chat.completion.chunk",
		"created": openaiResponse["created"],
		"model":   request.Model,
		"choices": streamingChoices,
	}

	// Send as single SSE event
	jsonData, _ := json.Marshal(streamChunk)
	fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
	flusher.Flush()

	// Send the final [DONE] marker
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func handleStreamingChatCompletion(w http.ResponseWriter, r *http.Request, request *models.OpenAIChatCompletionRequest, geminiPayload map[string]interface{}) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error":{"message":"Streaming not supported","type":"api_error","code":500}}`, http.StatusInternalServerError)
		return
	}

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
		fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
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
		fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	responseID := "chatcmpl-" + uuid.New().String()
	log.Printf("Starting streaming response: %s", responseID)

	// State tracking - zoals CLIProxyAPI-Extended
	var textBuffer strings.Builder
	textBuffer.Grow(2048)

	// Tool call accumulator: index -> accumulated data
	toolCallAccumulator := make(map[int]*ToolCallAccumulator)
	hasToolCalls := false
	toolCallsFlushed := false // Track if we already flushed tool calls
	finishReasonSent := false // Track if we already sent finish_reason
	lastFlush := time.Now()

	const (
		maxBufferTime = 20 * time.Millisecond
		maxBufferSize = 2048
	)

	// Helper to flush text
	flushTextBuffer := func() {
		if textBuffer.Len() == 0 {
			return
		}

		chunk := map[string]interface{}{
			"id":      responseID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   request.Model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"content": textBuffer.String(),
					},
					"finish_reason": nil,
				},
			},
		}

		jsonData, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		flusher.Flush()

		textBuffer.Reset()
		lastFlush = time.Now()
	}

	// Helper to flush accumulated tool calls
	flushToolCalls := func() {
		if len(toolCallAccumulator) == 0 || toolCallsFlushed {
			return // Skip if empty or already flushed
		}

		// Convert to sorted list by index
		indices := make([]int, 0, len(toolCallAccumulator))
		for idx := range toolCallAccumulator {
			indices = append(indices, idx)
		}
		sort.Ints(indices)

		toolCalls := make([]map[string]interface{}, 0, len(indices))
		for _, idx := range indices {
			acc := toolCallAccumulator[idx]
			if acc.Name == "" {
				continue // Skip incomplete tool calls
			}

			toolCalls = append(toolCalls, map[string]interface{}{
				"index": idx,
				"id":    acc.ID,
				"type":  "function",
				"function": map[string]interface{}{
					"name":      acc.Name,
					"arguments": acc.Arguments,
				},
			})
		}

		if len(toolCalls) == 0 {
			return
		}

		chunk := map[string]interface{}{
			"id":      responseID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   request.Model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": toolCalls,
					},
					"finish_reason": nil,
				},
			},
		}

		jsonData, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
		flusher.Flush()

		toolCallsFlushed = true // Mark as flushed

		if config.IsDebugEnabled() {
			log.Printf("[DEBUG] Flushed %d tool call(s) (will not flush again)", len(toolCalls))
			for _, tc := range toolCalls {
				fn := tc["function"].(map[string]interface{})
				log.Printf("[DEBUG]   - %s: %s", fn["name"], fn["arguments"])
			}
		}
	}

	// Process streaming chunks
	for chunk := range streamChan {
		var geminiChunk map[string]interface{}
		if err := json.Unmarshal([]byte(chunk), &geminiChunk); err != nil {
			continue
		}

		// Handle errors
		if errObj, ok := geminiChunk["error"]; ok {
			flushTextBuffer()
			flushToolCalls()
			errorData := map[string]interface{}{"error": errObj}
			jsonData, _ := json.Marshal(errorData)
			fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
			flusher.Flush()
			break
		}

		candidates, _ := geminiChunk["candidates"].([]interface{})
		hasFinish := false

		for _, candidate := range candidates {
			candMap, _ := candidate.(map[string]interface{})
			content, _ := candMap["content"].(map[string]interface{})
			parts, _ := content["parts"].([]interface{})

			candidateIndex := 0
			if idx, ok := candMap["index"].(float64); ok {
				candidateIndex = int(idx)
			}

			for _, part := range parts {
				partMap, _ := part.(map[string]interface{})

				// Handle function calls - accumulate per index
				if fnCall, ok := partMap["functionCall"].(map[string]interface{}); ok {
					flushTextBuffer() // Flush text before tool calls
					hasToolCalls = true

					name, _ := fnCall["name"].(string)
					args, _ := fnCall["args"].(map[string]interface{})
					argsJSON, _ := json.Marshal(args)

					// Get or create accumulator for this index
					if _, exists := toolCallAccumulator[candidateIndex]; !exists {
						toolCallAccumulator[candidateIndex] = &ToolCallAccumulator{
							Index: candidateIndex,
							ID:    "call_" + uuid.New().String(),
						}
					}

					acc := toolCallAccumulator[candidateIndex]

					// Set name (should be consistent across chunks)
					if acc.Name == "" {
						acc.Name = name
					} else if acc.Name != name {
						// Name changed - this shouldn't happen but log it
						if config.IsDebugEnabled() {
							log.Printf("[WARN] Tool call name changed from %s to %s at index %d", acc.Name, name, candidateIndex)
						}
					}

					// Accumulate arguments (replace, don't append)
					acc.Arguments = string(argsJSON)

					if config.IsDebugEnabled() {
						log.Printf("[DEBUG] Accumulated tool call chunk: index=%d, name=%s, args_len=%d",
							candidateIndex, name, len(acc.Arguments))
					}
					continue
				}

				// Handle text
				if text, ok := partMap["text"].(string); ok {
					if thought, _ := partMap["thought"].(bool); !thought {
						textBuffer.WriteString(text)
					}
				}
			}

			// Handle finish reason
			if finishReason, ok := candMap["finishReason"].(string); ok && finishReason != "" && !finishReasonSent {
				hasFinish = true
				finishReasonSent = true // Mark as sent to prevent duplicates

				flushTextBuffer()

				// Flush tool calls before finish
				if len(toolCallAccumulator) > 0 {
					flushToolCalls()
				}

				mappedFinishReason := transformers.MapFinishReason(finishReason)
				if hasToolCalls {
					mappedFinishReason = "tool_calls"
				}

				finishChunk := map[string]interface{}{
					"id":      responseID,
					"object":  "chat.completion.chunk",
					"created": time.Now().Unix(),
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

				if config.IsDebugEnabled() {
					log.Printf("[DEBUG] Sent finish_reason: %s (will not send again)", mappedFinishReason)
				}
			}
		}

		// Periodic text flush
		if !hasFinish && (time.Since(lastFlush) >= maxBufferTime || textBuffer.Len() >= maxBufferSize) {
			flushTextBuffer()
		}
	}

	// Final cleanup
	flushTextBuffer()
	if len(toolCallAccumulator) > 0 {
		if config.IsDebugEnabled() {
			log.Printf("[WARN] Flushing remaining tool calls without finish_reason")
		}
		flushToolCalls()
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
	log.Printf("Completed streaming response: %s", responseID)
}

// ToolCallAccumulator accumulates tool call data across multiple chunks
type ToolCallAccumulator struct {
	Index     int
	ID        string
	Name      string
	Arguments string
}

func handleNonStreamingChatCompletion(w http.ResponseWriter, r *http.Request, request *models.OpenAIChatCompletionRequest, geminiPayload map[string]interface{}) {
	// Send request to Gemini API
	result, err := client.SendGeminiRequest(geminiPayload, false)
	if err != nil {
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

	// Transform to OpenAI format
	openaiResponse := transformers.GeminiResponseToOpenAI(geminiResponse, request.Model)

	log.Printf("Successfully processed non-streaming response for model: %s", request.Model)

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

	// Convert Gemini models to OpenAI format
	openaiModels := make([]map[string]interface{}, 0)
	for _, model := range config.SupportedModels {
		modelID := strings.TrimPrefix(model.Name, "models/")

		// Add base model
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

		// Add fake streaming variant only for models that support it
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
