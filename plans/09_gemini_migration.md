# Step 09: Gemini Provider Migration

## Context

This step refactors existing Gemini-specific code into the new provider architecture. Existing code in [`internal/routes/gemini.go`](../internal/routes/gemini.go:1), [`internal/routes/openai.go`](../internal/routes/openai.go:1), [`internal/client/client.go`](../internal/client/client.go:1), and [`internal/transformers/transformers.go`](../internal/transformers/transformers.go:1) will be migrated to implement the Provider interface.

## Objectives

1. Create Gemini provider directory structure
2. Migrate existing client code to new architecture
3. Migrate existing transformer code to new architecture
4. Implement Provider interface for Gemini
5. Maintain backward compatibility with existing routes

## Design Pattern

**Adapter Pattern**: Wrap existing Gemini-specific code to implement the Provider interface without modifying the underlying implementation.

## Files to Create

### 1. `internal/providers/gemini/provider.go`

**Purpose**: Main Gemini provider implementation

**Full Implementation**:

```go
package gemini

import (
    "context"
    "fmt"
    "log"
    
    "gcli2apigo/internal/providers"
    "gcli2apigo/internal/transformers"
)

// GeminiProvider implements Provider interface for Google's Gemini API
type GeminiProvider struct {
    config        providers.ProviderConfig
    client        *GeminiClient
    transformer   *GeminiTransformer
    authenticator providers.Authenticator
    models        []providers.ModelInfo
}

// NewGeminiProvider creates a new Gemini provider
func NewGeminiProvider(cfg providers.ProviderConfig, deps providers.ProviderDependencies) (providers.Provider, error) {
    // Create transformer
    transformer := NewGeminiTransformer()
    
    // Create client with proxy
    client, err := NewGeminiClient(cfg, deps.ProxyManager)
    if err != nil {
        return nil, fmt.Errorf("failed to create Gemini client: %w", err)
    }
    
    provider := &GeminiProvider{
        config:      cfg,
        client:      client,
        transformer: transformer,
        models:      getGeminiModels(),
    }
    
    // Set authenticator if provided
    if deps.AuthManager != nil {
        provider.authenticator = deps.AuthManager.GetAuthenticator("gemini")
    }
    
    return provider, nil
}

// GetType returns the provider type
func (p *GeminiProvider) GetType() providers.ProviderType {
    return providers.ProviderGemini
}

// GetName returns the provider name
func (p *GeminiProvider) GetName() string {
    return "Gemini"
}

// GetVersion returns the provider version
func (p *GeminiProvider) GetVersion() string {
    return "1.0.0"
}

// ListModels returns all available models
func (p *GeminiProvider) ListModels(ctx context.Context) ([]providers.ModelInfo, error) {
    return p.models, nil
}

// ValidateModel checks if a model ID is valid
func (p *GeminiProvider) ValidateModel(modelID string) bool {
    for _, model := range p.models {
        if model.ID == modelID || model.Name == modelID {
            return true
        }
    }
    return false
}

// HandleChatCompletion processes a non-streaming chat completion request
func (p *GeminiProvider) HandleChatCompletion(ctx context.Context, req *providers.ChatCompletionRequest) (*providers.ChatCompletionResponse, error) {
    // Transform request to Gemini format
    geminiReq, err := p.transformer.RequestToProvider(req)
    if err != nil {
        return nil, fmt.Errorf("failed to transform request: %w", err)
    }
    
    // Send request to Gemini API
    geminiResp, err := p.client.SendRequest(ctx, geminiReq, false)
    if err != nil {
        return nil, fmt.Errorf("failed to send request to Gemini: %w", err)
    }
    
    // Transform response to OpenAI format
    openaiResp, err := p.transformer.ResponseToOpenAI(geminiResp, req.Model)
    if err != nil {
        return nil, fmt.Errorf("failed to transform response: %w", err)
    }
    
    return openaiResp, nil
}

// HandleStreamingChatCompletion processes a streaming chat completion request
func (p *GeminiProvider) HandleStreamingChatCompletion(ctx context.Context, req *providers.ChatCompletionRequest) (<-chan string, error) {
    // Transform request to Gemini format
    geminiReq, err := p.transformer.RequestToProvider(req)
    if err != nil {
        return nil, fmt.Errorf("failed to transform request: %w", err)
    }
    
    // Send streaming request to Gemini API
    streamChan, err := p.client.SendRequest(ctx, geminiReq, true)
    if err != nil {
        return nil, fmt.Errorf("failed to send streaming request to Gemini: %w", err)
    }
    
    // Transform streaming chunks to OpenAI format
    openaiChan := make(chan string, 100)
    
    go func() {
        defer close(openaiChan)
        
        for chunk := range streamChan {
            openaiChunk, err := p.transformer.StreamChunkToOpenAI(chunk)
            if err != nil {
                log.Printf("[ERROR] Failed to transform stream chunk: %v", err)
                continue
            }
            
            // Serialize to JSON
            jsonData, err := openaiChunk.ToJSON()
            if err != nil {
                log.Printf("[ERROR] Failed to serialize chunk: %v", err)
                continue
            }
            
            openaiChan <- jsonData
        }
    }()
    
    return openaiChan, nil
}

// GetAuthenticator returns the authenticator
func (p *GeminiProvider) GetAuthenticator() providers.Authenticator {
    return p.authenticator
}

// RequiresAuth indicates if this provider needs authentication
func (p *GeminiProvider) RequiresAuth() bool {
    return true
}

// GetConfig returns the current configuration
func (p *GeminiProvider) GetConfig() providers.ProviderConfig {
    return p.config
}

// UpdateConfig updates the provider's configuration
func (p *GeminiProvider) UpdateConfig(cfg providers.ProviderConfig) error {
    p.config = cfg
    
    // Recreate client with new configuration
    // This would require access to ProxyManager
    // For now, just update config
    
    return nil
}

// HealthCheck performs a health check on the provider
func (p *GeminiProvider) HealthCheck(ctx context.Context) error {
    // Simple health check by listing models
    _, err := p.ListModels(ctx)
    return err
}

// Close performs cleanup operations
func (p *GeminiProvider) Close() error {
    if p.client != nil {
        return p.client.Close()
    }
    return nil
}

// getGeminiModels returns the list of Gemini models
// This data should be moved from internal/config/config.go
func getGeminiModels() []providers.ModelInfo {
    return []providers.ModelInfo{
        {
            ID:                       "gemini-2.5-pro",
            Name:                     "models/gemini-2.5-pro",
            DisplayName:              "Gemini 2.5 Pro",
            Description:              "Stable release of Gemini 2.5 Pro",
            InputTokenLimit:         1048576,
            OutputTokenLimit:        65536,
            SupportedGenerationMethods: []string{"generateContent", "countTokens"},
            Thinking: &providers.ThinkingSupport{
                Min:            128,
                Max:            32768,
                ZeroAllowed:    false,
                DynamicAllowed: true,
            },
        },
        {
            ID:                       "gemini-2.5-flash",
            Name:                     "models/gemini-2.5-flash",
            DisplayName:              "Gemini 2.5 Flash",
            Description:              "Stable version of Gemini 2.5 Flash",
            InputTokenLimit:         1048576,
            OutputTokenLimit:        65536,
            SupportedGenerationMethods: []string{"generateContent", "countTokens"},
            Thinking: &providers.ThinkingSupport{
                Min:            0,
                Max:            24576,
                ZeroAllowed:    true,
                DynamicAllowed: true,
            },
        },
        {
            ID:                       "gemini-2.5-flash-lite",
            Name:                     "models/gemini-2.5-flash-lite",
            DisplayName:              "Gemini 2.5 Flash Lite",
            Description:              "Smallest and most cost effective model",
            InputTokenLimit:         1048576,
            OutputTokenLimit:        65536,
            SupportedGenerationMethods: []string{"generateContent", "countTokens"},
            Thinking: &providers.ThinkingSupport{
                Min:            0,
                Max:            24576,
                ZeroAllowed:    true,
                DynamicAllowed: true,
            },
        },
    }
}
```

### 2. `internal/providers/gemini/client.go`

**Purpose**: Gemini HTTP client (migrated from existing code)

**Full Implementation**:

```go
package gemini

import (
    "bufio"
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net/http"
    "sync"
    "time"
    
    "gcli2apigo/internal/proxy"
)

// GeminiClient handles HTTP requests to Gemini API
type GeminiClient struct {
    config    providers.ProviderConfig
    proxyMgr  proxy.ProxyManager
    httpClient *http.Client
    mu        sync.Mutex
}

// NewGeminiClient creates a new Gemini client
func NewGeminiClient(cfg providers.ProviderConfig, proxyMgr proxy.ProxyManager) (*GeminiClient, error) {
    client := &GeminiClient{
        config:   cfg,
        proxyMgr: proxyMgr,
    }
    
    // Get HTTP client with proxy
    if proxyMgr != nil && cfg.Proxy != nil && cfg.Proxy.Enabled {
        if proxy, err := proxyMgr.GetProxy("gemini"); err == nil {
            if httpClient, err := proxy.GetHTTPClient(); err == nil {
                client.httpClient = httpClient
            } else {
                log.Printf("[WARN] Failed to get proxy HTTP client: %v", err)
            }
        }
    }
    
    // Create default client if proxy not available
    if client.httpClient == nil {
        client.httpClient = &http.Client{
            Timeout: 5 * time.Minute,
        }
    }
    
    return client, nil
}

// SendRequest sends a request to Gemini API
func (c *GeminiClient) SendRequest(ctx context.Context, payload interface{}, isStreaming bool) (interface{}, error) {
    // Build URL
    action := "generateContent"
    if isStreaming {
        action = "streamGenerateContent"
    }
    
    url := fmt.Sprintf("%s/v1internal:%s", c.config.APIEndpoint, action)
    if isStreaming {
        url += "?alt=sse"
    }
    
    // Serialize payload
    buf := getBufferFromPool()
    defer returnBufferToPool(buf)
    
    if err := json.NewEncoder(buf).Encode(payload); err != nil {
        return nil, fmt.Errorf("failed to encode payload: %w", err)
    }
    
    // Create request
    req, err := http.NewRequestWithContext(ctx, "POST", url, buf)
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }
    
    // Set headers
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("User-Agent", "GeminiCLI/0.1.5 (windows; amd64)")
    req.Header.Set("Connection", "keep-alive")
    
    // Send request
    resp, err := c.httpClient.Do(req)
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
    }
    
    // Handle response
    if isStreaming {
        return c.handleStreamingResponse(resp)
    }
    
    return c.handleNonStreamingResponse(resp)
}

// handleStreamingResponse handles streaming SSE response
func (c *GeminiClient) handleStreamingResponse(resp *http.Response) (<-chan string, error) {
    streamChan := make(chan string, 10)
    
    go func() {
        defer close(streamChan)
        defer resp.Body.Close()
        
        reader := bufio.NewReader(resp.Body)
        
        for {
            line, err := reader.ReadString('\n')
            if err != nil {
                if err != io.EOF {
                    log.Printf("[ERROR] Stream read error: %v", err)
                }
                break
            }
            
            line = line[:len(line)-1] // Trim newline
            
            if line == "" {
                continue
            }
            
            // Parse SSE format
            if chunk, found := cutPrefix(line, "data: "); found {
                if chunk == "[DONE]" {
                    break
                }
                streamChan <- chunk
            }
        }
    }()
    
    return streamChan, nil
}

// handleNonStreamingResponse handles non-streaming response
func (c *GeminiClient) handleNonStreamingResponse(resp *http.Response) (map[string]interface{}, error) {
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("failed to read response: %w", err)
    }
    
    var result map[string]interface{}
    if err := json.Unmarshal(body, &result); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }
    
    return result, nil
}

// Close cleans up client resources
func (c *GeminiClient) Close() error {
    // Close idle connections
    if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
        transport.CloseIdleConnections()
    }
    return nil
}

// Buffer pool for JSON encoding

var jsonEncoderPool = sync.Pool{
    New: func() interface{} {
        return &bytes.Buffer{}
    },
}

func getBufferFromPool() *bytes.Buffer {
    return jsonEncoderPool.Get().(*bytes.Buffer)
}

func returnBufferToPool(buf *bytes.Buffer) {
    buf.Reset()
    if buf.Cap() < 64*1024 {
        jsonEncoderPool.Put(buf)
    }
}

func cutPrefix(s, prefix string) (string, bool) {
    if len(s) < len(prefix) {
        return s, false
    }
    return s[len(prefix):], true
}
```

### 3. `internal/providers/gemini/transformer.go`

**Purpose**: Request/response transformer for Gemini (migrated from existing code)

**Full Implementation**:

```go
package gemini

import (
    "encoding/json"
    "fmt"
    "log"
    
    "gcli2apigo/internal/providers"
)

// GeminiTransformer transforms between OpenAI and Gemini formats
type GeminiTransformer struct{}

// NewGeminiTransformer creates a new Gemini transformer
func NewGeminiTransformer() *GeminiTransformer {
    return &GeminiTransformer{}
}

// RequestToProvider transforms OpenAI request to Gemini format
func (t *GeminiTransformer) RequestToProvider(req *providers.ChatCompletionRequest) (interface{}, error) {
    // This implementation should migrate logic from internal/transformers/transformers.go
    // For now, return a simple transformation
    
    geminiReq := map[string]interface{}{
        "model":   req.Model,
        "request": map[string]interface{}{
            "contents": convertMessages(req.Messages),
            "generationConfig": buildGenerationConfig(req),
        },
    }
    
    if req.Tools != nil && len(req.Tools) > 0 {
        geminiReq["request"].(map[string]interface{})["tools"] = convertTools(req.Tools)
    }
    
    return geminiReq, nil
}

// ResponseToOpenAI transforms Gemini response to OpenAI format
func (t *GeminiTransformer) ResponseToOpenAI(resp interface{}, model string) (*providers.ChatCompletionResponse, error) {
    // This implementation should migrate logic from internal/transformers/transformers.go
    // For now, return a simple transformation
    
    geminiResp, ok := resp.(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("invalid response type")
    }
    
    // Extract candidates
    candidates, _ := geminiResp["candidates"].([]interface{})
    if len(candidates) == 0 {
        return nil, fmt.Errorf("no candidates in response")
    }
    
    // Build OpenAI response
    openaiResp := &providers.ChatCompletionResponse{
        ID:      generateID(),
        Object:  "chat.completion",
        Created: time.Now().Unix(),
        Model:   model,
        Choices: []providers.CompletionChoice{
            {
                Index:        0,
                Message:      convertCandidate(candidates[0]),
                FinishReason: mapFinishReason(extractFinishReason(candidates[0])),
            },
        },
    }
    
    return openaiResp, nil
}

// StreamChunkToOpenAI transforms streaming chunk to OpenAI format
func (t *GeminiTransformer) StreamChunkToOpenAI(chunk string) (*providers.ChatCompletionResponse, error) {
    // This implementation should migrate logic from internal/routes/openai.go
    // For now, return a simple transformation
    
    var geminiChunk map[string]interface{}
    if err := json.Unmarshal([]byte(chunk), &geminiChunk); err != nil {
        return nil, fmt.Errorf("failed to parse chunk: %w", err)
    }
    
    // Build OpenAI streaming response
    openaiResp := &providers.ChatCompletionResponse{
        ID:      generateID(),
        Object:  "chat.completion.chunk",
        Created: time.Now().Unix(),
        Model:   "gemini-pro",
        Choices: []providers.CompletionChoice{
            {
                Index: 0,
                Message: providers.ChatMessage{
                    Role:    "assistant",
                    Content: extractContent(geminiChunk),
                },
            },
        },
    }
    
    return openaiResp, nil
}

// ValidateRequest validates an OpenAI request
func (t *GeminiTransformer) ValidateRequest(req *providers.ChatCompletionRequest) error {
    if req.Model == "" {
        return fmt.Errorf("model is required")
    }
    
    if len(req.Messages) == 0 {
        return fmt.Errorf("at least one message is required")
    }
    
    return nil
}

// GetSupportedFeatures returns features supported by Gemini
func (t *GeminiTransformer) GetSupportedFeatures() *transformers.FeatureSupport {
    return &transformers.FeatureSupport{
        Streaming:        true,
        Thinking:         true,
        Tools:            true,
        Images:           true,
        JSONMode:         true,
        SystemPrompt:     true,
        FunctionCalling:  true,
    }
}

// Helper functions

func convertMessages(msgs []providers.ChatMessage) []map[string]interface{} {
    result := make([]map[string]interface{}, len(msgs))
    for i, msg := range msgs {
        result[i] = map[string]interface{}{
            "role":  msg.Role,
            "parts": convertContentToParts(msg.Content),
        }
    }
    return result
}

func convertContentToParts(content interface{}) []map[string]interface{} {
    // Handle different content types
    switch v := content.(type) {
    case string:
        return []map[string]interface{}{
            {"text": v},
        }
    case []interface{}:
        parts := make([]map[string]interface{}, 0)
        for _, item := range v {
            if part, ok := item.(map[string]interface{}); ok {
                parts = append(parts, part)
            }
        }
        return parts
    default:
        return []map[string]interface{}{
            {"text": fmt.Sprintf("%v", content)},
        }
    }
}

func buildGenerationConfig(req *providers.ChatCompletionRequest) map[string]interface{} {
    config := make(map[string]interface{})
    
    if req.Temperature != nil {
        config["temperature"] = *req.Temperature
    }
    if req.MaxTokens != nil {
        config["maxOutputTokens"] = *req.MaxTokens
    }
    if req.TopP != nil {
        config["topP"] = *req.TopP
    }
    
    return config
}

func convertTools(tools []providers.Tool) []map[string]interface{} {
    result := make([]map[string]interface{}, len(tools))
    for i, tool := range tools {
        result[i] = map[string]interface{}{
            "function_declarations": []map[string]interface{}{
                {
                    "name":        tool.Function.Name,
                    "description": tool.Function.Description,
                    "parameters":  tool.Function.Parameters,
                },
            },
        }
    }
    return result
}

func convertCandidate(cand interface{}) providers.ChatMessage {
    candMap, _ := cand.(map[string]interface{})
    content, _ := candMap["content"].(map[string]interface{})
    parts, _ := content["parts"].([]interface{})
    
    var textContent string
    for _, part := range parts {
        if partMap, ok := part.(map[string]interface{}); ok {
            if text, ok := partMap["text"].(string); ok {
                textContent += text
            }
        }
    }
    
    return providers.ChatMessage{
        Role:    "assistant",
        Content: textContent,
    }
}

func extractContent(chunk map[string]interface{}) interface{} {
    candidates, _ := chunk["candidates"].([]interface{})
    if len(candidates) == 0 {
        return ""
    }
    
    cand, _ := candidates[0].(map[string]interface{})
    content, _ := cand["content"].(map[string]interface{})
    parts, _ := content["parts"].([]interface{})
    
    var text string
    for _, part := range parts {
        if partMap, ok := part.(map[string]interface{}); ok {
            if txt, ok := partMap["text"].(string); ok {
                text += txt
            }
        }
    }
    
    return text
}

func extractFinishReason(cand interface{}) *string {
    candMap, _ := cand.(map[string]interface{})
    if reason, ok := candMap["finishReason"].(string); ok {
        return mapFinishReason(reason)
    }
    return nil
}

func mapFinishReason(geminiReason string) *string {
    switch geminiReason {
    case "STOP":
        result := "stop"
        return &result
    case "MAX_TOKENS":
        result := "length"
        return &result
    case "SAFETY", "RECITATION":
        result := "content_filter"
        return &result
    default:
        return nil
    }
}

func generateID() string {
    return fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
}
```

## Dependencies

- **Step 01**: Core Interfaces (Provider interface)
- **Step 02**: Shared Models (request/response types)
- **Step 03**: Proxy Infrastructure (HTTP client)
- **Step 05**: Provider Factory (ProviderDependencies)

## Migration Notes

### Files to Refactor

1. **`internal/client/client.go`** → `internal/providers/gemini/client.go`
   - Move `SendGeminiRequest` logic
   - Move `handleStreamingResponse` logic
   - Move `handleNonStreamingResponse` logic

2. **`internal/transformers/transformers.go`** → `internal/providers/gemini/transformer.go`
   - Move `OpenAIRequestToGemini` logic
   - Move `GeminiResponseToOpenAI` logic
   - Move `ExtractUsageFromGeminiResponse` logic

3. **`internal/routes/openai.go`** → `internal/providers/gemini/transformer.go`
   - Move streaming chunk transformation logic
   - Move `sendStreamChunk` logic

### Backward Compatibility

Keep existing routes working during migration:

- `/v1/chat/completions` → Route to Gemini provider
- `/v1/models` → Route to Gemini provider
- `/v1beta/models` → Route to Gemini provider

## Verification

After completing this step, verify:

1. Gemini provider implements Provider interface
2. Existing functionality is preserved
3. Backward compatibility is maintained
4. Tests pass for all migrated code
5. No breaking changes to API

## Next Steps

After completing this step, proceed to:
- **Step 10**: Copilot Provider Implementation
- **Step 11**: Qwen Provider Implementation
- **Step 12**: Antigravity Provider Implementation
