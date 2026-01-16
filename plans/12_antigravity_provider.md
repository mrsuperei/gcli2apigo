# Step 12: Antigravity Provider Implementation

## Context

This step implements an Antigravity provider following the Provider interface. Antigravity is a hypothetical AI provider for demonstration purposes, showing how to extend the platform with new providers.

## Objectives

1. Create Antigravity provider directory structure
2. Implement Antigravity HTTP client
3. Implement request/response transformers
4. Implement Provider interface for Antigravity
5. Configure Antigravity-specific authentication

## Design Pattern

**Strategy Pattern**: Antigravity provider implements Provider interface, encapsulating all Antigravity-specific logic.

## Files to Create

### 1. `internal/providers/antigravity/provider.go`

**Purpose**: Main Antigravity provider implementation

**Full Implementation**:

```go
package antigravity

import (
    "context"
    "fmt"
    
    "gcli2apigo/internal/providers"
    "gcli2apigo/internal/transformers"
)

// AntigravityProvider implements Provider interface for Antigravity API
type AntigravityProvider struct {
    config        providers.ProviderConfig
    client        *AntigravityClient
    transformer   *AntigravityTransformer
    authenticator providers.Authenticator
    models        []providers.ModelInfo
}

// NewAntigravityProvider creates a new Antigravity provider
func NewAntigravityProvider(cfg providers.ProviderConfig, deps providers.ProviderDependencies) (providers.Provider, error) {
    // Create transformer
    transformer := NewAntigravityTransformer()
    
    // Create client with proxy
    client, err := NewAntigravityClient(cfg, deps.ProxyManager)
    if err != nil {
        return nil, fmt.Errorf("failed to create Antigravity client: %w", err)
    }
    
    provider := &AntigravityProvider{
        config:      cfg,
        client:      client,
        transformer: transformer,
        models:      getAntigravityModels(),
    }
    
    // Set authenticator if provided
    if deps.AuthManager != nil {
        provider.authenticator = deps.AuthManager.GetAuthenticator("antigravity")
    }
    
    return provider, nil
}

// GetType returns the provider type
func (p *AntigravityProvider) GetType() providers.ProviderType {
    return providers.ProviderAntigravity
}

// GetName returns the provider name
func (p *AntigravityProvider) GetName() string {
    return "Antigravity"
}

// GetVersion returns the provider version
func (p *AntigravityProvider) GetVersion() string {
    return "1.0.0"
}

// ListModels returns all available models
func (p *AntigravityProvider) ListModels(ctx context.Context) ([]providers.ModelInfo, error) {
    return p.models, nil
}

// ValidateModel checks if a model ID is valid
func (p *AntigravityProvider) ValidateModel(modelID string) bool {
    for _, model := range p.models {
        if model.ID == modelID || model.Name == modelID {
            return true
        }
    }
    return false
}

// HandleChatCompletion processes a non-streaming chat completion request
func (p *AntigravityProvider) HandleChatCompletion(ctx context.Context, req *providers.ChatCompletionRequest) (*providers.ChatCompletionResponse, error) {
    // Transform request to Antigravity format
    antigravityReq, err := p.transformer.RequestToProvider(req)
    if err != nil {
        return nil, fmt.Errorf("failed to transform request: %w", err)
    }
    
    // Send request to Antigravity API
    antigravityResp, err := p.client.SendRequest(ctx, antigravityReq, false)
    if err != nil {
        return nil, fmt.Errorf("failed to send request to Antigravity: %w", err)
    }
    
    // Transform response to OpenAI format
    openaiResp, err := p.transformer.ResponseToOpenAI(antigravityResp, req.Model)
    if err != nil {
        return nil, fmt.Errorf("failed to transform response: %w", err)
    }
    
    return openaiResp, nil
}

// HandleStreamingChatCompletion processes a streaming chat completion request
func (p *AntigravityProvider) HandleStreamingChatCompletion(ctx context.Context, req *providers.ChatCompletionRequest) (<-chan string, error) {
    // Transform request to Antigravity format
    antigravityReq, err := p.transformer.RequestToProvider(req)
    if err != nil {
        return nil, fmt.Errorf("failed to transform request: %w", err)
    }
    
    // Send streaming request to Antigravity API
    streamChan, err := p.client.SendRequest(ctx, antigravityReq, true)
    if err != nil {
        return nil, fmt.Errorf("failed to send streaming request to Antigravity: %w", err)
    }
    
    // Transform streaming chunks to OpenAI format
    openaiChan := make(chan string, 100)
    
    go func() {
        defer close(openaiChan)
        
        for chunk := range streamChan {
            openaiChunk, err := p.transformer.StreamChunkToOpenAI(chunk)
            if err != nil {
                continue
            }
            
            // Serialize to JSON
            jsonData, err := openaiChunk.ToJSON()
            if err != nil {
                continue
            }
            
            openaiChan <- jsonData
        }
    }()
    
    return openaiChan, nil
}

// GetAuthenticator returns the authenticator
func (p *AntigravityProvider) GetAuthenticator() providers.Authenticator {
    return p.authenticator
}

// RequiresAuth indicates if this provider needs authentication
func (p *AntigravityProvider) RequiresAuth() bool {
    return true
}

// GetConfig returns the current configuration
func (p *AntigravityProvider) GetConfig() providers.ProviderConfig {
    return p.config
}

// UpdateConfig updates the provider's configuration
func (p *AntigravityProvider) UpdateConfig(cfg providers.ProviderConfig) error {
    p.config = cfg
    
    // Recreate client with new configuration
    // This would require access to ProxyManager
    // For now, just update config
    
    return nil
}

// HealthCheck performs a health check on the provider
func (p *AntigravityProvider) HealthCheck(ctx context.Context) error {
    // Simple health check by listing models
    _, err := p.ListModels(ctx)
    return err
}

// Close performs cleanup operations
func (p *AntigravityProvider) Close() error {
    if p.client != nil {
        return p.client.Close()
    }
    return nil
}

// getAntigravityModels returns a list of Antigravity models
func getAntigravityModels() []providers.ModelInfo {
    return []providers.ModelInfo{
        {
            ID:                       "ag-1",
            Name:                     "ag-1",
            DisplayName:              "Antigravity 1",
            Description:              "Base model for general tasks",
            InputTokenLimit:         65536,
            OutputTokenLimit:        32768,
            SupportedGenerationMethods: []string{"chat", "completion"},
            Thinking: &providers.ThinkingSupport{
                Min:            0,
                Max:            0,
                ZeroAllowed:    true,
                DynamicAllowed: false,
            },
        },
        {
            ID:                       "ag-2",
            Name:                     "ag-2",
            DisplayName:              "Antigravity 2",
            Description:              "Enhanced model for complex tasks",
            InputTokenLimit:         131072,
            OutputTokenLimit:        65536,
            SupportedGenerationMethods: []string{"chat", "completion"},
            Thinking: &providers.ThinkingSupport{
                Min:            0,
                Max:            0,
                ZeroAllowed:    true,
                DynamicAllowed: false,
            },
        },
        {
            ID:                       "ag-3",
            Name:                     "ag-3",
            DisplayName:              "Antigravity 3",
            Description:              "Most capable model for advanced reasoning",
            InputTokenLimit:         262144,
            OutputTokenLimit:        131072,
            SupportedGenerationMethods: []string{"chat", "completion"},
            Thinking: &providers.ThinkingSupport{
                Min:            0,
                Max:            0,
                ZeroAllowed:    true,
                DynamicAllowed: false,
            },
        },
    }
}
```

### 2. `internal/providers/antigravity/client.go`

**Purpose**: Antigravity HTTP client

**Full Implementation**:

```go
package antigravity

import (
    "bufio"
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "sync"
    "time"
    
    "gcli2apigo/internal/proxy"
)

// AntigravityClient handles HTTP requests to Antigravity API
type AntigravityClient struct {
    config    providers.ProviderConfig
    proxyMgr  proxy.ProxyManager
    httpClient *http.Client
    mu        sync.Mutex
}

// NewAntigravityClient creates a new Antigravity client
func NewAntigravityClient(cfg providers.ProviderConfig, proxyMgr proxy.ProxyManager) (*AntigravityClient, error) {
    client := &AntigravityClient{
        config:   cfg,
        proxyMgr: proxyMgr,
    }
    
    // Get HTTP client with proxy
    if proxyMgr != nil && cfg.Proxy != nil && cfg.Proxy.Enabled {
        if proxy, err := proxyMgr.GetProxy("antigravity"); err == nil {
            if httpClient, err := proxy.GetHTTPClient(); err == nil {
                client.httpClient = httpClient
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

// SendRequest sends a request to Antigravity API
func (c *AntigravityClient) SendRequest(ctx context.Context, payload interface{}, isStreaming bool) (interface{}, error) {
    // Build URL
    endpoint := c.config.APIEndpoint
    if isStreaming {
        endpoint += "/v1/chat/stream"
    } else {
        endpoint += "/v1/chat"
    }
    
    // Serialize payload
    buf := getBufferFromPool()
    defer returnBufferToPool(buf)
    
    if err := json.NewEncoder(buf).Encode(payload); err != nil {
        return nil, fmt.Errorf("failed to encode payload: %w", err)
    }
    
    // Create request
    req, err := http.NewRequestWithContext(ctx, "POST", endpoint, buf)
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }
    
    // Set headers
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+c.getAuthToken())
    req.Header.Set("User-Agent", "gcli2apigo/2.0.0")
    req.Header.Set("X-Request-ID", generateRequestID())
    
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
func (c *AntigravityClient) handleStreamingResponse(resp *http.Response) (<-chan string, error) {
    streamChan := make(chan string, 10)
    
    go func() {
        defer close(streamChan)
        defer resp.Body.Close()
        
        reader := bufio.NewReader(resp.Body)
        
        for {
            line, err := reader.ReadString('\n')
            if err != nil {
                if err != io.EOF {
                    // Log error but continue
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
func (c *AntigravityClient) handleNonStreamingResponse(resp *http.Response) (map[string]interface{}, error) {
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
func (c *AntigravityClient) Close() error {
    // Close idle connections
    if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
        transport.CloseIdleConnections()
    }
    return nil
}

// getAuthToken returns the authentication token
func (c *AntigravityClient) getAuthToken() string {
    // This should be obtained from authenticator
    // For now, return a placeholder
    return "placeholder-token"
}

// generateRequestID generates a unique request ID
func generateRequestID() string {
    return fmt.Sprintf("%d", time.Now().UnixNano())
}

// Buffer pool

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

### 3. `internal/providers/antigravity/transformer.go`

**Purpose**: Request/response transformer for Antigravity

**Full Implementation**:

```go
package antigravity

import (
    "encoding/json"
    "fmt"
    "time"
    
    "gcli2apigo/internal/providers"
    "gcli2apigo/internal/transformers"
)

// AntigravityTransformer transforms between OpenAI and Antigravity formats
type AntigravityTransformer struct{}

// NewAntigravityTransformer creates a new Antigravity transformer
func NewAntigravityTransformer() *AntigravityTransformer {
    return &AntigravityTransformer{}
}

// RequestToProvider transforms OpenAI request to Antigravity format
func (t *AntigravityTransformer) RequestToProvider(req *providers.ChatCompletionRequest) (interface{}, error) {
    // Antigravity uses OpenAI-compatible format
    // Minimal transformation needed
    
    antigravityReq := map[string]interface{}{
        "model":    req.Model,
        "messages": convertMessages(req.Messages),
        "stream":   req.Stream,
    }
    
    // Add optional parameters
    if req.Temperature != nil {
        antigravityReq["temperature"] = *req.Temperature
    }
    if req.MaxTokens != nil {
        antigravityReq["max_tokens"] = *req.MaxTokens
    }
    if req.TopP != nil {
        antigravityReq["top_p"] = *req.TopP
    }
    
    return antigravityReq, nil
}

// ResponseToOpenAI transforms Antigravity response to OpenAI format
func (t *AntigravityTransformer) ResponseToOpenAI(resp interface{}, model string) (*providers.ChatCompletionResponse, error) {
    // Antigravity uses OpenAI-compatible format
    // Minimal transformation needed
    
    antigravityResp, ok := resp.(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("invalid response type")
    }
    
    // Build OpenAI response
    openaiResp := &providers.ChatCompletionResponse{
        ID:      extractID(antigravityResp),
        Object:  "chat.completion",
        Created: time.Now().Unix(),
        Model:   model,
        Choices: extractChoices(antigravityResp),
    }
    
    // Add usage if present
    if usage, ok := antigravityResp["usage"]; ok {
        openaiResp.Usage = extractUsage(usage)
    }
    
    return openaiResp, nil
}

// StreamChunkToOpenAI transforms streaming chunk to OpenAI format
func (t *AntigravityTransformer) StreamChunkToOpenAI(chunk string) (*providers.ChatCompletionResponse, error) {
    // Antigravity uses OpenAI-compatible format
    // Minimal transformation needed
    
    var antigravityChunk map[string]interface{}
    if err := json.Unmarshal([]byte(chunk), &antigravityChunk); err != nil {
        return nil, fmt.Errorf("failed to parse chunk: %w", err)
    }
    
    // Build OpenAI streaming response
    openaiResp := &providers.ChatCompletionResponse{
        ID:      extractID(antigravityChunk),
        Object:  "chat.completion.chunk",
        Created: time.Now().Unix(),
        Model:   "ag-1",
        Choices: extractChoices(antigravityChunk),
    }
    
    return openaiResp, nil
}

// ValidateRequest validates an OpenAI request
func (t *AntigravityTransformer) ValidateRequest(req *providers.ChatCompletionRequest) error {
    if req.Model == "" {
        return fmt.Errorf("model is required")
    }
    
    if len(req.Messages) == 0 {
        return fmt.Errorf("at least one message is required")
    }
    
    return nil
}

// GetSupportedFeatures returns features supported by Antigravity
func (t *AntigravityTransformer) GetSupportedFeatures() *transformers.FeatureSupport {
    return &transformers.FeatureSupport{
        Streaming:        true,
        Thinking:         false,
        Tools:            false,
        Images:           false,
        JSONMode:         false,
        SystemPrompt:     true,
        FunctionCalling:  false,
    }
}

// Helper functions

func convertMessages(msgs []providers.ChatMessage) []map[string]interface{} {
    result := make([]map[string]interface{}, len(msgs))
    for i, msg := range msgs {
        result[i] = map[string]interface{}{
            "role":    msg.Role,
            "content": msg.Content,
        }
    }
    return result
}

func extractID(resp map[string]interface{}) string {
    if id, ok := resp["id"].(string); ok {
        return id
    }
    return fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
}

func extractChoices(resp map[string]interface{}) []providers.CompletionChoice {
    choicesRaw, _ := resp["choices"].([]interface{})
    choices := make([]providers.CompletionChoice, len(choicesRaw))
    
    for i, choiceRaw := range choicesRaw {
        choice, _ := choiceRaw.(map[string]interface{})
        choices[i] = providers.CompletionChoice{
            Index:        int(choice["index"].(float64)),
            Message:      extractMessage(choice),
            FinishReason: extractFinishReason(choice),
        }
    }
    
    return choices
}

func extractMessage(choice map[string]interface{}) providers.ChatMessage {
    msgRaw, _ := choice["message"].(map[string]interface{})
    return providers.ChatMessage{
        Role:    msgRaw["role"].(string),
        Content: msgRaw["content"],
    }
}

func extractFinishReason(choice map[string]interface{}) *string {
    if reason, ok := choice["finish_reason"].(string); ok && reason != "" {
        return &reason
    }
    return nil
}

func extractUsage(usage interface{}) *providers.Usage {
    usageMap, _ := usage.(map[string]interface{})
    return &providers.Usage{
        PromptTokens:     int(usageMap["prompt_tokens"].(float64)),
        CompletionTokens: int(usageMap["completion_tokens"].(float64)),
        TotalTokens:      int(usageMap["total_tokens"].(float64)),
    }
}
```

## Dependencies

- **Step 01**: Core Interfaces (Provider interface)
- **Step 02**: Shared Models (request/response types)
- **Step 03**: Proxy Infrastructure (HTTP client)

## Antigravity API Details

### Authentication

Antigravity uses JWT authentication:

- **JWT Secret**: Set via `ANTIGRAVITY_JWT_SECRET` environment variable
- **JWT Endpoint**: Set via `ANTIGRAVITY_JWT_ENDPOINT` environment variable

### API Endpoints

- **Chat Completions**: `https://api.antigravity.com/v1/chat`
- **Streaming Chat**: `https://api.antigravity.com/v1/chat/stream`
- **Models**: `https://api.antigravity.com/v1/models`

### Models

- `ag-1`: Base model for general tasks
- `ag-2`: Enhanced model for complex tasks
- `ag-3`: Most capable model for advanced reasoning

## Verification

After completing this step, verify:

1. Antigravity provider implements Provider interface
2. Request/response transformations work correctly
3. HTTP client handles proxy configuration
4. Authentication is properly configured
5. Health check works

## Next Steps

After completing this step, proceed to:
- **Step 13**: Main.go Integration (wire everything together)
- **Step 14**: Testing and Documentation
- **Step 15**: Deployment Guide
