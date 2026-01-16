# Step 11: Qwen Provider Implementation

## Context

This step implements the Qwen provider following the Provider interface. Qwen (Alibaba Cloud's AI model) provides Chinese language models and general AI capabilities.

## Objectives

1. Create Qwen provider directory structure
2. Implement Qwen HTTP client
3. Implement request/response transformers
4. Implement Provider interface for Qwen
5. Configure Qwen-specific authentication

## Design Pattern

**Strategy Pattern**: Qwen provider implements Provider interface, encapsulating all Qwen-specific logic.

## Files to Create

### 1. `internal/providers/qwen/provider.go`

**Purpose**: Main Qwen provider implementation

**Full Implementation**:

```go
package qwen

import (
    "context"
    "fmt"
    
    "gcli2apigo/internal/providers"
    "gcli2apigo/internal/transformers"
)

// QwenProvider implements Provider interface for Alibaba's Qwen API
type QwenProvider struct {
    config        providers.ProviderConfig
    client        *QwenClient
    transformer   *QwenTransformer
    authenticator providers.Authenticator
    models        []providers.ModelInfo
}

// NewQwenProvider creates a new Qwen provider
func NewQwenProvider(cfg providers.ProviderConfig, deps providers.ProviderDependencies) (providers.Provider, error) {
    // Create transformer
    transformer := NewQwenTransformer()
    
    // Create client with proxy
    client, err := NewQwenClient(cfg, deps.ProxyManager)
    if err != nil {
        return nil, fmt.Errorf("failed to create Qwen client: %w", err)
    }
    
    provider := &QwenProvider{
        config:      cfg,
        client:      client,
        transformer: transformer,
        models:      getQwenModels(),
    }
    
    // Set authenticator if provided
    if deps.AuthManager != nil {
        provider.authenticator = deps.AuthManager.GetAuthenticator("qwen")
    }
    
    return provider, nil
}

// GetType returns the provider type
func (p *QwenProvider) GetType() providers.ProviderType {
    return providers.ProviderQwen
}

// GetName returns the provider name
func (p *QwenProvider) GetName() string {
    return "Qwen"
}

// GetVersion returns the provider version
func (p *QwenProvider) GetVersion() string {
    return "1.0.0"
}

// ListModels returns all available models
func (p *QwenProvider) ListModels(ctx context.Context) ([]providers.ModelInfo, error) {
    return p.models, nil
}

// ValidateModel checks if a model ID is valid
func (p *QwenProvider) ValidateModel(modelID string) bool {
    for _, model := range p.models {
        if model.ID == modelID || model.Name == modelID {
            return true
        }
    }
    return false
}

// HandleChatCompletion processes a non-streaming chat completion request
func (p *QwenProvider) HandleChatCompletion(ctx context.Context, req *providers.ChatCompletionRequest) (*providers.ChatCompletionResponse, error) {
    // Transform request to Qwen format
    qwenReq, err := p.transformer.RequestToProvider(req)
    if err != nil {
        return nil, fmt.Errorf("failed to transform request: %w", err)
    }
    
    // Send request to Qwen API
    qwenResp, err := p.client.SendRequest(ctx, qwenReq, false)
    if err != nil {
        return nil, fmt.Errorf("failed to send request to Qwen: %w", err)
    }
    
    // Transform response to OpenAI format
    openaiResp, err := p.transformer.ResponseToOpenAI(qwenResp, req.Model)
    if err != nil {
        return nil, fmt.Errorf("failed to transform response: %w", err)
    }
    
    return openaiResp, nil
}

// HandleStreamingChatCompletion processes a streaming chat completion request
func (p *QwenProvider) HandleStreamingChatCompletion(ctx context.Context, req *providers.ChatCompletionRequest) (<-chan string, error) {
    // Transform request to Qwen format
    qwenReq, err := p.transformer.RequestToProvider(req)
    if err != nil {
        return nil, fmt.Errorf("failed to transform request: %w", err)
    }
    
    // Send streaming request to Qwen API
    streamChan, err := p.client.SendRequest(ctx, qwenReq, true)
    if err != nil {
        return nil, fmt.Errorf("failed to send streaming request to Qwen: %w", err)
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
func (p *QwenProvider) GetAuthenticator() providers.Authenticator {
    return p.authenticator
}

// RequiresAuth indicates if this provider needs authentication
func (p *QwenProvider) RequiresAuth() bool {
    return true
}

// GetConfig returns the current configuration
func (p *QwenProvider) GetConfig() providers.ProviderConfig {
    return p.config
}

// UpdateConfig updates the provider's configuration
func (p *QwenProvider) UpdateConfig(cfg providers.ProviderConfig) error {
    p.config = cfg
    
    // Recreate client with new configuration
    // This would require access to ProxyManager
    // For now, just update config
    
    return nil
}

// HealthCheck performs a health check on the provider
func (p *QwenProvider) HealthCheck(ctx context.Context) error {
    // Simple health check by listing models
    _, err := p.ListModels(ctx)
    return err
}

// Close performs cleanup operations
func (p *QwenProvider) Close() error {
    if p.client != nil {
        return p.client.Close()
    }
    return nil
}

// getQwenModels returns the list of Qwen models
func getQwenModels() []providers.ModelInfo {
    return []providers.ModelInfo{
        {
            ID:                       "qwen-turbo",
            Name:                     "qwen-turbo",
            DisplayName:              "Qwen Turbo",
            Description:              "High-performance model for complex tasks",
            InputTokenLimit:         8192,
            OutputTokenLimit:        2048,
            SupportedGenerationMethods: []string{"chat", "completion"},
            Thinking: &providers.ThinkingSupport{
                Min:            0,
                Max:            0,
                ZeroAllowed:    true,
                DynamicAllowed: false,
            },
        },
        {
            ID:                       "qwen-plus",
            Name:                     "qwen-plus",
            DisplayName:              "Qwen Plus",
            Description:              "Balanced model for general tasks",
            InputTokenLimit:         32768,
            OutputTokenLimit:        8192,
            SupportedGenerationMethods: []string{"chat", "completion"},
            Thinking: &providers.ThinkingSupport{
                Min:            0,
                Max:            0,
                ZeroAllowed:    true,
                DynamicAllowed: false,
            },
        },
        {
            ID:                       "qwen-max",
            Name:                     "qwen-max",
            DisplayName:              "Qwen Max",
            Description:              "Most capable model for complex reasoning",
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
    }
}
```

### 2. `internal/providers/qwen/client.go`

**Purpose**: Qwen HTTP client

**Full Implementation**:

```go
package qwen

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

// QwenClient handles HTTP requests to Qwen API
type QwenClient struct {
    config    providers.ProviderConfig
    proxyMgr  proxy.ProxyManager
    httpClient *http.Client
    mu        sync.Mutex
}

// NewQwenClient creates a new Qwen client
func NewQwenClient(cfg providers.ProviderConfig, proxyMgr proxy.ProxyManager) (*QwenClient, error) {
    client := &QwenClient{
        config:   cfg,
        proxyMgr: proxyMgr,
    }
    
    // Get HTTP client with proxy
    if proxyMgr != nil && cfg.Proxy != nil && cfg.Proxy.Enabled {
        if proxy, err := proxyMgr.GetProxy("qwen"); err == nil {
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

// SendRequest sends a request to Qwen API
func (c *QwenClient) SendRequest(ctx context.Context, payload interface{}, isStreaming bool) (interface{}, error) {
    // Build URL
    endpoint := c.config.APIEndpoint
    if isStreaming {
        endpoint += "/v1/services/aigc/text-generation/generation"
    } else {
        endpoint += "/v1/services/aigc/text-generation/generation"
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
    req.Header.Set("X-DashScope", "dashscope")
    
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
func (c *QwenClient) handleStreamingResponse(resp *http.Response) (<-chan string, error) {
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
func (c *QwenClient) handleNonStreamingResponse(resp *http.Response) (map[string]interface{}, error) {
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
func (c *QwenClient) Close() error {
    // Close idle connections
    if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
        transport.CloseIdleConnections()
    }
    return nil
}

// getAuthToken returns the authentication token
func (c *QwenClient) getAuthToken() string {
    // This should be obtained from authenticator
    // For now, return a placeholder
    return "placeholder-token"
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

### 3. `internal/providers/qwen/transformer.go`

**Purpose**: Request/response transformer for Qwen

**Full Implementation**:

```go
package qwen

import (
    "encoding/json"
    "fmt"
    "time"
    
    "gcli2apigo/internal/providers"
    "gcli2apigo/internal/transformers"
)

// QwenTransformer transforms between OpenAI and Qwen formats
type QwenTransformer struct{}

// NewQwenTransformer creates a new Qwen transformer
func NewQwenTransformer() *QwenTransformer {
    return &QwenTransformer{}
}

// RequestToProvider transforms OpenAI request to Qwen format
func (t *QwenTransformer) RequestToProvider(req *providers.ChatCompletionRequest) (interface{}, error) {
    // Qwen uses OpenAI-compatible format with some differences
    qwenReq := map[string]interface{}{
        "model": req.Model,
        "input": map[string]interface{}{
            "messages": convertMessages(req.Messages),
        },
        "parameters": map[string]interface{}{
            "result_format": "message",
        },
    }
    
    // Add optional parameters
    if req.Temperature != nil {
        qwenReq["parameters"].(map[string]interface{})["temperature"] = *req.Temperature
    }
    if req.MaxTokens != nil {
        qwenReq["parameters"].(map[string]interface{})["max_tokens"] = *req.MaxTokens
    }
    if req.TopP != nil {
        qwenReq["parameters"].(map[string]interface{})["top_p"] = *req.TopP
    }
    
    return qwenReq, nil
}

// ResponseToOpenAI transforms Qwen response to OpenAI format
func (t *QwenTransformer) ResponseToOpenAI(resp interface{}, model string) (*providers.ChatCompletionResponse, error) {
    qwenResp, ok := resp.(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("invalid response type")
    }
    
    // Extract output
    output, _ := qwenResp["output"].(map[string]interface{})
    text, _ := output["text"].(string)
    
    // Build OpenAI response
    openaiResp := &providers.ChatCompletionResponse{
        ID:      generateID(),
        Object:  "chat.completion",
        Created: time.Now().Unix(),
        Model:   model,
        Choices: []providers.CompletionChoice{
            {
                Index:   0,
                Message: providers.ChatMessage{
                    Role:    "assistant",
                    Content: text,
                },
            },
        },
    }
    
    return openaiResp, nil
}

// StreamChunkToOpenAI transforms streaming chunk to OpenAI format
func (t *QwenTransformer) StreamChunkToOpenAI(chunk string) (*providers.ChatCompletionResponse, error) {
    var qwenChunk map[string]interface{}
    if err := json.Unmarshal([]byte(chunk), &qwenChunk); err != nil {
        return nil, fmt.Errorf("failed to parse chunk: %w", err)
    }
    
    // Extract output
    output, _ := qwenChunk["output"].(map[string]interface{})
    text, _ := output["text"].(string)
    
    // Build OpenAI streaming response
    openaiResp := &providers.ChatCompletionResponse{
        ID:      generateID(),
        Object:  "chat.completion.chunk",
        Created: time.Now().Unix(),
        Model:   "qwen-turbo",
        Choices: []providers.CompletionChoice{
            {
                Index: 0,
                Message: providers.ChatMessage{
                    Role:    "assistant",
                    Content: text,
                },
            },
        },
    }
    
    return openaiResp, nil
}

// ValidateRequest validates an OpenAI request
func (t *QwenTransformer) ValidateRequest(req *providers.ChatCompletionRequest) error {
    if req.Model == "" {
        return fmt.Errorf("model is required")
    }
    
    if len(req.Messages) == 0 {
        return fmt.Errorf("at least one message is required")
    }
    
    return nil
}

// GetSupportedFeatures returns features supported by Qwen
func (t *QwenTransformer) GetSupportedFeatures() *transformers.FeatureSupport {
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

func generateID() string {
    return fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
}
```

## Dependencies

- **Step 01**: Core Interfaces (Provider interface)
- **Step 02**: Shared Models (request/response types)
- **Step 03**: Proxy Infrastructure (HTTP client)

## Qwen API Details

### Authentication

Qwen uses API Key authentication:

- **API Key**: Set via `QWEN_API_KEY` environment variable

### API Endpoints

- **Chat Completions**: `https://dashscope.aliyuncs.com/v1/services/aigc/text-generation/generation`
- **Models**: `https://dashscope.aliyuncs.com/v1/services/aigc/text-generation/models`

### Models

- `qwen-turbo`: High-performance model
- `qwen-plus`: Balanced model
- `qwen-max`: Most capable model

## Verification

After completing this step, verify:

1. Qwen provider implements Provider interface
2. Request/response transformations work correctly
3. HTTP client handles proxy configuration
4. Authentication is properly configured
5. Health check works

## Next Steps

After completing this step, proceed to:
- **Step 12**: Antigravity Provider Implementation
- **Step 13**: Main.go Integration (wire everything together)
- **Step 14**: Testing and Documentation
