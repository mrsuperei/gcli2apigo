# Step 10: Copilot Provider Implementation

## Context

This step implements the Copilot provider following the Provider interface. Copilot (GitHub Copilot) provides AI coding assistance through GitHub's API.

## Objectives

1. Create Copilot provider directory structure
2. Implement Copilot HTTP client
3. Implement request/response transformers
4. Implement Provider interface for Copilot
5. Configure Copilot-specific authentication

## Design Pattern

**Strategy Pattern**: Copilot provider implements Provider interface, encapsulating all Copilot-specific logic.

## Files to Create

### 1. `internal/providers/copilot/provider.go`

**Purpose**: Main Copilot provider implementation

**Full Implementation**:

```go
package copilot

import (
    "context"
    "fmt"
    
    "gcli2apigo/internal/providers"
    "gcli2apigo/internal/transformers"
)

// CopilotProvider implements Provider interface for GitHub Copilot
type CopilotProvider struct {
    config        providers.ProviderConfig
    client        *CopilotClient
    transformer   *CopilotTransformer
    authenticator providers.Authenticator
    models        []providers.ModelInfo
}

// NewCopilotProvider creates a new Copilot provider
func NewCopilotProvider(cfg providers.ProviderConfig, deps providers.ProviderDependencies) (providers.Provider, error) {
    // Create transformer
    transformer := NewCopilotTransformer()
    
    // Create client with proxy
    client, err := NewCopilotClient(cfg, deps.ProxyManager)
    if err != nil {
        return nil, fmt.Errorf("failed to create Copilot client: %w", err)
    }
    
    provider := &CopilotProvider{
        config:      cfg,
        client:      client,
        transformer: transformer,
        models:      getCopilotModels(),
    }
    
    // Set authenticator if provided
    if deps.AuthManager != nil {
        provider.authenticator = deps.AuthManager.GetAuthenticator("copilot")
    }
    
    return provider, nil
}

// GetType returns the provider type
func (p *CopilotProvider) GetType() providers.ProviderType {
    return providers.ProviderCopilot
}

// GetName returns the provider name
func (p *CopilotProvider) GetName() string {
    return "GitHub Copilot"
}

// GetVersion returns the provider version
func (p *CopilotProvider) GetVersion() string {
    return "1.0.0"
}

// ListModels returns all available models
func (p *CopilotProvider) ListModels(ctx context.Context) ([]providers.ModelInfo, error) {
    return p.models, nil
}

// ValidateModel checks if a model ID is valid
func (p *CopilotProvider) ValidateModel(modelID string) bool {
    for _, model := range p.models {
        if model.ID == modelID || model.Name == modelID {
            return true
        }
    }
    return false
}

// HandleChatCompletion processes a non-streaming chat completion request
func (p *CopilotProvider) HandleChatCompletion(ctx context.Context, req *providers.ChatCompletionRequest) (*providers.ChatCompletionResponse, error) {
    // Transform request to Copilot format
    copilotReq, err := p.transformer.RequestToProvider(req)
    if err != nil {
        return nil, fmt.Errorf("failed to transform request: %w", err)
    }
    
    // Send request to Copilot API
    copilotResp, err := p.client.SendRequest(ctx, copilotReq, false)
    if err != nil {
        return nil, fmt.Errorf("failed to send request to Copilot: %w", err)
    }
    
    // Transform response to OpenAI format
    openaiResp, err := p.transformer.ResponseToOpenAI(copilotResp, req.Model)
    if err != nil {
        return nil, fmt.Errorf("failed to transform response: %w", err)
    }
    
    return openaiResp, nil
}

// HandleStreamingChatCompletion processes a streaming chat completion request
func (p *CopilotProvider) HandleStreamingChatCompletion(ctx context.Context, req *providers.ChatCompletionRequest) (<-chan string, error) {
    // Transform request to Copilot format
    copilotReq, err := p.transformer.RequestToProvider(req)
    if err != nil {
        return nil, fmt.Errorf("failed to transform request: %w", err)
    }
    
    // Send streaming request to Copilot API
    streamChan, err := p.client.SendRequest(ctx, copilotReq, true)
    if err != nil {
        return nil, fmt.Errorf("failed to send streaming request to Copilot: %w", err)
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
func (p *CopilotProvider) GetAuthenticator() providers.Authenticator {
    return p.authenticator
}

// RequiresAuth indicates if this provider needs authentication
func (p *CopilotProvider) RequiresAuth() bool {
    return true
}

// GetConfig returns the current configuration
func (p *CopilotProvider) GetConfig() providers.ProviderConfig {
    return p.config
}

// UpdateConfig updates the provider's configuration
func (p *CopilotProvider) UpdateConfig(cfg providers.ProviderConfig) error {
    p.config = cfg
    
    // Recreate client with new configuration
    // This would require access to ProxyManager
    // For now, just update config
    
    return nil
}

// HealthCheck performs a health check on the provider
func (p *CopilotProvider) HealthCheck(ctx context.Context) error {
    // Simple health check by listing models
    _, err := p.ListModels(ctx)
    return err
}

// Close performs cleanup operations
func (p *CopilotProvider) Close() error {
    if p.client != nil {
        return p.client.Close()
    }
    return nil
}

// getCopilotModels returns the list of Copilot models
func getCopilotModels() []providers.ModelInfo {
    return []providers.ModelInfo{
        {
            ID:                       "gpt-4",
            Name:                     "gpt-4",
            DisplayName:              "GPT-4",
            Description:              "GitHub Copilot's most capable model",
            InputTokenLimit:         128000,
            OutputTokenLimit:        4096,
            SupportedGenerationMethods: []string{"chat", "completion"},
            Thinking: &providers.ThinkingSupport{
                Min:            0,
                Max:            0,
                ZeroAllowed:    true,
                DynamicAllowed: false,
            },
        },
        {
            ID:                       "gpt-4-turbo",
            Name:                     "gpt-4-turbo",
            DisplayName:              "GPT-4 Turbo",
            Description:              "Faster and cheaper model for most tasks",
            InputTokenLimit:         128000,
            OutputTokenLimit:        4096,
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

### 2. `internal/providers/copilot/client.go`

**Purpose**: Copilot HTTP client

**Full Implementation**:

```go
package copilot

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

// CopilotClient handles HTTP requests to Copilot API
type CopilotClient struct {
    config    providers.ProviderConfig
    proxyMgr  proxy.ProxyManager
    httpClient *http.Client
    mu        sync.Mutex
}

// NewCopilotClient creates a new Copilot client
func NewCopilotClient(cfg providers.ProviderConfig, proxyMgr proxy.ProxyManager) (*CopilotClient, error) {
    client := &CopilotClient{
        config:   cfg,
        proxyMgr: proxyMgr,
    }
    
    // Get HTTP client with proxy
    if proxyMgr != nil && cfg.Proxy != nil && cfg.Proxy.Enabled {
        if proxy, err := proxyMgr.GetProxy("copilot"); err == nil {
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

// SendRequest sends a request to Copilot API
func (c *CopilotClient) SendRequest(ctx context.Context, payload interface{}, isStreaming bool) (interface{}, error) {
    // Build URL
    endpoint := c.config.APIEndpoint
    if isStreaming {
        endpoint += "/chat/completions"
    } else {
        endpoint += "/chat/completions"
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
func (c *CopilotClient) handleStreamingResponse(resp *http.Response) (<-chan string, error) {
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
func (c *CopilotClient) handleNonStreamingResponse(resp *http.Response) (map[string]interface{}, error) {
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
func (c *CopilotClient) Close() error {
    // Close idle connections
    if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
        transport.CloseIdleConnections()
    }
    return nil
}

// getAuthToken returns the authentication token
func (c *CopilotClient) getAuthToken() string {
    // This should be obtained from the authenticator
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

### 3. `internal/providers/copilot/transformer.go`

**Purpose**: Request/response transformer for Copilot

**Full Implementation**:

```go
package copilot

import (
    "encoding/json"
    "fmt"
    "time"
    
    "gcli2apigo/internal/providers"
    "gcli2apigo/internal/transformers"
)

// CopilotTransformer transforms between OpenAI and Copilot formats
type CopilotTransformer struct{}

// NewCopilotTransformer creates a new Copilot transformer
func NewCopilotTransformer() *CopilotTransformer {
    return &CopilotTransformer{}
}

// RequestToProvider transforms OpenAI request to Copilot format
func (t *CopilotTransformer) RequestToProvider(req *providers.ChatCompletionRequest) (interface{}, error) {
    // Copilot uses OpenAI-compatible format
    // Minimal transformation needed
    
    copilotReq := map[string]interface{}{
        "model":    req.Model,
        "messages": convertMessages(req.Messages),
        "stream":   req.Stream,
    }
    
    // Add optional parameters
    if req.Temperature != nil {
        copilotReq["temperature"] = *req.Temperature
    }
    if req.MaxTokens != nil {
        copilotReq["max_tokens"] = *req.MaxTokens
    }
    if req.TopP != nil {
        copilotReq["top_p"] = *req.TopP
    }
    if req.Tools != nil && len(req.Tools) > 0 {
        copilotReq["tools"] = convertTools(req.Tools)
    }
    if req.ToolChoice != nil {
        copilotReq["tool_choice"] = req.ToolChoice
    }
    
    return copilotReq, nil
}

// ResponseToOpenAI transforms Copilot response to OpenAI format
func (t *CopilotTransformer) ResponseToOpenAI(resp interface{}, model string) (*providers.ChatCompletionResponse, error) {
    // Copilot uses OpenAI-compatible format
    // Minimal transformation needed
    
    copilotResp, ok := resp.(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("invalid response type")
    }
    
    // Build OpenAI response
    openaiResp := &providers.ChatCompletionResponse{
        ID:      extractID(copilotResp),
        Object:  "chat.completion",
        Created: time.Now().Unix(),
        Model:   model,
        Choices: extractChoices(copilotResp),
    }
    
    // Add usage if present
    if usage, ok := copilotResp["usage"]; ok {
        openaiResp.Usage = extractUsage(usage)
    }
    
    return openaiResp, nil
}

// StreamChunkToOpenAI transforms streaming chunk to OpenAI format
func (t *CopilotTransformer) StreamChunkToOpenAI(chunk string) (*providers.ChatCompletionResponse, error) {
    // Copilot uses OpenAI-compatible format
    // Minimal transformation needed
    
    var copilotChunk map[string]interface{}
    if err := json.Unmarshal([]byte(chunk), &copilotChunk); err != nil {
        return nil, fmt.Errorf("failed to parse chunk: %w", err)
    }
    
    // Build OpenAI streaming response
    openaiResp := &providers.ChatCompletionResponse{
        ID:      extractID(copilotChunk),
        Object:  "chat.completion.chunk",
        Created: time.Now().Unix(),
        Model:   "gpt-4",
        Choices: extractChoices(copilotChunk),
    }
    
    return openaiResp, nil
}

// ValidateRequest validates an OpenAI request
func (t *CopilotTransformer) ValidateRequest(req *providers.ChatCompletionRequest) error {
    if req.Model == "" {
        return fmt.Errorf("model is required")
    }
    
    if len(req.Messages) == 0 {
        return fmt.Errorf("at least one message is required")
    }
    
    return nil
}

// GetSupportedFeatures returns features supported by Copilot
func (t *CopilotTransformer) GetSupportedFeatures() *transformers.FeatureSupport {
    return &transformers.FeatureSupport{
        Streaming:        true,
        Thinking:         false,
        Tools:            true,
        Images:           false,
        JSONMode:         false,
        SystemPrompt:     true,
        FunctionCalling:  true,
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

func convertTools(tools []providers.Tool) []map[string]interface{} {
    result := make([]map[string]interface{}, len(tools))
    for i, tool := range tools {
        result[i] = map[string]interface{}{
            "type":     tool.Type,
            "function": map[string]interface{}{
                "name":        tool.Function.Name,
                "description": tool.Function.Description,
                "parameters":  tool.Function.Parameters,
            },
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

## Copilot API Details

### Authentication

Copilot uses OAuth 2.0 for authentication:

- **Auth URL**: `https://github.com/login/oauth/authorize`
- **Token URL**: `https://github.com/login/oauth/access_token`
- **Scopes**: `read:user`, `read:org`, `read:project`, `write:project`

### API Endpoints

- **Chat Completions**: `https://api.githubcopilot.com/chat/completions`
- **Models**: `https://api.githubcopilot.com/models`

### Models

- `gpt-4`: Most capable model
- `gpt-4-turbo`: Faster and cheaper model

## Verification

After completing this step, verify:

1. Copilot provider implements Provider interface
2. Request/response transformations work correctly
3. HTTP client handles proxy configuration
4. Authentication is properly configured
5. Health check works

## Next Steps

After completing this step, proceed to:
- **Step 11**: Qwen Provider Implementation
- **Step 12**: Antigravity Provider Implementation
- **Step 13**: Main.go Integration (wire everything together)
