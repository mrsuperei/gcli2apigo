# Step 01: Core Interfaces

## Context

This step establishes the foundational interfaces for the multi-provider architecture. These interfaces define contracts that all providers must implement, enabling the Strategy pattern where provider-specific logic is encapsulated behind common abstractions.

## Objectives

1. Define the `Provider` interface - the core abstraction for all AI providers
2. Define the `Transformer` interface - for request/response transformation
3. Define the `Proxy` interface - for HTTP proxy configuration
4. Define supporting types and configurations
5. Ensure interfaces are extensible for future providers

## Design Pattern

**Strategy Pattern**: Each provider implements the same interface, allowing the system to switch providers at runtime without changing client code.

## Files to Create

### 1. `internal/providers/interfaces.go`

**Purpose**: Core provider interface and supporting types

**Full Implementation**:

```go
package providers

import (
    "context"
    "net/http"
)

// ProviderType identifies the AI provider
type ProviderType string

const (
    ProviderGemini      ProviderType = "gemini"
    ProviderCopilot    ProviderType = "copilot"
    ProviderQwen       ProviderType = "qwen"
    ProviderAntigravity ProviderType = "antigravity"
)

// Provider defines the contract that all AI providers must implement
// This interface enables the Strategy pattern - providers can be swapped at runtime
type Provider interface {
    // Provider identification
    GetType() ProviderType
    GetName() string
    GetVersion() string
    
    // Model management
    // ListModels returns all available models for this provider
    ListModels(ctx context.Context) ([]ModelInfo, error)
    
    // ValidateModel checks if a model ID is valid for this provider
    ValidateModel(modelID string) bool
    
    // Request handling
    // HandleChatCompletion processes a non-streaming chat completion request
    HandleChatCompletion(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error)
    
    // HandleStreamingChatCompletion processes a streaming chat completion request
    // Returns a channel of response chunks
    HandleStreamingChatCompletion(ctx context.Context, req *ChatCompletionRequest) (<-chan string, error)
    
    // Authentication
    // GetAuthenticator returns the authenticator for this provider
    GetAuthenticator() Authenticator
    
    // RequiresAuth indicates if this provider needs authentication
    RequiresAuth() bool
    
    // Configuration
    // GetConfig returns the current configuration for this provider
    GetConfig() ProviderConfig
    
    // UpdateConfig updates the provider's configuration
    UpdateConfig(config ProviderConfig) error
    
    // Health check
    // HealthCheck performs a health check on the provider
    HealthCheck(ctx context.Context) error
    
    // Cleanup
    // Close performs cleanup operations when shutting down
    Close() error
}

// Authenticator defines the contract for authentication
type Authenticator interface {
    // Authenticate validates credentials and returns an access token
    Authenticate(ctx context.Context, credentials interface{}) (*Token, error)
    
    // RefreshToken refreshes an expired access token
    RefreshToken(ctx context.Context, token *Token) (*Token, error)
    
    // ValidateToken checks if a token is valid and not expired
    ValidateToken(token *Token) bool
}

// Token represents an authentication token
type Token struct {
    AccessToken  string    `json:"access_token"`
    RefreshToken string    `json:"refresh_token,omitempty"`
    TokenType   string    `json:"token_type"`
    Expiry      time.Time `json:"expiry"`
    Extra       map[string]interface{} `json:"extra,omitempty"`
}

// ProviderConfig holds provider-specific configuration
type ProviderConfig struct {
    Type        ProviderType            `json:"type"`
    Enabled     bool                   `json:"enabled"`
    APIEndpoint string                 `json:"api_endpoint"`
    Proxy       *ProxyConfig           `json:"proxy,omitempty"`
    Auth        *AuthConfig            `json:"auth,omitempty"`
    RateLimit   *RateLimitConfig       `json:"rate_limit,omitempty"`
    Models      []ModelInfo            `json:"models,omitempty"`
    Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ProxyConfig defines proxy settings for a provider
type ProxyConfig struct {
    Enabled    bool   `json:"enabled"`
    HTTPProxy  string `json:"http_proxy"`
    HTTPSProxy string `json:"https_proxy"`
    NoProxy    string `json:"no_proxy"`
}

// AuthConfig defines authentication settings
type AuthConfig struct {
    Type           string                 `json:"type"` // oauth, api_key, jwt
    ClientID       string                 `json:"client_id,omitempty"`
    ClientSecret   string                 `json:"client_secret,omitempty"`
    Scopes         []string               `json:"scopes,omitempty"`
    TokenEndpoint  string                 `json:"token_endpoint,omitempty"`
    AuthEndpoint   string                 `json:"auth_endpoint,omitempty"`
    APIKey         string                 `json:"api_key,omitempty"`
    JWTEndpoint   string                 `json:"jwt_endpoint,omitempty"`
    JWTSecret      string                 `json:"jwt_secret,omitempty"`
    Extra          map[string]interface{} `json:"extra,omitempty"`
}

// RateLimitConfig defines rate limiting settings
type RateLimitConfig struct {
    Enabled           bool `json:"enabled"`
    RequestsPerSecond int  `json:"requests_per_second"`
    BurstSize         int  `json:"burst_size"`
}

// ModelInfo represents information about a model
type ModelInfo struct {
    ID                       string   `json:"id"`
    Name                     string   `json:"name"`
    DisplayName              string   `json:"display_name"`
    Description              string   `json:"description"`
    InputTokenLimit         int      `json:"input_token_limit"`
    OutputTokenLimit        int      `json:"output_token_limit"`
    SupportedGenerationMethods []string `json:"supported_generation_methods"`
    Thinking                *ThinkingSupport `json:"thinking,omitempty"`
}

// ThinkingSupport represents thinking/reasoning capabilities
type ThinkingSupport struct {
    Min            int      `json:"min"`
    Max            int      `json:"max"`
    ZeroAllowed    bool     `json:"zero_allowed"`
    DynamicAllowed bool     `json:"dynamic_allowed"`
    Levels         []string `json:"levels,omitempty"`
}

// ChatCompletionRequest represents a chat completion request
type ChatCompletionRequest struct {
    Model            string                 `json:"model"`
    Messages         []ChatMessage         `json:"messages"`
    Stream           bool                   `json:"stream,omitempty"`
    Temperature      *float64               `json:"temperature,omitempty"`
    TopP             *float64               `json:"top_p,omitempty"`
    MaxTokens        *int                   `json:"max_tokens,omitempty"`
    Stop             interface{}            `json:"stop,omitempty"`
    Tools            []Tool                 `json:"tools,omitempty"`
    ToolChoice       interface{}            `json:"tool_choice,omitempty"`
    ResponseFormat   map[string]interface{} `json:"response_format,omitempty"`
}

// ChatMessage represents a chat message
type ChatMessage struct {
    Role             string      `json:"role"`
    Content          interface{} `json:"content"`
    ToolCalls        []ToolCall  `json:"tool_calls,omitempty"`
    ToolCallID       string      `json:"tool_call_id,omitempty"`
    Name             string      `json:"name,omitempty"`
}

// Tool represents a tool definition
type Tool struct {
    Type     string   `json:"type"`
    Function Function `json:"function"`
}

// Function represents a function definition
type Function struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description,omitempty"`
    Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ToolCall represents a tool call
type ToolCall struct {
    ID       string   `json:"id"`
    Type     string   `json:"type"`
    Function Function `json:"function"`
}

// ChatCompletionResponse represents a chat completion response
type ChatCompletionResponse struct {
    ID      string               `json:"id"`
    Object  string               `json:"object"`
    Created int64                `json:"created"`
    Model   string               `json:"model"`
    Choices []CompletionChoice   `json:"choices"`
    Usage   *Usage               `json:"usage,omitempty"`
}

// CompletionChoice represents a completion choice
type CompletionChoice struct {
    Index        int         `json:"index"`
    Message      ChatMessage `json:"message"`
    FinishReason *string     `json:"finish_reason,omitempty"`
}

// Usage represents token usage
type Usage struct {
    PromptTokens            int                      `json:"prompt_tokens"`
    CompletionTokens        int                      `json:"completion_tokens"`
    TotalTokens             int                      `json:"total_tokens"`
    CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// CompletionTokensDetails provides breakdown of completion tokens
type CompletionTokensDetails struct {
    ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
    AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
    RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
}
```

### 2. `internal/transformers/interfaces.go`

**Purpose**: Transformer interface for request/response transformation

**Full Implementation**:

```go
package transformers

import (
    "gcli2apigo/internal/providers"
)

// Transformer defines the contract for request/response transformation
// Each provider implements this to convert between OpenAI format and provider-specific format
type Transformer interface {
    // RequestToProvider transforms an OpenAI request to provider-specific format
    RequestToProvider(req *providers.ChatCompletionRequest) (interface{}, error)
    
    // ResponseToOpenAI transforms a provider response to OpenAI format
    ResponseToOpenAI(resp interface{}, model string) (*providers.ChatCompletionResponse, error)
    
    // StreamChunkToOpenAI transforms a streaming response chunk to OpenAI format
    StreamChunkToOpenAI(chunk string) (*providers.ChatCompletionResponse, error)
    
    // ValidateRequest validates the request format
    ValidateRequest(req *providers.ChatCompletionRequest) error
    
    // GetSupportedFeatures returns the features supported by this provider
    GetSupportedFeatures() *FeatureSupport
}

// FeatureSupport indicates which features are supported by a provider
type FeatureSupport struct {
    Streaming          bool `json:"streaming"`
    Thinking           bool `json:"thinking"`
    Tools             bool `json:"tools"`
    Images            bool `json:"images"`
    JSONMode          bool `json:"json_mode"`
    SystemPrompt      bool `json:"system_prompt"`
    FunctionCalling   bool `json:"function_calling"`
}
```

### 3. `internal/proxy/interfaces.go`

**Purpose**: Proxy interface for HTTP proxy configuration

**Full Implementation**:

```go
package proxy

import (
    "context"
    "net/http"
    "net/url"
)

// Proxy defines the contract for HTTP proxying
// Each provider can have its own proxy configuration
type Proxy interface {
    // GetProxyURL returns the proxy URL for a given request
    GetProxyURL(req *http.Request) (*url.URL, error)
    
    // GetHTTPClient returns an HTTP client configured with this proxy
    GetHTTPClient() (*http.Client, error)
    
    // WrapRequest wraps a request with proxy configuration
    WrapRequest(req *http.Request) (*http.Request, error)
    
    // Validate validates the proxy configuration
    Validate() error
    
    // HealthCheck performs a health check on the proxy
    HealthCheck(ctx context.Context) error
}

// ProxyManager defines the contract for managing multiple proxies
type ProxyManager interface {
    // GetProxy returns the proxy for a specific provider
    GetProxy(providerType string) (Proxy, error)
    
    // RegisterProxy registers a proxy for a provider
    RegisterProxy(providerType string, proxy Proxy) error
    
    // RemoveProxy removes a proxy for a provider
    RemoveProxy(providerType string) error
    
    // UpdateProxy updates the proxy configuration for a provider
    UpdateProxy(providerType string, config ProxyConfig) error
    
    // GetAllProxies returns all registered proxies
    GetAllProxies() map[string]Proxy
}

// ProxyConfig defines proxy configuration
type ProxyConfig struct {
    Enabled    bool   `json:"enabled"`
    HTTPProxy  string `json:"http_proxy"`
    HTTPSProxy string `json:"https_proxy"`
    NoProxy    string `json:"no_proxy"`
}
```

## Dependencies

None - this is the foundational step with no dependencies on other refactoring steps.

## Verification

After completing this step, verify:

1. All interface files compile without errors
2. Interfaces are properly documented
3. Types are exported (capitalized) for use by other packages
4. No circular dependencies exist

## Next Steps

After completing this step, proceed to:
- **Step 02**: Shared Models (define common data structures)
- **Step 03**: Proxy Infrastructure (implement proxy interfaces)
- **Step 04**: Proxy Manager (manage multiple proxies)
