# Step 08: Provider Router

## Context

This step implements routing mechanism that maps distinct endpoints (`/geminicli`, `/copilotcli`, `/qwencli`, `/antigravitycli`) to their respective provider handlers. All routes share common middleware.

## Objectives

1. Implement ProviderRouter for managing provider-specific routes
2. Map provider endpoints to their handlers
3. Apply shared middleware to all routes
4. Support dynamic provider registration
5. Handle OpenAI-compatible default routes

## Design Pattern

**Router Pattern**: Centralized routing logic that maps URL paths to provider-specific handlers with shared middleware.

## Files to Create

### 1. `internal/routes/provider_router.go`

**Purpose**: Route requests to different providers based on URL path

**Full Implementation**:

```go
package routes

import (
    "context"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net/http"
    "strings"
    
    "gcli2apigo/internal/models"
    "gcli2apigo/internal/providers"
)

// ProviderRouter handles routing to different providers
type ProviderRouter struct {
    registry    *providers.ProviderRegistry
    middleware  []Middleware
    defaultProvider providers.ProviderType
}

// NewProviderRouter creates a new provider router
func NewProviderRouter(registry *providers.ProviderRegistry, defaultProvider providers.ProviderType) *ProviderRouter {
    return &ProviderRouter{
        registry:       registry,
        middleware:      []Middleware{},
        defaultProvider: defaultProvider,
    }
}

// UseMiddleware adds middleware to the router
func (r *ProviderRouter) UseMiddleware(middleware ...Middleware) {
    r.middleware = append(r.middleware, middleware...)
}

// SetupRoutes configures all provider routes on a mux
func (r *ProviderRouter) SetupRoutes(mux *http.ServeMux) {
    // Health check (no auth required)
    mux.HandleFunc("/health", r.handleHealth)
    
    // Provider-specific endpoints
    r.setupProviderRoutes(mux, providers.ProviderGemini, "/geminicli")
    r.setupProviderRoutes(mux, providers.ProviderCopilot, "/copilotcli")
    r.setupProviderRoutes(mux, providers.ProviderQwen, "/qwencli")
    r.setupProviderRoutes(mux, providers.ProviderAntigravity, "/antigravitycli")
    
    // OpenAI-compatible routes (default provider)
    mux.HandleFunc("/v1/chat/completions", r.handleDefaultChatCompletions)
    mux.HandleFunc("/v1/models", r.handleDefaultListModels)
    
    // Legacy routes for backward compatibility
    mux.HandleFunc("/v1beta/models", r.handleGeminiListModels)
    mux.HandleFunc("/googleapis", r.handleGoogleAPIsInfo)
    mux.HandleFunc("/googleapis/", r.handleGoogleAPIsProxy)
}

// setupProviderRoutes sets up routes for a specific provider
func (r *ProviderRouter) setupProviderRoutes(mux *http.ServeMux, pType providers.ProviderType, basePath string) {
    // Apply middleware chain
    handler := r.applyMiddleware(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
        r.handleProviderRequest(w, req, pType)
    }))
    
    // Register routes
    mux.HandleFunc(basePath+"/chat/completions", handler.ServeHTTP)
    mux.HandleFunc(basePath+"/models", handler.ServeHTTP)
    mux.HandleFunc(basePath+"/", handler.ServeHTTP)
}

// handleProviderRequest routes requests to the appropriate provider handler
func (r *ProviderRouter) handleProviderRequest(w http.ResponseWriter, req *http.Request, pType providers.ProviderType) {
    provider, err := r.registry.Get(pType)
    if err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "not_found",
            Message: fmt.Sprintf("Provider %s not configured", pType),
            Code:    404,
        }, http.StatusNotFound)
        return
    }
    
    if !provider.GetConfig().Enabled {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "service_unavailable",
            Message: fmt.Sprintf("Provider %s is disabled", pType),
            Code:    503,
        }, http.StatusServiceUnavailable)
        return
    }
    
    // Route based on request path
    path := strings.TrimPrefix(req.URL.Path, "/"+string(pType))
    
    switch {
    case path == "/chat/completions":
        r.handleChatCompletions(w, req, provider)
    case path == "/models":
        r.handleListModels(w, req, provider)
    default:
        r.handleProxy(w, req, provider)
    }
}

// handleDefaultChatCompletions handles OpenAI-compatible chat completions
func (r *ProviderRouter) handleDefaultChatCompletions(w http.ResponseWriter, req *http.Request) {
    provider, err := r.registry.Get(r.defaultProvider)
    if err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "not_found",
            Message: fmt.Sprintf("Default provider %s not configured", r.defaultProvider),
            Code:    404,
        }, http.StatusNotFound)
        return
    }
    
    r.handleChatCompletions(w, req, provider)
}

// handleDefaultListModels handles OpenAI-compatible models list
func (r *ProviderRouter) handleDefaultListModels(w http.ResponseWriter, req *http.Request) {
    provider, err := r.registry.Get(r.defaultProvider)
    if err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "not_found",
            Message: fmt.Sprintf("Default provider %s not configured", r.defaultProvider),
            Code:    404,
        }, http.StatusNotFound)
        return
    }
    
    r.handleListModels(w, req, provider)
}

// handleChatCompletions handles chat completion requests
func (r *ProviderRouter) handleChatCompletions(w http.ResponseWriter, req *http.Request, provider providers.Provider) {
    if req.Method != http.MethodPost {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "method_not_allowed",
            Message: "Method not allowed",
            Code:    405,
        }, http.StatusMethodNotAllowed)
        return
    }
    
    // Read request body
    body, err := io.ReadAll(req.Body)
    if err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "invalid_request",
            Message: "Failed to read request body",
            Code:    400,
        }, http.StatusBadRequest)
        return
    }
    
    // Parse OpenAI request
    var openaiReq models.OpenAIChatCompletionRequest
    if err := json.Unmarshal(body, &openaiReq); err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "invalid_request",
            Message: "Invalid JSON in request body",
            Code:    400,
        }, http.StatusBadRequest)
        return
    }
    
    // Validate request
    if err := openaiReq.ValidateRequest(); err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "validation_error",
            Message: err.Error(),
            Code:    400,
        }, http.StatusBadRequest)
        return
    }
    
    // Convert to provider request format
    providerReq := &providers.ChatCompletionRequest{
        Model:            openaiReq.Model,
        Messages:         convertMessages(openaiReq.Messages),
        Stream:           openaiReq.Stream,
        Temperature:      openaiReq.Temperature,
        TopP:             openaiReq.TopP,
        MaxTokens:        openaiReq.MaxTokens,
        Stop:             openaiReq.Stop,
        Tools:            convertTools(openaiReq.Tools),
        ToolChoice:       openaiReq.ToolChoice,
        ResponseFormat:   openaiReq.ResponseFormat,
    }
    
    // Handle streaming vs non-streaming
    if openaiReq.Stream {
        r.handleStreamingChatCompletion(w, req, provider, providerReq)
    } else {
        r.handleNonStreamingChatCompletion(w, req, provider, providerReq)
    }
}

// handleStreamingChatCompletion handles streaming chat completion
func (r *ProviderRouter) handleStreamingChatCompletion(w http.ResponseWriter, req *http.Request, provider providers.Provider, providerReq *providers.ChatCompletionRequest) {
    // Set SSE headers
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")
    w.Header().Set("X-Accel-Buffering", "no")
    
    flusher, ok := w.(http.Flusher)
    if !ok {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "internal_error",
            Message: "Streaming not supported",
            Code:    500,
        }, http.StatusInternalServerError)
        return
    }
    
    // Call provider's streaming handler
    ctx := req.Context()
    streamChan, err := provider.HandleStreamingChatCompletion(ctx, providerReq)
    if err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "api_error",
            Message: fmt.Sprintf("Provider error: %v", err),
            Code:    500,
        }, http.StatusInternalServerError)
        return
    }
    
    // Stream chunks to client
    for chunk := range streamChan {
        fmt.Fprintf(w, "data: %s\n\n", chunk)
        flusher.Flush()
    }
    
    fmt.Fprintf(w, "data: [DONE]\n\n")
    flusher.Flush()
}

// handleNonStreamingChatCompletion handles non-streaming chat completion
func (r *ProviderRouter) handleNonStreamingChatCompletion(w http.ResponseWriter, req *http.Request, provider providers.Provider, providerReq *providers.ChatCompletionRequest) {
    ctx := req.Context()
    response, err := provider.HandleChatCompletion(ctx, providerReq)
    if err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "api_error",
            Message: fmt.Sprintf("Provider error: %v", err),
            Code:    500,
        }, http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// handleListModels handles model list requests
func (r *ProviderRouter) handleListModels(w http.ResponseWriter, req *http.Request, provider providers.Provider) {
    if req.Method != http.MethodGet {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "method_not_allowed",
            Message: "Method not allowed",
            Code:    405,
        }, http.StatusMethodNotAllowed)
        return
    }
    
    ctx := req.Context()
    models, err := provider.ListModels(ctx)
    if err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "api_error",
            Message: fmt.Sprintf("Failed to list models: %v", err),
            Code:    500,
        }, http.StatusInternalServerError)
        return
    }
    
    response := models.ModelsListResponse{
        Object: "list",
        Data:   models,
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// handleProxy handles generic proxy requests to provider
func (r *ProviderRouter) handleProxy(w http.ResponseWriter, req *http.Request, provider providers.Provider) {
    // This will be implemented by each provider
    // For now, return not found
    sendErrorResponse(w, models.ErrorDetail{
        Type:    "not_found",
        Message: "Endpoint not found",
        Code:    404,
    }, http.StatusNotFound)
}

// handleHealth handles health check requests
func (r *ProviderRouter) handleHealth(w http.ResponseWriter, req *http.Request) {
    response := map[string]string{
        "status":  "healthy",
        "service": "gcli2apigo",
        "version": "2.0.0",
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// handleGeminiListModels handles legacy Gemini models endpoint
func (r *ProviderRouter) handleGeminiListModels(w http.ResponseWriter, req *http.Request) {
    provider, err := r.registry.Get(providers.ProviderGemini)
    if err != nil {
        sendErrorResponse(w, models.ErrorDetail{
            Type:    "not_found",
            Message: "Gemini provider not configured",
            Code:    404,
        }, http.StatusNotFound)
        return
    }
    
    r.handleListModels(w, req, provider)
}

// handleGoogleAPIsInfo handles Google APIs info endpoint
func (r *ProviderRouter) handleGoogleAPIsInfo(w http.ResponseWriter, req *http.Request) {
    response := map[string]interface{}{
        "name":        "gcli2apigo",
        "description": "Multi-provider API proxy for AI models",
        "version":     "2.0.0",
        "providers":   r.registry.GetProviderTypes(),
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// handleGoogleAPIsProxy handles Google APIs proxy endpoint
func (r *ProviderRouter) handleGoogleAPIsProxy(w http.ResponseWriter, req *http.Request) {
    // This will be implemented by Gemini provider
    sendErrorResponse(w, models.ErrorDetail{
        Type:    "not_found",
        Message: "Google APIs proxy not implemented yet",
        Code:    501,
    }, http.StatusNotImplemented)
}

// applyMiddleware applies middleware chain to a handler
func (r *ProviderRouter) applyMiddleware(handler http.Handler) http.Handler {
    for i := len(r.middleware) - 1; i >= 0; i-- {
        handler = r.middleware[i](handler)
    }
    return handler
}

// Helper functions for type conversion

func convertMessages(openaiMsgs []models.OpenAIChatMessage) []providers.ChatMessage {
    msgs := make([]providers.ChatMessage, len(openaiMsgs))
    for i, msg := range openaiMsgs {
        msgs[i] = providers.ChatMessage{
            Role:             msg.Role,
            Content:          msg.Content,
            ReasoningContent: msg.ReasoningContent,
            ToolCalls:        convertToolCalls(msg.ToolCalls),
            ToolCallID:       msg.ToolCallID,
            Name:             msg.Name,
        }
    }
    return msgs
}

func convertTools(openaiTools []models.Tool) []providers.Tool {
    tools := make([]providers.Tool, len(openaiTools))
    for i, tool := range openaiTools {
        tools[i] = providers.Tool{
            Type:     tool.Type,
            Function: providers.Function{
                Name:        tool.Function.Name,
                Description: tool.Function.Description,
                Parameters:  tool.Function.Parameters,
            },
        }
    }
    return tools
}

func convertToolCalls(openaiCalls []models.ToolCall) []providers.ToolCall {
    calls := make([]providers.ToolCall, len(openaiCalls))
    for i, call := range openaiCalls {
        calls[i] = providers.ToolCall{
            ID:   call.ID,
            Type: call.Type,
            Function: providers.Function{
                Name:      call.Function.Name,
                Arguments: call.Function.Arguments,
            },
        }
    }
    return calls
}
```

## Dependencies

- **Step 01**: Core Interfaces (Provider interface)
- **Step 02**: Shared Models (request/response types)
- **Step 05**: Provider Factory (ProviderRegistry)
- **Step 07**: Shared Middleware (middleware functions)

## Route Mapping

| Endpoint Pattern | Provider | Handler |
|----------------|----------|----------|
| `/geminicli/*` | Gemini | Gemini provider handlers |
| `/copilotcli/*` | Copilot | Copilot provider handlers |
| `/qwencli/*` | Qwen | Qwen provider handlers |
| `/antigravitycli/*` | Antigravity | Antigravity provider handlers |
| `/v1/chat/completions` | Default | Default provider handler |
| `/v1/models` | Default | Default provider handler |

## Verification

After completing this step, verify:

1. Routes are registered correctly
2. Middleware is applied to all routes
3. Provider-specific endpoints work
4. Default routes work
5. Error responses are consistent

## Next Steps

After completing this step, proceed to:
- **Step 09**: Gemini Provider Migration (refactor existing code)
- **Step 10**: Copilot Provider Implementation
- **Step 11**: Qwen Provider Implementation
