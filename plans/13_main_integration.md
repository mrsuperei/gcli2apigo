# Step 13: Main.go Integration

## Context

This step integrates all the refactored components into the main application. The [`main.go`](../main.go:1) file will be updated to use the new multi-provider architecture.

## Objectives

1. Initialize shared dependencies (proxy manager, auth manager, factory)
2. Create and register all providers
3. Set up provider router with middleware
4. Configure HTTP server with all routes
5. Implement graceful shutdown

## Design Pattern

**Dependency Injection**: Main function creates and injects dependencies into components, enabling loose coupling and testability.

## File to Modify

### `main.go`

**Purpose**: Application entry point with multi-provider support

**Full Implementation**:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
    
    "gcli2apigo/internal/auth"
    "gcli2apigo/internal/config"
    "gcli2apigo/internal/dashboard"
    "gcli2apigo/internal/providers"
    "gcli2apigo/internal/proxy"
    "gcli2apigo/internal/routes"
)

func main() {
    // Configure logging
    log.SetFlags(log.LstdFlags | log.Lshortfile)
    
    // Explicitly load .env file
    if err := loadEnvFile(); err != nil {
        log.Printf("[WARN] No .env file found or error loading: %v", err)
        log.Printf("[INFO] Using default configuration or environment variables")
    } else {
        log.Printf("[INFO] Loaded configuration from .env file")
    }
    
    // Reload config to pick up values from .env
    config.ReloadConfig()
    log.Printf("[INFO] Configuration: HOST=%s, PORT=%s, MAX_RETRY_ATTEMPTS=%s",
        os.Getenv("HOST"), os.Getenv("PORT"), os.Getenv("MAX_RETRY_ATTEMPTS"))
    
    // Initialize shared dependencies
    proxyMgr := initializeProxyManager()
    authMgr := initializeAuthManager()
    
    // Create provider dependencies
    providerDeps := providers.ProviderDependencies{
        ProxyManager: proxyMgr,
        AuthManager:  authMgr,
    }
    
    // Create provider factory
    factory := providers.NewProviderFactory(providerDeps)
    log.Printf("[INFO] Provider factory created with %d registered providers", factory.GetProviderCount())
    
    // Create all enabled providers
    providerInstances, err := factory.CreateAllProvidersFromEnv()
    if err != nil {
        log.Fatalf("[ERROR] Failed to create providers: %v", err)
    }
    
    log.Printf("[INFO] Created %d providers: %v", len(providerInstances), getProviderNames(providerInstances))
    
    // Register providers in registry
    registry := providers.NewProviderRegistry()
    for pType, provider := range providerInstances {
        if err := registry.Register(provider); err != nil {
            log.Printf("[WARN] Failed to register provider %s: %v", pType, err)
        }
    }
    
    // Initialize dashboard handlers
    dashboardHandlers := dashboard.NewDashboardHandlers()
    oauthHandler := dashboard.NewOAuthHandler()
    
    // Setup routes
    mux := setupRoutes(registry, dashboardHandlers, oauthHandler)
    
    // Wrap with CORS middleware
    handler := routes.CORSMiddleware(mux)
    
    // Setup graceful shutdown
    setupGracefulShutdown(proxyMgr, registry)
    
    // Get server configuration
    host := os.Getenv("HOST")
    if host == "" {
        host = "0.0.0.0"
    }
    
    port := os.Getenv("PORT")
    if port == "" {
        port = "7860"
    }
    
    // Start server
    addr := fmt.Sprintf("%s:%s", host, port)
    log.Printf("[INFO] Starting multi-provider API server...")
    log.Printf("[INFO] Server listening on %s", addr)
    log.Printf("[INFO] Authentication required - Password: see .env file")
    
    if err := http.ListenAndServe(addr, handler); err != nil {
        log.Fatalf("[ERROR] Server failed to start: %v", err)
    }
}

// loadEnvFile loads the .env file
func loadEnvFile() error {
    return godotenv.Load()
}

// initializeProxyManager creates and initializes the proxy manager
func initializeProxyManager() *proxy.ProxyManagerImpl {
    proxyMgr := proxy.NewProxyManager()
    
    // Initialize proxies from environment
    if err := proxyMgr.InitializeFromConfig(); err != nil {
        log.Printf("[WARN] Failed to initialize proxies: %v", err)
    } else {
        log.Printf("[INFO] Proxies initialized for providers: %v", proxyMgr.GetProviderTypes())
    }
    
    return proxyMgr
}

// initializeAuthManager creates and initializes the auth manager
func initializeAuthManager() *auth.AuthManager {
    // Create auth manager with existing auth infrastructure
    authMgr := auth.NewAuthManager()
    
    // Initialize credential pool
    if err := auth.InitializeCredentialPool(); err != nil {
        log.Printf("[WARN] Credential pool initialization error: %v", err)
    } else {
        log.Printf("[INFO] Credential pool initialized")
    }
    
    return authMgr
}

// setupRoutes configures all application routes
func setupRoutes(registry *providers.ProviderRegistry, dashboardHandlers *dashboard.DashboardHandlers, oauthHandler *dashboard.OAuthHandler) *http.ServeMux {
    mux := http.NewServeMux()
    
    // Health check endpoint (no auth required)
    mux.HandleFunc("/health", handleHealth)
    
    // Dashboard routes
    mux.HandleFunc("/dashboard/login", dashboardHandlers.HandleLogin)
    mux.HandleFunc("/dashboard/logout", dashboardHandlers.HandleLogout)
    mux.HandleFunc("/dashboard/oauth/start", dashboardHandlers.RequireAuth(oauthHandler.StartOAuthFlow))
    mux.HandleFunc("/dashboard/oauth/callback", oauthHandler.HandleCallback)
    mux.HandleFunc("/dashboard/oauth/process", oauthHandler.HandleOAuthProcess)
    mux.HandleFunc("/dashboard/api/credentials", dashboardHandlers.RequireAuth(func(w http.ResponseWriter, r *http.Request) {
        if r.Method == http.MethodGet {
            dashboardHandlers.HandleListCredentials(w, r)
        } else if r.Method == http.MethodDelete {
            http.Error(w, `{"error":{"message":"Project ID required","code":400}}`, http.StatusBadRequest)
        } else {
            http.Error(w, `{"error":{"message":"Method not allowed","code":405}}`, http.StatusMethodNotAllowed)
        }
    }))
    mux.HandleFunc("/dashboard/api/credentials/upload", dashboardHandlers.RequireAuth(dashboardHandlers.HandleUploadCredentials))
    mux.HandleFunc("/dashboard/api/credentials/ban", dashboardHandlers.RequireAuth(dashboardHandlers.HandleBanCredential))
    mux.HandleFunc("/dashboard/api/credentials/unban", dashboardHandlers.RequireAuth(dashboardHandlers.HandleUnbanCredential))
    mux.HandleFunc("/dashboard/api/stats", dashboardHandlers.RequireAuth(dashboardHandlers.HandleDashboardStats))
    mux.HandleFunc("/dashboard/api/language", dashboardHandlers.HandleSetLanguage)
    mux.HandleFunc("/dashboard/api/translations", dashboardHandlers.HandleGetTranslations)
    mux.HandleFunc("/dashboard/api/settings", dashboardHandlers.RequireAuth(func(w http.ResponseWriter, r *http.Request) {
        if r.Method == http.MethodGet {
            dashboardHandlers.HandleGetSettings(w, r)
        } else if r.Method == http.MethodPost {
            dashboardHandlers.HandleSaveSettings(w, r)
        } else {
            http.Error(w, `{"error":{"message":"Method not allowed","code":405}}`, http.StatusMethodNotAllowed)
        }
    }))
    mux.HandleFunc("/dashboard/api/credentials/", dashboardHandlers.RequireAuth(func(w http.ResponseWriter, r *http.Request) {
        if r.Method == http.MethodDelete {
            dashboardHandlers.HandleDeleteCredential(w, r)
        } else {
            http.Error(w, `{"error":{"message":"Method not allowed","code":405}}`, http.StatusMethodNotAllowed)
        }
    }))
    
    // Create provider router
    defaultProvider := providers.ProviderGemini // Default to Gemini
    providerRouter := routes.NewProviderRouter(registry, defaultProvider)
    
    // Apply middleware to provider router
    providerRouter.UseMiddleware(
        routes.LoggingMiddleware,
        routes.AuthMiddleware,
        routes.ContentTypeMiddleware,
    )
    
    // Setup provider-specific routes
    providerRouter.SetupRoutes(mux)
    
    // Root handler
    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        if r.URL.Path == "/" {
            handleRoot(w, r, dashboardHandlers)
        } else {
            // Provider router handles all other routes
            providerRouter.ServeHTTP(w, r)
        }
    })
    
    return mux
}

// handleHealth handles health check requests
func handleHealth(w http.ResponseWriter, r *http.Request) {
    response := map[string]string{
        "status":   "healthy",
        "service":  "gcli2apigo",
        "version":  "2.0.0",
        "providers": []string{"gemini", "copilot", "qwen", "antigravity"},
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// handleRoot handles root path requests
func handleRoot(w http.ResponseWriter, r *http.Request, dashboardHandlers *dashboard.DashboardHandlers) {
    if r.URL.Path != "/" {
        http.NotFound(w, r)
        return
    }
    
    // Content negotiation: check Accept header to determine response type
    acceptHeader := r.Header.Get("Accept")
    
    // If client explicitly requests JSON, return API info
    if contains(acceptHeader, "application/json") {
        handleAPIInfo(w)
        return
    }
    
    // For browser requests (text/html or no specific Accept header), show dashboard
    // Check if user is authenticated
    cookie, err := r.Cookie("session_id")
    if err != nil || cookie.Value == "" || !dashboardHandlers.GetSessionManager().ValidateSession(cookie.Value) {
        // Not authenticated, show login page
        lang := i18n.GetLanguageFromRequest(r)
        dashboard.RenderLogin(w, "", lang)
        return
    }
    
    // Authenticated, show dashboard
    dashboardHandlers.HandleDashboard(w, r)
}

// handleAPIInfo returns API information
func handleAPIInfo(w http.ResponseWriter) {
    response := map[string]interface{}{
        "name":        "gcli2apigo",
        "description": "Multi-provider API proxy for AI models (Gemini, Copilot, Qwen, Antigravity)",
        "purpose":     "Provides OpenAI-compatible endpoints and native provider endpoints for accessing multiple AI model providers",
        "version":     "2.0.0",
        "endpoints": map[string]interface{}{
            "openai_compatible": map[string]string{
                "chat_completions": "/v1/chat/completions",
                "models":           "/v1/models",
            },
            "provider_specific": map[string]interface{}{
                "gemini": map[string]string{
                    "chat_completions": "/geminicli/chat/completions",
                    "models":           "/geminicli/models",
                },
                "copilot": map[string]string{
                    "chat_completions": "/copilotcli/chat/completions",
                    "models":           "/copilotcli/models",
                },
                "qwen": map[string]string{
                    "chat_completions": "/qwencli/chat/completions",
                    "models":           "/qwencli/models",
                },
                "antigravity": map[string]string{
                    "chat_completions": "/antigravitycli/chat/completions",
                    "models":           "/antigravitycli/models",
                },
            },
            "dashboard": map[string]string{
                "login":       "/dashboard/login",
                "logout":      "/dashboard/logout",
                "oauth_start": "/dashboard/oauth/start",
                "credentials": "/dashboard/api/credentials",
                "stats":       "/dashboard/api/stats",
            },
            "health": "/health",
        },
        "authentication": "Required for all endpoints except root and health",
        "repository":     "https://github.com/Hype3808/gcli2apigo",
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// setupGracefulShutdown sets up signal handlers for graceful shutdown
func setupGracefulShutdown(proxyMgr *proxy.ProxyManagerImpl, registry *providers.ProviderRegistry) {
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, os.Interrupt)
    
    go func() {
        <-sigChan
        log.Println("[INFO] Received interrupt signal, shutting down gracefully...")
        
        // Close all providers
        if err := registry.CloseAll(); err != nil {
            log.Printf("[WARN] Error closing providers: %v", err)
        } else {
            log.Println("[INFO] All providers closed successfully")
        }
        
        // Close proxy manager
        if err := proxyMgr.Close(); err != nil {
            log.Printf("[WARN] Error closing proxy manager: %v", err)
        } else {
            log.Println("[INFO] Proxy manager closed successfully")
        }
        
        // Save usage stats
        if err := usage.GetTracker().Save(); err != nil {
            log.Printf("[WARN] Failed to save usage stats: %v", err)
        } else {
            log.Println("[INFO] Usage stats saved successfully")
        }
        
        // Save banlist
        if err := banlist.GetBanList().Save(); err != nil {
            log.Printf("[WARN] Failed to save banlist: %v", err)
        } else {
            log.Println("[INFO] Banlist saved successfully")
        }
        
        log.Println("[INFO] Shutdown complete")
        os.Exit(0)
    }()
}

// getProviderNames returns list of provider names
func getProviderNames(providers map[providers.ProviderType]providers.Provider) []string {
    names := make([]string, 0, len(providers))
    for pType := range providers {
        names = append(names, string(pType))
    }
    return names
}

// contains checks if a string contains a substring
func contains(s, substr string) bool {
    return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && s[:len(substr)] == substr || s[len(s)-len(substr):] == substr)
}
```

## Dependencies

- **Step 01**: Core Interfaces (Provider interface)
- **Step 02**: Shared Models (request/response types)
- **Step 03**: Proxy Infrastructure (ProxyManager)
- **Step 04**: Proxy Manager (ProxyManagerImpl)
- **Step 05**: Provider Factory (ProviderFactory, ProviderRegistry)
- **Step 06**: Configuration Management (GetProviderConfig)
- **Step 07**: Shared Middleware (middleware functions)
- **Step 08**: Provider Router (ProviderRouter)
- **Step 09**: Gemini Provider (GeminiProvider)
- **Step 10**: Copilot Provider (CopilotProvider)
- **Step 11**: Qwen Provider (QwenProvider)
- **Step 12**: Antigravity Provider (AntigravityProvider)

## Route Summary

After integration, the application will support the following routes:

| Endpoint Pattern | Provider | Description |
|----------------|----------|-------------|
| `/health` | All | Health check |
| `/v1/chat/completions` | Default (Gemini) | OpenAI-compatible chat |
| `/v1/models` | Default (Gemini) | OpenAI-compatible models |
| `/geminicli/*` | Gemini | Gemini-specific endpoints |
| `/copilotcli/*` | Copilot | Copilot-specific endpoints |
| `/qwencli/*` | Qwen | Qwen-specific endpoints |
| `/antigravitycli/*` | Antigravity | Antigravity-specific endpoints |
| `/dashboard/*` | All | Dashboard and management |

## Verification

After completing this step, verify:

1. Application starts without errors
2. All providers are initialized correctly
3. Routes are registered for all providers
4. Health check returns provider list
5. Dashboard is accessible
6. Graceful shutdown works correctly

## Testing

```bash
# Test health check
curl http://localhost:7860/health

# Test OpenAI-compatible endpoint (default: Gemini)
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Test provider-specific endpoint
curl -X POST http://localhost:7860/copilotcli/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Next Steps

After completing this step, proceed to:
- **Step 14**: Testing and Documentation
- **Step 15**: Deployment Guide
