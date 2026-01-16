# Step 06: Configuration Management

## Context

This step implements configuration management for loading provider-specific settings from environment variables. Each provider (Gemini, Copilot, Qwen, Antigravity) has its own configuration namespace.

## Objectives

1. Implement provider configuration loading from environment
2. Support provider-specific settings (endpoints, auth, proxies)
3. Provide validation for configuration
4. Support default values
5. Enable runtime configuration updates

## Design Pattern

**Configuration Object Pattern**: Centralized configuration management with environment variable binding and validation.

## Files to Create

### 1. `internal/config/provider_config.go`

**Purpose**: Load and manage provider-specific configurations

**Full Implementation**:

```go
package config

import (
    "fmt"
    "os"
    "strconv"
    "strings"
    
    "gcli2apigo/internal/providers"
    "gcli2apigo/internal/proxy"
)

// GetProviderConfig returns configuration for a specific provider
func GetProviderConfig(providerType string) (providers.ProviderConfig, error) {
    prefix := strings.ToUpper(providerType)
    
    cfg := providers.ProviderConfig{
        Type:    providers.ProviderType(providerType),
        Enabled: getEnvBool(prefix+"_ENABLED", false),
        Proxy:   loadProxyConfig(prefix),
        Auth:    loadAuthConfig(prefix),
        RateLimit: loadRateLimitConfig(prefix),
    }
    
    // Set API endpoint based on provider
    cfg.APIEndpoint = getEnv(prefix+"_API_ENDPOINT", getDefaultAPIEndpoint(providerType))
    
    // Validate configuration
    if err := validateProviderConfig(cfg); err != nil {
        return providers.ProviderConfig{}, err
    }
    
    return cfg, nil
}

// GetProxyConfig returns proxy configuration for a provider
func GetProxyConfig(providerType string) (*proxy.ProxyConfig, error) {
    cfg, err := GetProviderConfig(providerType)
    if err != nil {
        return nil, err
    }
    return cfg.Proxy, nil
}

// loadProxyConfig loads proxy configuration from environment
func loadProxyConfig(prefix string) *proxy.ProxyConfig {
    return &proxy.ProxyConfig{
        Enabled:    getEnvBool(prefix+"_PROXY_ENABLED", false),
        HTTPProxy:  getEnv(prefix+"_PROXY_HTTP"),
        HTTPSProxy: getEnv(prefix+"_PROXY_HTTPS"),
        NoProxy:    getEnv(prefix+"_PROXY_NO"),
    }
}

// loadAuthConfig loads authentication configuration from environment
func loadAuthConfig(prefix string) *providers.AuthConfig {
    authType := getEnv(prefix+"_AUTH_TYPE", "oauth")
    
    cfg := &providers.AuthConfig{
        Type:          authType,
        ClientID:      getEnv(prefix+"_CLIENT_ID"),
        ClientSecret:  getEnv(prefix+"_CLIENT_SECRET"),
        TokenEndpoint: getEnv(prefix+"_TOKEN_ENDPOINT"),
        AuthEndpoint:  getEnv(prefix+"_AUTH_ENDPOINT"),
        APIKey:        getEnv(prefix+"_API_KEY"),
        JWTEndpoint:   getEnv(prefix+"_JWT_ENDPOINT"),
        JWTSecret:      getEnv(prefix+"_JWT_SECRET"),
    }
    
    // Load scopes for OAuth
    if authType == "oauth" {
        cfg.Scopes = loadScopes(prefix)
    }
    
    return cfg
}

// loadScopes loads OAuth scopes from environment
func loadScopes(prefix string) []string {
    scopesStr := getEnv(prefix+"_SCOPES", "")
    if scopesStr == "" {
        return getDefaultScopes(prefix)
    }
    
    return strings.Split(scopesStr, ",")
}

// getDefaultScopes returns default scopes for a provider
func getDefaultScopes(prefix string) []string {
    switch strings.ToLower(prefix) {
    case "gemini":
        return []string{
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
        }
    case "copilot":
        return []string{
            "read:user",
            "read:org",
            "read:project",
            "write:project",
        }
    case "qwen":
        return []string{
            "dashscope:all",
        }
    case "antigravity":
        return []string{
            "api:all",
        }
    default:
        return []string{}
    }
}

// loadRateLimitConfig loads rate limiting configuration from environment
func loadRateLimitConfig(prefix string) *providers.RateLimitConfig {
    return &providers.RateLimitConfig{
        Enabled:           getEnvBool(prefix+"_RATE_LIMIT_ENABLED", true),
        RequestsPerSecond: getEnvInt(prefix+"_RATE_LIMIT_RPS", 8),
        BurstSize:         getEnvInt(prefix+"_RATE_LIMIT_BURST", 10),
    }
}

// getDefaultAPIEndpoint returns default API endpoint for a provider
func getDefaultAPIEndpoint(providerType string) string {
    switch strings.ToLower(providerType) {
    case "gemini":
        return "https://cloudcode-pa.googleapis.com"
    case "copilot":
        return "https://api.githubcopilot.com"
    case "qwen":
        return "https://dashscope.aliyuncs.com"
    case "antigravity":
        return "https://api.antigravity.com"
    default:
        return ""
    }
}

// validateProviderConfig validates provider configuration
func validateProviderConfig(cfg providers.ProviderConfig) error {
    if cfg.APIEndpoint == "" {
        return fmt.Errorf("API endpoint is required for provider %s", cfg.Type)
    }
    
    if cfg.Enabled {
        // Validate auth configuration if required
        if cfg.Auth != nil {
            if err := validateAuthConfig(cfg.Type, cfg.Auth); err != nil {
                return err
            }
        }
        
        // Validate proxy configuration if enabled
        if cfg.Proxy != nil && cfg.Proxy.Enabled {
            if err := cfg.Proxy.Validate(); err != nil {
                return fmt.Errorf("invalid proxy configuration: %w", err)
            }
        }
    }
    
    return nil
}

// validateAuthConfig validates authentication configuration
func validateAuthConfig(providerType providers.ProviderType, authCfg *providers.AuthConfig) error {
    if authCfg == nil {
        return nil
    }
    
    switch authCfg.Type {
    case "oauth":
        if authCfg.ClientID == "" {
            return fmt.Errorf("client_id is required for OAuth authentication")
        }
        if authCfg.ClientSecret == "" {
            return fmt.Errorf("client_secret is required for OAuth authentication")
        }
        if authCfg.TokenEndpoint == "" {
            return fmt.Errorf("token_endpoint is required for OAuth authentication")
        }
        if authCfg.AuthEndpoint == "" {
            return fmt.Errorf("auth_endpoint is required for OAuth authentication")
        }
        
    case "api_key":
        if authCfg.APIKey == "" {
            return fmt.Errorf("api_key is required for API key authentication")
        }
        
    case "jwt":
        if authCfg.JWTSecret == "" {
            return fmt.Errorf("jwt_secret is required for JWT authentication")
        }
        if authCfg.JWTEndpoint == "" {
            return fmt.Errorf("jwt_endpoint is required for JWT authentication")
        }
        
    default:
        return fmt.Errorf("unsupported authentication type: %s", authCfg.Type)
    }
    
    return nil
}

// GetAllProviderConfigs returns configurations for all providers
func GetAllProviderConfigs() (map[string]providers.ProviderConfig, error) {
    providers := []string{"gemini", "copilot", "qwen", "antigravity"}
    
    configs := make(map[string]providers.ProviderConfig)
    
    for _, provider := range providers {
        cfg, err := GetProviderConfig(provider)
        if err != nil {
            // Log warning but continue
            fmt.Printf("[WARN] Failed to load config for %s: %v", provider, err)
            continue
        }
        configs[provider] = cfg
    }
    
    return configs, nil
}

// GetEnabledProviders returns list of enabled provider types
func GetEnabledProviders() []string {
    configs, _ := GetAllProviderConfigs()
    
    enabled := make([]string, 0)
    for provider, cfg := range configs {
        if cfg.Enabled {
            enabled = append(enabled, provider)
        }
    }
    
    return enabled
}

// ReloadConfig reloads configuration from environment
// This should be called when environment variables change
func ReloadConfig() {
    // Configuration is loaded on-demand, so no action needed
    // This function exists for compatibility with existing code
}

// Helper functions for environment variable access

func getEnv(key, defaultValue string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
    if value := os.Getenv(key); value != "" {
        parsed, err := strconv.ParseBool(value)
        if err == nil {
            return parsed
        }
    }
    return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
    if value := os.Getenv(key); value != "" {
        parsed, err := strconv.Atoi(value)
        if err == nil {
            return parsed
        }
    }
    return defaultValue
}

func getEnvFloat64(key string, defaultValue float64) float64 {
    if value := os.Getenv(key); value != "" {
        parsed, err := strconv.ParseFloat(value, 64)
        if err == nil {
            return parsed
        }
    }
    return defaultValue
}
```

### 2. `internal/config/provider_config_test.go`

**Purpose**: Unit tests for configuration management

**Full Implementation**:

```go
package config

import (
    "os"
    "testing"
    
    "gcli2apigo/internal/providers"
)

func TestGetProviderConfig(t *testing.T) {
    // Set environment variables
    os.Set("GEMINI_ENABLED", "true")
    os.Set("GEMINI_API_ENDPOINT", "https://test.api.com")
    defer func() {
        os.Unset("GEMINI_ENABLED")
        os.Unset("GEMINI_API_ENDPOINT")
    }()
    
    cfg, err := GetProviderConfig("gemini")
    assert.NoError(t, err)
    assert.True(t, cfg.Enabled)
    assert.Equal(t, "https://test.api.com", cfg.APIEndpoint)
}

func TestLoadProxyConfig(t *testing.T) {
    os.Set("GEMINI_PROXY_ENABLED", "true")
    os.Set("GEMINI_PROXY_HTTP", "http://proxy.example.com:8080")
    defer func() {
        os.Unset("GEMINI_PROXY_ENABLED")
        os.Unset("GEMINI_PROXY_HTTP")
    }()
    
    proxyCfg := loadProxyConfig("GEMINI")
    assert.True(t, proxyCfg.Enabled)
    assert.Equal(t, "http://proxy.example.com:8080", proxyCfg.HTTPProxy)
}

func TestLoadAuthConfig(t *testing.T) {
    os.Set("GEMINI_AUTH_TYPE", "api_key")
    os.Set("GEMINI_API_KEY", "test-key-123")
    defer func() {
        os.Unset("GEMINI_AUTH_TYPE")
        os.Unset("GEMINI_API_KEY")
    }()
    
    authCfg := loadAuthConfig("GEMINI")
    assert.Equal(t, "api_key", authCfg.Type)
    assert.Equal(t, "test-key-123", authCfg.APIKey)
}

func TestValidateAuthConfig(t *testing.T) {
    tests := []struct {
        name    string
        authCfg *providers.AuthConfig
        wantErr bool
    }{
        {
            name: "valid oauth",
            authCfg: &providers.AuthConfig{
                Type:          "oauth",
                ClientID:      "test-id",
                ClientSecret:  "test-secret",
                TokenEndpoint: "https://token.example.com",
                AuthEndpoint:  "https://auth.example.com",
            },
            wantErr: false,
        },
        {
            name: "missing client_id",
            authCfg: &providers.AuthConfig{
                Type:          "oauth",
                ClientSecret:  "test-secret",
                TokenEndpoint: "https://token.example.com",
                AuthEndpoint:  "https://auth.example.com",
            },
            wantErr: true,
        },
        {
            name: "valid api_key",
            authCfg: &providers.AuthConfig{
                Type:   "api_key",
                APIKey: "test-key",
            },
            wantErr: false,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := validateAuthConfig(providers.ProviderGemini, tt.authCfg)
            if tt.wantErr {
                assert.Error(t, err)
            } else {
                assert.NoError(t, err)
            }
        })
    }
}

func TestGetDefaultAPIEndpoint(t *testing.T) {
    tests := []struct {
        providerType string
        expected     string
    }{
        {"gemini", "https://cloudcode-pa.googleapis.com"},
        {"copilot", "https://api.githubcopilot.com"},
        {"qwen", "https://dashscope.aliyuncs.com"},
        {"antigravity", "https://api.antigravity.com"},
        {"unknown", ""},
    }
    
    for _, tt := range tests {
        t.Run(tt.providerType, func(t *testing.T) {
            endpoint := getDefaultAPIEndpoint(tt.providerType)
            assert.Equal(t, tt.expected, endpoint)
        })
    }
}
```

## Dependencies

- **Step 01**: Core Interfaces (ProviderConfig type)
- **Step 02**: Shared Models (no direct dependency)
- **Step 03**: Proxy Infrastructure (ProxyConfig type)

## Environment Variables

### Example `.env` Configuration

```bash
# Gemini Provider
GEMINI_ENABLED=true
GEMINI_API_ENDPOINT=https://cloudcode-pa.googleapis.com
GEMINI_AUTH_TYPE=oauth
GEMINI_CLIENT_ID=681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com
GEMINI_CLIENT_SECRET=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
GEMINI_TOKEN_ENDPOINT=https://oauth2.googleapis.com/token
GEMINI_AUTH_ENDPOINT=https://accounts.google.com/o/oauth2/v2/auth
GEMINI_PROXY_ENABLED=true
GEMINI_PROXY_HTTP=http://proxy.example.com:8080
GEMINI_PROXY_HTTPS=http://proxy.example.com:8080
GEMINI_RATE_LIMIT_ENABLED=true
GEMINI_RATE_LIMIT_RPS=8
GEMINI_RATE_LIMIT_BURST=10

# Copilot Provider
COPILOT_ENABLED=true
COPILOT_API_ENDPOINT=https://api.githubcopilot.com
COPILOT_AUTH_TYPE=oauth
COPILOT_CLIENT_ID=your_copilot_client_id
COPILOT_CLIENT_SECRET=your_copilot_client_secret
COPILOT_TOKEN_ENDPOINT=https://github.com/login/oauth/access_token
COPILOT_AUTH_ENDPOINT=https://github.com/login/oauth/authorize
COPILOT_PROXY_ENABLED=true
COPILOT_PROXY_HTTP=http://copilot-proxy.example.com:8080

# Qwen Provider
QWEN_ENABLED=true
QWEN_API_ENDPOINT=https://dashscope.aliyuncs.com
QWEN_AUTH_TYPE=api_key
QWEN_API_KEY=your_qwen_api_key
QWEN_PROXY_ENABLED=true
QWEN_PROXY_HTTP=http://qwen-proxy.example.com:8080

# Antigravity Provider
ANTIGRAVITY_ENABLED=true
ANTIGRAVITY_API_ENDPOINT=https://api.antigravity.com
ANTIGRAVITY_AUTH_TYPE=jwt
ANTIGRAVITY_JWT_SECRET=your_jwt_secret
ANTIGRAVITY_JWT_ENDPOINT=https://api.antigravity.com/auth
ANTIGRAVITY_PROXY_ENABLED=true
ANTIGRAVITY_PROXY_HTTP=http://antigravity-proxy.example.com:8080
```

## Verification

After completing this step, verify:

1. Configuration loads correctly from environment
2. Default values are applied when variables not set
3. Validation catches invalid configurations
4. Provider-specific settings work correctly
5. Multiple providers can be loaded simultaneously

## Next Steps

After completing this step, proceed to:
- **Step 07**: Shared Middleware (CORS, auth, rate limiting)
- **Step 08**: Provider Router (route requests to providers)
- **Step 09**: Gemini Provider Migration (refactor existing code)
