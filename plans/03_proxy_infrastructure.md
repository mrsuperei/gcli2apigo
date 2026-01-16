# Step 03: Proxy Infrastructure

## Context

This step implements the HTTP proxy infrastructure that allows each provider to route traffic through distinct proxies. This is critical for preventing IP blocking when accessing different AI provider APIs.

## Objectives

1. Implement HTTP proxy with per-provider configuration
2. Support HTTP and HTTPS proxies
3. Implement proxy bypass rules (no_proxy)
4. Provide HTTP client configured with proxy
5. Implement proxy validation and health checks

## Design Pattern

**Proxy Pattern**: Intercept and route HTTP requests through intermediate servers, hiding the original client IP address.

## Files to Create

### 1. `internal/proxy/http_proxy.go`

**Purpose**: HTTP proxy implementation supporting per-provider configuration

**Full Implementation**:

```go
package proxy

import (
    "context"
    "fmt"
    "net"
    "net/http"
    "net/url"
    "strings"
    "time"
)

// HTTPProxy implements Proxy interface for HTTP/HTTPS proxying
type HTTPProxy struct {
    config ProxyConfig
    client *http.Client
}

// NewHTTPProxy creates a new HTTP proxy from configuration
func NewHTTPProxy(cfg ProxyConfig) (*HTTPProxy, error) {
    proxy := &HTTPProxy{
        config: cfg,
    }
    
    // Create HTTP client with proxy configuration
    client, err := proxy.createHTTPClient()
    if err != nil {
        return nil, fmt.Errorf("failed to create HTTP client: %w", err)
    }
    
    proxy.client = client
    return proxy, nil
}

// createHTTPClient creates an HTTP client configured with proxy
func (p *HTTPProxy) createHTTPClient() (*http.Client, error) {
    var proxyURL *url.URL
    var err error
    
    // Determine which proxy to use based on scheme
    if p.config.HTTPSProxy != "" {
        proxyURL, err = url.Parse(p.config.HTTPSProxy)
    } else if p.config.HTTPProxy != "" {
        proxyURL, err = url.Parse(p.config.HTTPProxy)
    }
    
    if err != nil {
        return nil, fmt.Errorf("invalid proxy URL: %w", err)
    }
    
    // Create transport with proxy
    transport := &http.Transport{
        Proxy: http.ProxyURL(proxyURL),
        DialContext: (&net.Dialer{
            Timeout:   30 * time.Second,
            KeepAlive: 30 * time.Second,
        }).DialContext,
        MaxIdleConns:          100,
        IdleConnTimeout:       90 * time.Second,
        TLSHandshakeTimeout:   10 * time.Second,
        ExpectContinueTimeout:  1 * time.Second,
        ResponseHeaderTimeout:  30 * time.Second,
    }
    
    return &http.Client{
        Transport: transport,
        Timeout:   5 * time.Minute,
    }, nil
}

// GetProxyURL returns proxy URL for a request
// Implements Proxy interface
func (p *HTTPProxy) GetProxyURL(req *http.Request) (*url.URL, error) {
    if !p.config.Enabled {
        return nil, nil
    }
    
    // Check if request should bypass proxy
    if p.config.NoProxy != "" {
        host := req.URL.Hostname()
        for _, bypass := range strings.Split(p.config.NoProxy, ",") {
            if strings.TrimSpace(bypass) == host {
                return nil, nil
            }
        }
    }
    
    // Select appropriate proxy based on scheme
    proxyStr := p.config.HTTPSProxy
    if req.URL.Scheme == "http" {
        proxyStr = p.config.HTTPProxy
    }
    
    if proxyStr == "" {
        return nil, nil
    }
    
    return url.Parse(proxyStr)
}

// GetHTTPClient returns the configured HTTP client
// Implements Proxy interface
func (p *HTTPProxy) GetHTTPClient() (*http.Client, error) {
    return p.client, nil
}

// WrapRequest wraps a request with proxy configuration
// The HTTP client's transport handles actual proxying
// This method can be used for request-specific modifications
// Implements Proxy interface
func (p *HTTPProxy) WrapRequest(req *http.Request) (*http.Request, error) {
    // Add proxy-specific headers if needed
    if p.config.Enabled {
        req.Header.Set("X-Forwarded-For", req.RemoteAddr)
    }
    return req, nil
}

// Validate validates the proxy configuration
// Implements Proxy interface
func (p *HTTPProxy) Validate() error {
    if !p.config.Enabled {
        return nil
    }
    
    if p.config.HTTPProxy == "" && p.config.HTTPSProxy == "" {
        return fmt.Errorf("proxy enabled but no proxy URL configured")
    }
    
    // Validate HTTP proxy URL
    if p.config.HTTPProxy != "" {
        if _, err := url.Parse(p.config.HTTPProxy); err != nil {
            return fmt.Errorf("invalid HTTP proxy URL: %w", err)
        }
    }
    
    // Validate HTTPS proxy URL
    if p.config.HTTPSProxy != "" {
        if _, err := url.Parse(p.config.HTTPSProxy); err != nil {
            return fmt.Errorf("invalid HTTPS proxy URL: %w", err)
        }
    }
    
    return nil
}

// HealthCheck performs a health check on the proxy
// Implements Proxy interface
func (p *HTTPProxy) HealthCheck(ctx context.Context) error {
    if !p.config.Enabled {
        return nil
    }
    
    // Simple health check by making a request through proxy
    // Use a reliable endpoint for testing
    req, err := http.NewRequestWithContext(ctx, "GET", "https://www.google.com", nil)
    if err != nil {
        return fmt.Errorf("failed to create health check request: %w", err)
    }
    
    resp, err := p.client.Do(req)
    if err != nil {
        return fmt.Errorf("proxy health check failed: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("proxy health check returned status %d", resp.StatusCode)
    }
    
    return nil
}

// Close cleans up proxy resources
func (p *HTTPProxy) Close() error {
    // Close idle connections
    if transport, ok := p.client.Transport.(*http.Transport); ok {
        transport.CloseIdleConnections()
    }
    return nil
}
```

### 2. `internal/proxy/proxy_config.go`

**Purpose**: Proxy configuration management

**Full Implementation**:

```go
package proxy

import (
    "fmt"
    "os"
    "strings"
)

// ProxyConfig defines proxy configuration for a provider
type ProxyConfig struct {
    Enabled    bool   `json:"enabled"`
    HTTPProxy  string `json:"http_proxy"`
    HTTPSProxy string `json:"https_proxy"`
    NoProxy    string `json:"no_proxy"`
}

// NewProxyConfigFromEnv creates proxy configuration from environment variables
func NewProxyConfigFromEnv(providerPrefix string) ProxyConfig {
    prefix := strings.ToUpper(providerPrefix) + "_PROXY"
    
    return ProxyConfig{
        Enabled:    getEnvBool(prefix+"_ENABLED", false),
        HTTPProxy:  getEnv(prefix+"_HTTP"),
        HTTPSProxy: getEnv(prefix+"_HTTPS"),
        NoProxy:    getEnv(prefix+"_NO"),
    }
}

// Validate validates the proxy configuration
func (pc *ProxyConfig) Validate() error {
    if !pc.Enabled {
        return nil
    }
    
    if pc.HTTPProxy == "" && pc.HTTPSProxy == "" {
        return fmt.Errorf("proxy enabled but no proxy URL configured")
    }
    
    return nil
}

// String returns string representation
func (pc ProxyConfig) String() string {
    if !pc.Enabled {
        return "disabled"
    }
    
    parts := []string{}
    if pc.HTTPProxy != "" {
        parts = append(parts, fmt.Sprintf("http=%s", pc.HTTPProxy))
    }
    if pc.HTTPSProxy != "" {
        parts = append(parts, fmt.Sprintf("https=%s", pc.HTTPSProxy))
    }
    if pc.NoProxy != "" {
        parts = append(parts, fmt.Sprintf("no_proxy=%s", pc.NoProxy))
    }
    
    return strings.Join(parts, ", ")
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
        parsed, err := parseBool(value)
        if err == nil {
            return parsed
        }
    }
    return defaultValue
}

// parseBool parses a boolean string
func parseBool(s string) (bool, error) {
    switch strings.ToLower(strings.TrimSpace(s)) {
    case "1", "t", "true", "yes", "on":
        return true, nil
    case "0", "f", "false", "no", "off":
        return false, nil
    default:
        return false, fmt.Errorf("invalid boolean value: %s", s)
    }
}
```

## Dependencies

- **Step 01**: Core Interfaces (defines Proxy interface)
- **Step 02**: Shared Models (no direct dependency, but used together)

## Verification

After completing this step, verify:

1. HTTP proxy can be created from configuration
2. Proxy correctly routes HTTP and HTTPS requests
3. Bypass rules (no_proxy) work correctly
4. Health check validates proxy connectivity
5. Validation catches invalid configurations

## Testing

```go
// Example test cases to implement:

func TestHTTPProxyCreation(t *testing.T) {
    cfg := ProxyConfig{
        Enabled:    true,
        HTTPProxy:  "http://proxy.example.com:8080",
        HTTPSProxy: "http://proxy.example.com:8080",
    }
    
    proxy, err := NewHTTPProxy(cfg)
    assert.NoError(t, err)
    assert.NotNil(t, proxy)
}

func TestProxyBypass(t *testing.T) {
    cfg := ProxyConfig{
        Enabled:    true,
        HTTPProxy:  "http://proxy.example.com:8080",
        NoProxy:    "localhost,127.0.0.1",
    }
    
    proxy, _ := NewHTTPProxy(cfg)
    
    // Test bypass for localhost
    req, _ := http.NewRequest("GET", "http://localhost:8080", nil)
    proxyURL, _ := proxy.GetProxyURL(req)
    assert.Nil(t, proxyURL)
    
    // Test proxy for external URL
    req2, _ := http.NewRequest("GET", "https://api.example.com", nil)
    proxyURL2, _ := proxy.GetProxyURL(req2)
    assert.NotNil(t, proxyURL2)
}
```

## Next Steps

After completing this step, proceed to:
- **Step 04**: Proxy Manager (manage multiple proxies for different providers)
- **Step 05**: Provider Factory (create provider instances)
- **Step 06**: Configuration Management (load provider configs)
