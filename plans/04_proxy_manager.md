# Step 04: Proxy Manager

## Context

This step implements a proxy manager that handles multiple proxy configurations for different providers. Each provider (Gemini, Copilot, Qwen, Antigravity) can have its own distinct proxy to prevent IP blocking.

## Objectives

1. Implement ProxyManager interface
2. Manage multiple proxy instances
3. Provide thread-safe access to proxies
4. Support dynamic proxy registration/removal
5. Initialize proxies from configuration

## Design Pattern

**Registry Pattern**: Centralized registry of proxy instances with thread-safe access and lifecycle management.

## Files to Create

### 1. `internal/proxy/manager.go`

**Purpose**: Manage multiple proxy configurations for different providers

**Full Implementation**:

```go
package proxy

import (
    "context"
    "errors"
    "fmt"
    "sync"
)

// ProxyManagerImpl implements ProxyManager interface
type ProxyManagerImpl struct {
    mu      sync.RWMutex
    proxies map[string]Proxy
}

// NewProxyManager creates a new proxy manager
func NewProxyManager() *ProxyManagerImpl {
    return &ProxyManagerImpl{
        proxies: make(map[string]Proxy),
    }
}

// InitializeFromConfig initializes proxies from environment configuration
// This method should be called during application startup
func (pm *ProxyManagerImpl) InitializeFromConfig() error {
    // Define providers to load proxies for
    providers := []string{"gemini", "copilot", "qwen", "antigravity"}
    
    for _, provider := range providers {
        cfg := NewProxyConfigFromEnv(provider)
        
        if cfg.Enabled {
            proxy, err := NewHTTPProxy(cfg)
            if err != nil {
                return fmt.Errorf("failed to create proxy for %s: %w", provider, err)
            }
            
            // Validate proxy before registering
            if err := proxy.Validate(); err != nil {
                return fmt.Errorf("invalid proxy configuration for %s: %w", provider, err)
            }
            
            if err := pm.RegisterProxy(provider, proxy); err != nil {
                return fmt.Errorf("failed to register proxy for %s: %w", provider, err)
            }
            
            fmt.Printf("[INFO] Registered proxy for provider: %s (%s)", provider, cfg)
        }
    }
    
    return nil
}

// GetProxy returns the proxy for a specific provider
// Implements ProxyManager interface
func (pm *ProxyManagerImpl) GetProxy(providerType string) (Proxy, error) {
    pm.mu.RLock()
    defer pm.mu.RUnlock()
    
    proxy, exists := pm.proxies[providerType]
    if !exists {
        return nil, fmt.Errorf("no proxy configured for provider: %s", providerType)
    }
    
    return proxy, nil
}

// GetProxyOrDefault returns the proxy for a provider or nil if not configured
func (pm *ProxyManagerImpl) GetProxyOrDefault(providerType string) Proxy {
    proxy, err := pm.GetProxy(providerType)
    if err != nil {
        return nil
    }
    return proxy
}

// RegisterProxy registers a proxy for a provider
// Implements ProxyManager interface
func (pm *ProxyManagerImpl) RegisterProxy(providerType string, proxy Proxy) error {
    if proxy == nil {
        return errors.New("proxy cannot be nil")
    }
    
    // Validate proxy configuration
    if err := proxy.Validate(); err != nil {
        return fmt.Errorf("invalid proxy configuration: %w", err)
    }
    
    pm.mu.Lock()
    defer pm.mu.Unlock()
    
    // Close existing proxy if present
    if existing, exists := pm.proxies[providerType]; exists {
        existing.Close()
    }
    
    pm.proxies[providerType] = proxy
    return nil
}

// RemoveProxy removes a proxy for a provider
// Implements ProxyManager interface
func (pm *ProxyManagerImpl) RemoveProxy(providerType string) error {
    pm.mu.Lock()
    defer pm.mu.Unlock()
    
    proxy, exists := pm.proxies[providerType]
    if !exists {
        return fmt.Errorf("no proxy configured for provider: %s", providerType)
    }
    
    // Close proxy before removing
    proxy.Close()
    
    delete(pm.proxies, providerType)
    return nil
}

// UpdateProxy updates the proxy configuration for a provider
// Implements ProxyManager interface
func (pm *ProxyManagerImpl) UpdateProxy(providerType string, cfg ProxyConfig) error {
    proxy, err := NewHTTPProxy(cfg)
    if err != nil {
        return fmt.Errorf("failed to create proxy for %s: %w", providerType, err)
    }
    
    return pm.RegisterProxy(providerType, proxy)
}

// GetAllProxies returns all registered proxies
// Implements ProxyManager interface
func (pm *ProxyManagerImpl) GetAllProxies() map[string]Proxy {
    pm.mu.RLock()
    defer pm.mu.RUnlock()
    
    result := make(map[string]Proxy, len(pm.proxies))
    for k, v := range pm.proxies {
        result[k] = v
    }
    
    return result
}

// HealthCheckAll performs health checks on all registered proxies
func (pm *ProxyManagerImpl) HealthCheckAll(ctx context.Context) map[string]error {
    pm.mu.RLock()
    defer pm.mu.RUnlock()
    
    results := make(map[string]error)
    
    for providerType, proxy := range pm.proxies {
        if err := proxy.HealthCheck(ctx); err != nil {
            results[providerType] = err
        }
    }
    
    return results
}

// Close closes all proxies and cleans up resources
func (pm *ProxyManagerImpl) Close() error {
    pm.mu.Lock()
    defer pm.mu.Unlock()
    
    var lastErr error
    
    for providerType, proxy := range pm.proxies {
        if err := proxy.Close(); err != nil {
            fmt.Printf("[WARN] Error closing proxy for %s: %v", providerType, err)
            lastErr = err
        }
    }
    
    pm.proxies = make(map[string]Proxy)
    return lastErr
}

// GetProviderTypes returns list of providers with registered proxies
func (pm *ProxyManagerImpl) GetProviderTypes() []string {
    pm.mu.RLock()
    defer pm.mu.RUnlock()
    
    types := make([]string, 0, len(pm.proxies))
    for providerType := range pm.proxies {
        types = append(types, providerType)
    }
    
    return types
}

// HasProxy checks if a proxy is registered for a provider
func (pm *ProxyManagerImpl) HasProxy(providerType string) bool {
    pm.mu.RLock()
    defer pm.mu.RUnlock()
    
    _, exists := pm.proxies[providerType]
    return exists
}
```

### 2. `internal/proxy/manager_test.go`

**Purpose**: Unit tests for proxy manager

**Full Implementation**:

```go
package proxy

import (
    "context"
    "testing"
)

func TestNewProxyManager(t *testing.T) {
    pm := NewProxyManager()
    
    assert.NotNil(t, pm)
    assert.Equal(t, 0, len(pm.GetAllProxies()))
}

func TestRegisterProxy(t *testing.T) {
    pm := NewProxyManager()
    
    cfg := ProxyConfig{
        Enabled:    false, // Disabled for testing
        HTTPProxy:  "http://proxy.example.com:8080",
    }
    
    proxy, err := NewHTTPProxy(cfg)
    assert.NoError(t, err)
    
    err = pm.RegisterProxy("test", proxy)
    assert.NoError(t, err)
    assert.True(t, pm.HasProxy("test"))
}

func TestGetProxy(t *testing.T) {
    pm := NewProxyManager()
    
    cfg := ProxyConfig{
        Enabled:    false,
        HTTPProxy:  "http://proxy.example.com:8080",
    }
    
    proxy, _ := NewHTTPProxy(cfg)
    pm.RegisterProxy("test", proxy)
    
    retrieved, err := pm.GetProxy("test")
    assert.NoError(t, err)
    assert.Equal(t, proxy, retrieved)
}

func TestGetProxyNotFound(t *testing.T) {
    pm := NewProxyManager()
    
    _, err := pm.GetProxy("nonexistent")
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "no proxy configured")
}

func TestRemoveProxy(t *testing.T) {
    pm := NewProxyManager()
    
    cfg := ProxyConfig{
        Enabled:    false,
        HTTPProxy:  "http://proxy.example.com:8080",
    }
    
    proxy, _ := NewHTTPProxy(cfg)
    pm.RegisterProxy("test", proxy)
    
    err := pm.RemoveProxy("test")
    assert.NoError(t, err)
    assert.False(t, pm.HasProxy("test"))
}

func TestUpdateProxy(t *testing.T) {
    pm := NewProxyManager()
    
    cfg1 := ProxyConfig{
        Enabled:    false,
        HTTPProxy:  "http://proxy1.example.com:8080",
    }
    
    proxy1, _ := NewHTTPProxy(cfg1)
    pm.RegisterProxy("test", proxy1)
    
    cfg2 := ProxyConfig{
        Enabled:    false,
        HTTPProxy:  "http://proxy2.example.com:8080",
    }
    
    err := pm.UpdateProxy("test", cfg2)
    assert.NoError(t, err)
    
    retrieved, _ := pm.GetProxy("test")
    assert.NotEqual(t, proxy1, retrieved)
}

func TestHealthCheckAll(t *testing.T) {
    pm := NewProxyManager()
    
    cfg := ProxyConfig{
        Enabled:    false, // Disabled for testing
        HTTPProxy:  "http://proxy.example.com:8080",
    }
    
    proxy, _ := NewHTTPProxy(cfg)
    pm.RegisterProxy("test", proxy)
    
    ctx := context.Background()
    results := pm.HealthCheckAll(ctx)
    
    assert.NotNil(t, results)
    // Disabled proxy should have no error
    assert.Nil(t, results["test"])
}

func TestClose(t *testing.T) {
    pm := NewProxyManager()
    
    cfg := ProxyConfig{
        Enabled:    false,
        HTTPProxy:  "http://proxy.example.com:8080",
    }
    
    proxy, _ := NewHTTPProxy(cfg)
    pm.RegisterProxy("test", proxy)
    
    err := pm.Close()
    assert.NoError(t, err)
    assert.Equal(t, 0, len(pm.GetAllProxies()))
}
```

## Dependencies

- **Step 01**: Core Interfaces (defines ProxyManager interface)
- **Step 03**: Proxy Infrastructure (provides HTTPProxy implementation)

## Verification

After completing this step, verify:

1. Proxy manager can be instantiated
2. Proxies can be registered and retrieved
3. Thread-safe access works correctly
4. Health checks work for all proxies
5. Cleanup properly closes all proxies

## Integration Example

```go
// Example usage in main.go:

func main() {
    // Create proxy manager
    proxyMgr := proxy.NewProxyManager()
    
    // Initialize proxies from environment
    if err := proxyMgr.InitializeFromConfig(); err != nil {
        log.Printf("Warning: Failed to initialize proxies: %v", err)
    }
    
    // Use proxy for a provider
    if proxy, err := proxyMgr.GetProxy("gemini"); err == nil {
        client, _ := proxy.GetHTTPClient()
        // Use client for requests
    }
    
    // Cleanup on shutdown
    defer proxyMgr.Close()
}
```

## Next Steps

After completing this step, proceed to:
- **Step 05**: Provider Factory (create provider instances)
- **Step 06**: Configuration Management (load provider configs)
- **Step 07**: Shared Middleware (CORS, auth, rate limiting)
