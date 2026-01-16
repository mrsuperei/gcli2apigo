# Step 05: Provider Factory

## Context

This step implements the Factory pattern for creating provider instances dynamically. The factory allows the system to instantiate providers based on configuration without hard-coding provider-specific logic.

## Objectives

1. Implement ProviderFactory with registration mechanism
2. Support dynamic provider creation from configuration
3. Register built-in providers (Gemini, Copilot, Qwen, Antigravity)
4. Enable runtime provider switching
5. Support extensibility for future providers

## Design Pattern

**Factory Pattern**: Encapsulate object creation logic, allowing clients to request objects without knowing concrete implementation details.

## Files to Create

### 1. `internal/providers/factory.go`

**Purpose**: Create provider instances dynamically based on configuration

**Full Implementation**:

```go
package providers

import (
    "errors"
    "fmt"
    "sync"
)

// ProviderCreator is a function that creates a provider instance
// This allows registration of new providers without modifying factory code
type ProviderCreator func(cfg ProviderConfig, dependencies ProviderDependencies) (Provider, error)

// ProviderDependencies contains shared dependencies needed by providers
type ProviderDependencies struct {
    ProxyManager    ProxyManager
    AuthManager     AuthManager
    RateLimiter     RateLimiter
    UsageTracker     UsageTracker
}

// ProviderFactory manages provider creation and registration
type ProviderFactory struct {
    mu         sync.RWMutex
    registry    map[ProviderType]ProviderCreator
    deps        ProviderDependencies
}

// NewProviderFactory creates a new provider factory with dependencies
func NewProviderFactory(deps ProviderDependencies) *ProviderFactory {
    factory := &ProviderFactory{
        registry: make(map[ProviderType]ProviderCreator),
        deps:     deps,
    }
    
    // Register built-in providers
    factory.registerBuiltinProviders()
    
    return factory
}

// registerBuiltinProviders registers all built-in providers
func (f *ProviderFactory) registerBuiltinProviders() {
    // These will be implemented in later steps
    // For now, we register placeholders
    
    // Gemini provider
    f.RegisterProvider(ProviderGemini, func(cfg ProviderConfig, deps ProviderDependencies) (Provider, error) {
        return NewGeminiProvider(cfg, deps)
    })
    
    // Copilot provider
    f.RegisterProvider(ProviderCopilot, func(cfg ProviderConfig, deps ProviderDependencies) (Provider, error) {
        return NewCopilotProvider(cfg, deps)
    })
    
    // Qwen provider
    f.RegisterProvider(ProviderQwen, func(cfg ProviderConfig, deps ProviderDependencies) (Provider, error) {
        return NewQwenProvider(cfg, deps)
    })
    
    // Antigravity provider
    f.RegisterProvider(ProviderAntigravity, func(cfg ProviderConfig, deps ProviderDependencies) (Provider, error) {
        return NewAntigravityProvider(cfg, deps)
    })
}

// RegisterProvider registers a new provider creator
// This allows adding new providers without modifying factory code
func (f *ProviderFactory) RegisterProvider(pType ProviderType, creator ProviderCreator) error {
    if creator == nil {
        return errors.New("creator function cannot be nil")
    }
    
    f.mu.Lock()
    defer f.mu.Unlock()
    
    f.registry[pType] = creator
    return nil
}

// UnregisterProvider removes a provider from the registry
func (f *ProviderFactory) UnregisterProvider(pType ProviderType) error {
    f.mu.Lock()
    defer f.mu.Unlock()
    
    if _, exists := f.registry[pType]; !exists {
        return fmt.Errorf("provider %s is not registered", pType)
    }
    
    delete(f.registry, pType)
    return nil
}

// CreateProvider creates a provider instance from configuration
func (f *ProviderFactory) CreateProvider(cfg ProviderConfig) (Provider, error) {
    f.mu.RLock()
    creator, exists := f.registry[cfg.Type]
    f.mu.RUnlock()
    
    if !exists {
        return nil, fmt.Errorf("unsupported provider type: %s", cfg.Type)
    }
    
    provider, err := creator(cfg, f.deps)
    if err != nil {
        return nil, fmt.Errorf("failed to create provider %s: %w", cfg.Type, err)
    }
    
    return provider, nil
}

// CreateProviderFromEnv creates a provider from environment variables
func (f *ProviderFactory) CreateProviderFromEnv(pType ProviderType) (Provider, error) {
    cfg, err := GetProviderConfigFromEnv(string(pType))
    if err != nil {
        return nil, fmt.Errorf("failed to load config for %s: %w", pType, err)
    }
    
    return f.CreateProvider(cfg)
}

// CreateAllProvidersFromEnv creates all enabled providers from environment
func (f *ProviderFactory) CreateAllProvidersFromEnv() (map[ProviderType]Provider, error) {
    providers := make(map[ProviderType]Provider)
    
    // List of all provider types
    providerTypes := []ProviderType{
        ProviderGemini,
        ProviderCopilot,
        ProviderQwen,
        ProviderAntigravity,
    }
    
    for _, pType := range providerTypes {
        cfg, err := GetProviderConfigFromEnv(string(pType))
        if err != nil {
            // Skip providers with invalid config
            continue
        }
        
        if !cfg.Enabled {
            // Skip disabled providers
            continue
        }
        
        provider, err := f.CreateProvider(cfg)
        if err != nil {
            return nil, fmt.Errorf("failed to create provider %s: %w", pType, err)
        }
        
        providers[pType] = provider
    }
    
    return providers, nil
}

// GetSupportedProviders returns list of supported provider types
func (f *ProviderFactory) GetSupportedProviders() []ProviderType {
    f.mu.RLock()
    defer f.mu.RUnlock()
    
    types := make([]ProviderType, 0, len(f.registry))
    for t := range f.registry {
        types = append(types, t)
    }
    
    return types
}

// IsProviderSupported checks if a provider type is supported
func (f *ProviderFactory) IsProviderSupported(pType ProviderType) bool {
    f.mu.RLock()
    defer f.mu.RUnlock()
    
    _, exists := f.registry[pType]
    return exists
}

// GetProviderCount returns the number of registered providers
func (f *ProviderFactory) GetProviderCount() int {
    f.mu.RLock()
    defer f.mu.RUnlock()
    
    return len(f.registry)
}

// ClearRegistry removes all registered providers
// Use with caution - this removes all providers including built-in ones
func (f *ProviderFactory) ClearRegistry() {
    f.mu.Lock()
    defer f.mu.Unlock()
    
    f.registry = make(map[ProviderType]ProviderCreator)
}

// GetDependencies returns the factory's dependencies
func (f *ProviderFactory) GetDependencies() ProviderDependencies {
    return f.deps
}
```

### 2. `internal/providers/registry.go`

**Purpose**: Provider registry for managing active provider instances

**Full Implementation**:

```go
package providers

import (
    "sync"
)

// ProviderRegistry manages active provider instances
type ProviderRegistry struct {
    mu        sync.RWMutex
    providers map[ProviderType]Provider
}

// NewProviderRegistry creates a new provider registry
func NewProviderRegistry() *ProviderRegistry {
    return &ProviderRegistry{
        providers: make(map[ProviderType]Provider),
    }
}

// Register adds a provider to the registry
func (pr *ProviderRegistry) Register(provider Provider) error {
    if provider == nil {
        return errors.New("provider cannot be nil")
    }
    
    pr.mu.Lock()
    defer pr.mu.Unlock()
    
    pType := provider.GetType()
    
    // Close existing provider if present
    if existing, exists := pr.providers[pType]; exists {
        existing.Close()
    }
    
    pr.providers[pType] = provider
    return nil
}

// Get retrieves a provider from the registry
func (pr *ProviderRegistry) Get(pType ProviderType) (Provider, error) {
    pr.mu.RLock()
    defer pr.mu.RUnlock()
    
    provider, exists := pr.providers[pType]
    if !exists {
        return nil, fmt.Errorf("provider %s is not registered", pType)
    }
    
    return provider, nil
}

// GetOrDefault retrieves a provider or nil if not registered
func (pr *ProviderRegistry) GetOrDefault(pType ProviderType) Provider {
    provider, err := pr.Get(pType)
    if err != nil {
        return nil
    }
    return provider
}

// Remove removes a provider from the registry
func (pr *ProviderRegistry) Remove(pType ProviderType) error {
    pr.mu.Lock()
    defer pr.mu.Unlock()
    
    provider, exists := pr.providers[pType]
    if !exists {
        return fmt.Errorf("provider %s is not registered", pType)
    }
    
    // Close provider before removing
    provider.Close()
    
    delete(pr.providers, pType)
    return nil
}

// GetAll returns all registered providers
func (pr *ProviderRegistry) GetAll() map[ProviderType]Provider {
    pr.mu.RLock()
    defer pr.mu.RUnlock()
    
    result := make(map[ProviderType]Provider, len(pr.providers))
    for k, v := range pr.providers {
        result[k] = v
    }
    
    return result
}

// GetEnabled returns all enabled providers
func (pr *ProviderRegistry) GetEnabled() map[ProviderType]Provider {
    pr.mu.RLock()
    defer pr.mu.RUnlock()
    
    result := make(map[ProviderType]Provider)
    for k, v := range pr.providers {
        if v.GetConfig().Enabled {
            result[k] = v
        }
    }
    
    return result
}

// HasProvider checks if a provider is registered
func (pr *ProviderRegistry) HasProvider(pType ProviderType) bool {
    pr.mu.RLock()
    defer pr.mu.RUnlock()
    
    _, exists := pr.providers[pType]
    return exists
}

// GetCount returns the number of registered providers
func (pr *ProviderRegistry) GetCount() int {
    pr.mu.RLock()
    defer pr.mu.RUnlock()
    
    return len(pr.providers)
}

// CloseAll closes all registered providers
func (pr *ProviderRegistry) CloseAll() error {
    pr.mu.Lock()
    defer pr.mu.Unlock()
    
    var lastErr error
    
    for pType, provider := range pr.providers {
        if err := provider.Close(); err != nil {
            lastErr = err
        }
    }
    
    pr.providers = make(map[ProviderType]Provider)
    return lastErr
}

// HealthCheckAll performs health checks on all registered providers
func (pr *ProviderRegistry) HealthCheckAll(ctx context.Context) map[ProviderType]error {
    pr.mu.RLock()
    defer pr.mu.RUnlock()
    
    results := make(map[ProviderType]error)
    
    for pType, provider := range pr.providers {
        if err := provider.HealthCheck(ctx); err != nil {
            results[pType] = err
        }
    }
    
    return results
}
```

## Dependencies

- **Step 01**: Core Interfaces (defines Provider interface and types)
- **Step 02**: Shared Models (ProviderConfig type)
- **Step 04**: Proxy Manager (ProxyManager dependency)

## Placeholder Implementations

The following placeholder functions will be implemented in later steps:

```go
// These will be implemented in provider-specific steps:
func NewGeminiProvider(cfg ProviderConfig, deps ProviderDependencies) (Provider, error)
func NewCopilotProvider(cfg ProviderConfig, deps ProviderDependencies) (Provider, error)
func NewQwenProvider(cfg ProviderConfig, deps ProviderDependencies) (Provider, error)
func NewAntigravityProvider(cfg ProviderConfig, deps ProviderDependencies) (Provider, error)

// This will be implemented in configuration step:
func GetProviderConfigFromEnv(providerType string) (ProviderConfig, error)
```

## Verification

After completing this step, verify:

1. Factory can be instantiated with dependencies
2. Providers can be registered and unregistered
3. Provider creation from configuration works
4. Multiple providers can be created from environment
5. Registry manages provider instances correctly

## Testing

```go
// Example test cases:

func TestProviderFactoryCreation(t *testing.T) {
    deps := ProviderDependencies{
        // Mock dependencies
    }
    
    factory := NewProviderFactory(deps)
    assert.NotNil(t, factory)
    assert.Equal(t, 4, factory.GetProviderCount())
}

func TestRegisterProvider(t *testing.T) {
    factory := NewProviderFactory(ProviderDependencies{})
    
    customCreator := func(cfg ProviderConfig, deps ProviderDependencies) (Provider, error) {
        return &mockProvider{}, nil
    }
    
    err := factory.RegisterProvider("custom", customCreator)
    assert.NoError(t, err)
    assert.True(t, factory.IsProviderSupported("custom"))
}

func TestCreateProvider(t *testing.T) {
    factory := NewProviderFactory(ProviderDependencies{})
    
    cfg := ProviderConfig{
        Type:    ProviderGemini,
        Enabled: true,
    }
    
    // This will fail until Gemini provider is implemented
    // provider, err := factory.CreateProvider(cfg)
    // assert.NoError(t, err)
    // assert.NotNil(t, provider)
}
```

## Integration Example

```go
// Example usage in main.go:

func main() {
    // Create shared dependencies
    proxyMgr := proxy.NewProxyManager()
    proxyMgr.InitializeFromConfig()
    
    authMgr := auth.NewAuthManager()
    
    // Create factory with dependencies
    deps := providers.ProviderDependencies{
        ProxyManager: proxyMgr,
        AuthManager:  authMgr,
    }
    
    factory := providers.NewProviderFactory(deps)
    
    // Create all enabled providers
    providerInstances, err := factory.CreateAllProvidersFromEnv()
    if err != nil {
        log.Fatalf("Failed to create providers: %v", err)
    }
    
    // Register providers in registry
    registry := providers.NewProviderRegistry()
    for pType, provider := range providerInstances {
        registry.Register(provider)
    }
    
    // Use providers...
}
```

## Next Steps

After completing this step, proceed to:
- **Step 06**: Configuration Management (load provider configs from environment)
- **Step 07**: Shared Middleware (CORS, auth, rate limiting)
- **Step 08**: Gemini Provider Migration (refactor existing code)
