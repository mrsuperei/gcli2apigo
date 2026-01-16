# Step 07: Shared Middleware

## Context

This step implements shared HTTP middleware that applies to all provider routes. Middleware handles cross-cutting concerns like CORS, authentication, logging, and rate limiting.

## Objectives

1. Implement CORS middleware for cross-origin requests
2. Implement authentication middleware for API security
3. Implement logging middleware for request tracking
4. Implement rate limiting middleware for API protection
5. Implement error handling middleware for consistent error responses

## Design Pattern

**Decorator Pattern**: Middleware wraps HTTP handlers, adding behavior before and after handler execution.

## Files to Create

### 1. `internal/routes/middleware.go`

**Purpose**: Shared HTTP middleware for all routes

**Full Implementation**:

```go
package routes

import (
    "encoding/json"
    "log"
    "net/http"
    "strings"
    "time"
    
    "gcli2apigo/internal/auth"
    "gcli2apigo/internal/models"
)

// Middleware is a function that wraps an HTTP handler
type Middleware func(http.Handler) http.Handler

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
    http.ResponseWriter
    status int
    written bool
}

// WriteHeader captures the status code
func (rw *responseWriter) WriteHeader(code int) {
    rw.status = code
    rw.ResponseWriter.WriteHeader(code)
}

// Write marks that response has been written
func (rw *responseWriter) Write(b []byte) (int, error) {
    rw.written = true
    return rw.ResponseWriter.Write(b)
}

// corsMiddleware adds CORS headers to responses
func corsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, PATCH, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "*")
        w.Header().Set("Access-Control-Allow-Credentials", "true")
        w.Header().Set("Access-Control-Max-Age", "86400")
        
        // Handle preflight requests
        if r.Method == http.MethodOptions {
            w.WriteHeader(http.StatusOK)
            return
        }
        
        next.ServeHTTP(w, r)
    })
}

// loggingMiddleware logs HTTP requests and responses
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Wrap response writer to capture status
        wrapped := &responseWriter{ResponseWriter: w}
        
        // Log request
        log.Printf("[REQUEST] %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)
        
        // Call next handler
        next.ServeHTTP(wrapped, r)
        
        // Log response
        duration := time.Since(start)
        log.Printf("[RESPONSE] %s %s %d %v", r.Method, r.URL.Path, wrapped.status, duration)
    })
}

// authMiddleware validates authentication for API requests
func authMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Skip auth for health check
        if r.URL.Path == "/health" {
            next.ServeHTTP(w, r)
            return
        }
        
        // Skip auth for dashboard login
        if strings.HasPrefix(r.URL.Path, "/dashboard/login") {
            next.ServeHTTP(w, r)
            return
        }
        
        // Validate authentication
        if _, err := auth.AuthenticateUser(r); err != nil {
            sendErrorResponse(w, models.ErrorDetail{
                Type:    "authentication_error",
                Message: "Invalid authentication credentials",
                Code:    401,
            }, http.StatusUnauthorized)
            return
        }
        
        next.ServeHTTP(w, r)
    })
}

// rateLimitMiddleware applies rate limiting to requests
func rateLimitMiddleware(next http.Handler) http.Handler {
    // This will be implemented with rate limiting infrastructure
    // For now, just pass through
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // TODO: Implement rate limiting
        next.ServeHTTP(w, r)
    })
}

// errorHandlingMiddleware handles errors and returns consistent error responses
func errorHandlingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if r := recover(); r != nil {
                log.Printf("[PANIC] Recovered from panic: %v", r)
                sendErrorResponse(w, models.ErrorDetail{
                    Type:    "internal_error",
                    Message: "Internal server error",
                    Code:    500,
                }, http.StatusInternalServerError)
            }
        }()
        
        next.ServeHTTP(w, r)
    })
}

// contentTypeMiddleware ensures proper content-type headers
func contentTypeMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Set default content type for API responses
        if strings.HasPrefix(r.URL.Path, "/v1/") || 
           strings.HasPrefix(r.URL.Path, "/geminicli/") ||
           strings.HasPrefix(r.URL.Path, "/copilotcli/") ||
           strings.HasPrefix(r.URL.Path, "/qwencli/") ||
           strings.HasPrefix(r.URL.Path, "/antigravitycli/") {
            w.Header().Set("Content-Type", "application/json; charset=utf-8")
        }
        
        next.ServeHTTP(w, r)
    })
}

// requestIDMiddleware adds unique request ID to context
func requestIDMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        requestID := r.Header.Get("X-Request-ID")
        if requestID == "" {
            // Generate simple request ID
            requestID = generateRequestID()
        }
        
        w.Header().Set("X-Request-ID", requestID)
        next.ServeHTTP(w, r)
    })
}

// sendErrorResponse sends a consistent error response
func sendErrorResponse(w http.ResponseWriter, err models.ErrorDetail, statusCode int) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(statusCode)
    
    response := models.ErrorResponse{
        Error: err,
    }
    
    if err := json.NewEncoder(w).Encode(response); err != nil {
        log.Printf("[ERROR] Failed to encode error response: %v", err)
    }
}

// generateRequestID generates a simple request ID
func generateRequestID() string {
    return time.Now().Format("20060102-150405") + "-" + randomString(8)
}

// randomString generates a random alphanumeric string
func randomString(length int) string {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    b := make([]byte, length)
    for i := range b {
        b[i] = charset[time.Now().Nanosecond()%len(charset)]
    }
    return string(b)
}

// applyMiddleware applies middleware chain to a handler
func applyMiddleware(handler http.Handler, middlewares ...Middleware) http.Handler {
    for i := len(middlewares) - 1; i >= 0; i-- {
        handler = middlewares[i](handler)
    }
    return handler
}
```

### 2. `internal/routes/middleware_test.go`

**Purpose**: Unit tests for middleware

**Full Implementation**:

```go
package routes

import (
    "net/http"
    "net/http/httptest"
    "testing"
    
    "gcli2apigo/internal/models"
)

func TestCORSMiddleware(t *testing.T) {
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })
    
    wrapped := corsMiddleware(handler)
    
    req := httptest.NewRequest("GET", "/test", nil)
    req.Header.Set("Origin", "https://example.com")
    
    rr := httptest.NewRecorder()
    wrapped.ServeHTTP(rr, req)
    
    // Check CORS headers
    assert.Equal(t, "*", rr.Header().Get("Access-Control-Allow-Origin"))
    assert.Equal(t, "GET, POST, PUT, DELETE, PATCH, OPTIONS", rr.Header().Get("Access-Control-Allow-Methods"))
}

func TestCORSPreflight(t *testing.T) {
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
    })
    
    wrapped := corsMiddleware(handler)
    
    req := httptest.NewRequest("OPTIONS", "/test", nil)
    req.Header.Set("Origin", "https://example.com")
    
    rr := httptest.NewRecorder()
    wrapped.ServeHTTP(rr, req)
    
    // Preflight should return 200 OK
    assert.Equal(t, http.StatusOK, rr.Code)
}

func TestLoggingMiddleware(t *testing.T) {
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })
    
    wrapped := loggingMiddleware(handler)
    
    req := httptest.NewRequest("GET", "/test", nil)
    rr := httptest.NewRecorder()
    wrapped.ServeHTTP(rr, req)
    
    // Should still return OK
    assert.Equal(t, http.StatusOK, rr.Code)
}

func TestAuthMiddleware(t *testing.T) {
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })
    
    wrapped := authMiddleware(handler)
    
    // Test without auth
    req := httptest.NewRequest("GET", "/test", nil)
    rr := httptest.NewRecorder()
    wrapped.ServeHTTP(rr, req)
    
    // Should return 401
    assert.Equal(t, http.StatusUnauthorized, rr.Code)
    
    // Test health check bypass
    req2 := httptest.NewRequest("GET", "/health", nil)
    rr2 := httptest.NewRecorder()
    wrapped.ServeHTTP(rr2, req2)
    
    // Health check should pass without auth
    assert.Equal(t, http.StatusOK, rr2.Code)
}

func TestApplyMiddleware(t *testing.T) {
    handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
    })
    
    middlewares := []Middleware{
        corsMiddleware,
        loggingMiddleware,
        contentTypeMiddleware,
    }
    
    wrapped := applyMiddleware(handler, middlewares...)
    
    req := httptest.NewRequest("GET", "/test", nil)
    rr := httptest.NewRecorder()
    wrapped.ServeHTTP(rr, req)
    
    // Should return OK with all middleware applied
    assert.Equal(t, http.StatusOK, rr.Code)
    assert.Equal(t, "application/json; charset=utf-8", rr.Header().Get("Content-Type"))
}
```

## Dependencies

- **Step 01**: Core Interfaces (no direct dependency)
- **Step 02**: Shared Models (ErrorResponse type)
- **Step 06**: Configuration Management (no direct dependency)

## Middleware Chain Order

Recommended order for middleware application:

1. **errorHandlingMiddleware** - Outermost, catches panics
2. **requestIDMiddleware** - Adds request tracking
3. **loggingMiddleware** - Logs all requests
4. **corsMiddleware** - Handles CORS headers
5. **contentTypeMiddleware** - Sets content type
6. **authMiddleware** - Validates authentication
7. **rateLimitMiddleware** - Applies rate limiting

## Verification

After completing this step, verify:

1. CORS headers are set correctly
2. Authentication works for protected routes
3. Health check bypasses auth
4. Logging captures request/response details
5. Error responses are consistent

## Next Steps

After completing this step, proceed to:
- **Step 08**: Provider Router (route requests to different providers)
- **Step 09**: Gemini Provider Migration (refactor existing code)
- **Step 10**: Copilot Provider Implementation
