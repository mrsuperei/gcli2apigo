package httputil

import (
	"context"
	"net/http"
	"time"

	"golang.org/x/net/http2"
	"golang.org/x/oauth2"
)

var (
	// OAuthHTTPClient is een dedicated HTTP client voor OAuth token refresh
	// met hogere connection limits voor betere performance
	OAuthHTTPClient = createOAuthHTTPClient()
)

// createOAuthHTTPClient maakt een geoptimaliseerde HTTP client voor OAuth operations
func createOAuthHTTPClient() *http.Client {
	transport := &http.Transport{
		// Hogere limits specifiek voor OAuth endpoint
		MaxIdleConns:        500,
		MaxIdleConnsPerHost: 200,
		IdleConnTimeout:     120 * time.Second,

		// Agressievere timeouts voor OAuth
		TLSHandshakeTimeout:   5 * time.Second,
		ResponseHeaderTimeout: 10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,

		// Connection pooling optimalisaties
		DisableKeepAlives:  false,
		DisableCompression: false,
		ForceAttemptHTTP2:  true,

		// Buffer optimalisaties
		WriteBufferSize: 32 * 1024,
		ReadBufferSize:  32 * 1024,
	}

	// Configure HTTP/2
	if err := http2.ConfigureTransport(transport); err != nil {
		panic("Failed to configure HTTP/2 for OAuth client: " + err.Error())
	}

	return &http.Client{
		Timeout:   30 * time.Second,
		Transport: transport,
	}
}

// GetOAuthContext returns a context with the OAuth HTTP client
func GetOAuthContext(ctx context.Context) context.Context {
	return context.WithValue(ctx, oauth2.HTTPClient, OAuthHTTPClient)
}
