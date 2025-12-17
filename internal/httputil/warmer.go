package httputil

import (
	"context"
	"log"
	"net/http"
	"sync"
	"time"
)

// ConnectionWarmer warmt HTTP connections op bij startup
type ConnectionWarmer struct {
	client    *http.Client
	endpoints []string
	warmed    bool
	mu        sync.Mutex
}

// NewConnectionWarmer maakt een nieuwe connection warmer
func NewConnectionWarmer(client *http.Client, endpoints []string) *ConnectionWarmer {
	return &ConnectionWarmer{
		client:    client,
		endpoints: endpoints,
		warmed:    false,
	}
}

// WarmUp warmt connections op naar alle endpoints parallel
func (cw *ConnectionWarmer) WarmUp(ctx context.Context) {
	cw.mu.Lock()
	if cw.warmed {
		cw.mu.Unlock()
		return
	}
	cw.warmed = true
	cw.mu.Unlock()

	log.Printf("[INFO] Warming up HTTP connections to %d endpoints...", len(cw.endpoints))

	var wg sync.WaitGroup
	startTime := time.Now()

	// Warm up 3 connections per endpoint parallel
	for _, endpoint := range cw.endpoints {
		for i := 0; i < 3; i++ {
			wg.Add(1)
			go func(ep string, idx int) {
				defer wg.Done()

				// Maak een lightweight HEAD request om connection te openen
				req, err := http.NewRequestWithContext(ctx, "HEAD", ep, nil)
				if err != nil {
					log.Printf("[WARN] Failed to create warmup request for %s: %v", ep, err)
					return
				}

				resp, err := cw.client.Do(req)
				if err != nil {
					// Niet erg als warmup faalt, log alleen voor debug
					if log.Default().Writer() != nil {
						log.Printf("[DEBUG] Warmup request %d to %s failed (expected): %v", idx, ep, err)
					}
					return
				}
				resp.Body.Close()

				log.Printf("[DEBUG] Warmed connection %d to %s", idx, ep)
			}(endpoint, i)
		}
	}

	// Wacht op alle warmup requests met timeout
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		elapsed := time.Since(startTime)
		log.Printf("[INFO] Connection warmup completed in %v", elapsed)
	case <-time.After(5 * time.Second):
		log.Printf("[WARN] Connection warmup timed out after 5s")
	}
}

// WarmUpWithCredentials warmt connections op en probeert ook een echte auth flow
func (cw *ConnectionWarmer) WarmUpWithCredentials(ctx context.Context, token string, endpoint string) {
	log.Printf("[INFO] Warming up authenticated connection to %s", endpoint)

	req, err := http.NewRequestWithContext(ctx, "GET", endpoint, nil)
	if err != nil {
		return
	}

	req.Header.Set("Authorization", "Bearer "+token)

	resp, err := cw.client.Do(req)
	if err != nil {
		return
	}
	resp.Body.Close()

	log.Printf("[DEBUG] Warmed authenticated connection to %s", endpoint)
}

// GlobalWarmer is de globale connection warmer instance
var GlobalWarmer *ConnectionWarmer

// InitializeWarmer initialiseert de global warmer bij startup
func InitializeWarmer(endpoints []string) {
	GlobalWarmer = NewConnectionWarmer(SharedHTTPClient, endpoints)

	// Start warmup in background
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		GlobalWarmer.WarmUp(ctx)
	}()
}
