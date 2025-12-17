package client

import (
	"log"
	"time"
)

// PerformanceTimer tracks timing of different request phases
type PerformanceTimer struct {
	startTime            time.Time
	credentialSelectTime time.Duration
	tokenRefreshTime     time.Duration
	onboardingTime       time.Duration
	apiRequestTime       time.Duration
	totalTime            time.Duration
	projectID            string
	model                string
}

// NewPerformanceTimer creates a new performance timer
func NewPerformanceTimer(model string) *PerformanceTimer {
	return &PerformanceTimer{
		startTime: time.Now(),
		model:     model,
	}
}

// MarkCredentialSelect marks the end of credential selection
func (pt *PerformanceTimer) MarkCredentialSelect(projectID string) {
	pt.credentialSelectTime = time.Since(pt.startTime)
	pt.projectID = projectID
	log.Printf("[PERF] Credential selection took: %v (project: %s)", pt.credentialSelectTime, projectID)
}

// MarkTokenRefreshStart marks the start of token refresh
func (pt *PerformanceTimer) MarkTokenRefreshStart() time.Time {
	return time.Now()
}

// MarkTokenRefreshEnd marks the end of token refresh
func (pt *PerformanceTimer) MarkTokenRefreshEnd(startTime time.Time) {
	pt.tokenRefreshTime = time.Since(startTime)
	log.Printf("[PERF] Token refresh took: %v (project: %s)", pt.tokenRefreshTime, pt.projectID)
}

// MarkOnboardingStart marks the start of onboarding
func (pt *PerformanceTimer) MarkOnboardingStart() time.Time {
	return time.Now()
}

// MarkOnboardingEnd marks the end of onboarding
func (pt *PerformanceTimer) MarkOnboardingEnd(startTime time.Time) {
	pt.onboardingTime = time.Since(startTime)
	log.Printf("[PERF] Onboarding took: %v (project: %s)", pt.onboardingTime, pt.projectID)
}

// MarkAPIRequestStart marks the start of API request
func (pt *PerformanceTimer) MarkAPIRequestStart() time.Time {
	return time.Now()
}

// MarkAPIRequestEnd marks the end of API request
func (pt *PerformanceTimer) MarkAPIRequestEnd(startTime time.Time) {
	pt.apiRequestTime = time.Since(startTime)
	log.Printf("[PERF] 🚨 GEMINI API REQUEST took: %v (project: %s, model: %s)",
		pt.apiRequestTime, pt.projectID, pt.model)
}

// LogSummary logs a complete performance summary
func (pt *PerformanceTimer) LogSummary() {
	pt.totalTime = time.Since(pt.startTime)

	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Printf("[PERF SUMMARY] Total request time: %v", pt.totalTime)
	log.Printf("  📊 Breakdown:")
	log.Printf("    • Credential select:  %6v (%5.1f%%)", pt.credentialSelectTime, percentage(pt.credentialSelectTime, pt.totalTime))
	log.Printf("    • Token refresh:      %6v (%5.1f%%)", pt.tokenRefreshTime, percentage(pt.tokenRefreshTime, pt.totalTime))
	log.Printf("    • Onboarding:         %6v (%5.1f%%)", pt.onboardingTime, percentage(pt.onboardingTime, pt.totalTime))
	log.Printf("    • 🎯 Gemini API call:  %6v (%5.1f%%)", pt.apiRequestTime, percentage(pt.apiRequestTime, pt.totalTime))
	log.Printf("  Project: %s | Model: %s", pt.projectID, pt.model)
	log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// Warning if API call takes too long
	if pt.apiRequestTime > 5*time.Second {
		log.Printf("⚠️  WARNING: Gemini API is slow! (%v) - This is NOT our proxy!", pt.apiRequestTime)
	}

	// Warning if our overhead is high
	ourOverhead := pt.credentialSelectTime + pt.tokenRefreshTime + pt.onboardingTime
	if ourOverhead > 1*time.Second {
		log.Printf("⚠️  WARNING: Our proxy overhead is high! (%v)", ourOverhead)
	}
}

// percentage calculates percentage of part relative to total
func percentage(part, total time.Duration) float64 {
	if total == 0 {
		return 0
	}
	return float64(part) / float64(total) * 100
}
