package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// ChunkAnalysis holds detailed analysis of a single chunk
type ChunkAnalysis struct {
	ChunkNumber         int
	RawJSON             string
	TextContent         string
	CharCount           int
	WordCount           int
	EndsWithPeriod      bool
	EndsWithQuestion    bool
	EndsWithExclamation bool
	HasPartialWord      bool
	IsCompleteSentence  bool
	Timestamp           time.Time
	TimeSinceLastMs     int64
}

func main() {
	// Check for API key
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY environment variable not set")
	}

	// Model to test
	model := "gemini-2.0-flash-exp" // Using a model that supports streaming

	// Simple prompt that will generate multiple sentences
	prompt := "Explain the concept of RESTful APIs in detail. Include at least 5 sentences."

	fmt.Printf("Testing Gemini API streaming behavior...\n")
	fmt.Printf("Model: %s\n", model)
	fmt.Printf("Prompt: %s\n\n", prompt)
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Build the request payload
	payload := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]string{
					{"text": prompt},
				},
			},
		},
		"generationConfig": map[string]interface{}{
			"temperature":     0.7,
			"maxOutputTokens": 1000,
		},
	}

	// Create request
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", model, apiKey)
	jsonPayload, _ := json.Marshal(payload)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(jsonPayload)))
	if err != nil {
		log.Fatalf("Failed to create request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Make the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Fatalf("Failed to make request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Fatalf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Process the streaming response
	reader := bufio.NewReader(resp.Body)
	chunkNumber := 0
	startTime := time.Now()
	lastChunkTime := startTime
	var analyses []ChunkAnalysis

	fmt.Printf("\n%s CHUNK ANALYSIS %s\n\n", strings.Repeat("=", 35), strings.Repeat("=", 35))

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				log.Printf("Error reading stream: %v", err)
			}
			break
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Parse SSE data line
		if chunk, found := strings.CutPrefix(line, "data: "); found {
			if chunk == "[DONE]" {
				fmt.Printf("\n%s END OF STREAM %s\n", strings.Repeat("=", 35), strings.Repeat("=", 35))
				break
			}

			chunkNumber++
			now := time.Now()
			timeSinceLast := now.Sub(lastChunkTime).Milliseconds()
			timeSinceStart := now.Sub(startTime).Milliseconds()
			lastChunkTime = now

			// Parse the JSON
			var obj map[string]interface{}
			if err := json.Unmarshal([]byte(chunk), &obj); err != nil {
				log.Printf("Failed to parse chunk #%d: %v", chunkNumber, err)
				continue
			}

			// Extract text content
			textContent := extractTextContent(obj)

			// Analyze the chunk
			analysis := ChunkAnalysis{
				ChunkNumber:         chunkNumber,
				RawJSON:             chunk,
				TextContent:         textContent,
				CharCount:           len(textContent),
				WordCount:           countWords(textContent),
				EndsWithPeriod:      strings.HasSuffix(strings.TrimSpace(textContent), "."),
				EndsWithQuestion:    strings.HasSuffix(strings.TrimSpace(textContent), "?"),
				EndsWithExclamation: strings.HasSuffix(strings.TrimSpace(textContent), "!"),
				Timestamp:           now,
				TimeSinceLastMs:     timeSinceLast,
			}

			// Check for partial word (ends without space and doesn't end with punctuation)
			trimmed := strings.TrimSpace(textContent)
			if len(trimmed) > 0 {
				analysis.HasPartialWord = !strings.HasSuffix(trimmed, " ") &&
					!strings.HasSuffix(trimmed, ".") &&
					!strings.HasSuffix(trimmed, "?") &&
					!strings.HasSuffix(trimmed, "!") &&
					!strings.HasSuffix(trimmed, ",") &&
					!strings.HasSuffix(trimmed, ";") &&
					!strings.HasSuffix(trimmed, ":") &&
					!strings.HasSuffix(trimmed, "\n")
			}

			// Check if it's a complete sentence
			analysis.IsCompleteSentence = analysis.EndsWithPeriod ||
				analysis.EndsWithQuestion ||
				analysis.EndsWithExclamation

			analyses = append(analyses, analysis)

			// Print detailed analysis for this chunk
			printChunkAnalysis(analysis, timeSinceStart)
		}
	}

	// Print summary statistics
	printSummary(analyses, startTime)
}

func extractTextContent(obj map[string]interface{}) string {
	// Navigate through the Gemini response structure
	if candidates, ok := obj["candidates"].([]interface{}); ok && len(candidates) > 0 {
		if cand, ok := candidates[0].(map[string]interface{}); ok {
			if content, ok := cand["content"].(map[string]interface{}); ok {
				if parts, ok := content["parts"].([]interface{}); ok && len(parts) > 0 {
					if part, ok := parts[0].(map[string]interface{}); ok {
						if text, ok := part["text"].(string); ok {
							return text
						}
					}
				}
			}
		}
	}
	return ""
}

func countWords(text string) int {
	if text == "" {
		return 0
	}
	words := strings.Fields(text)
	return len(words)
}

func printChunkAnalysis(analysis ChunkAnalysis, timeSinceStart int64) {
	fmt.Printf("\n📦 CHUNK #%d\n", analysis.ChunkNumber)
	fmt.Printf("%s\n", strings.Repeat("-", 80))
	fmt.Printf("Time since start: %dms | Time since last: %dms\n", timeSinceStart, analysis.TimeSinceLastMs)
	fmt.Printf("Character count: %d | Word count: %d\n", analysis.CharCount, analysis.WordCount)
	fmt.Printf("Ends with period: %v | Ends with question: %v | Ends with exclamation: %v\n",
		analysis.EndsWithPeriod, analysis.EndsWithQuestion, analysis.EndsWithExclamation)
	fmt.Printf("Has partial word: %v | Is complete sentence: %v\n",
		analysis.HasPartialWord, analysis.IsCompleteSentence)
	fmt.Printf("\nText content (first 200 chars):\n")
	fmt.Printf("  %s\n", truncateString(analysis.TextContent, 200))

	// Show if it ends with punctuation
	if analysis.CharCount > 0 {
		lastChar := string(analysis.TextContent[len(analysis.TextContent)-1])
		fmt.Printf("\nLast character: '%s' (ASCII: %d)\n", lastChar, analysis.TextContent[len(analysis.TextContent)-1])
	}

	// Show raw JSON (truncated)
	fmt.Printf("\nRaw JSON (first 300 chars):\n")
	fmt.Printf("  %s\n", truncateString(analysis.RawJSON, 300))
}

func printSummary(analyses []ChunkAnalysis, startTime time.Time) {
	totalTime := time.Since(startTime).Milliseconds()
	totalChunks := len(analyses)
	if totalChunks == 0 {
		return
	}

	totalChars := 0
	totalWords := 0
	sentenceChunks := 0
	partialWordChunks := 0
	var chunkSizes []int

	for _, a := range analyses {
		totalChars += a.CharCount
		totalWords += a.WordCount
		if a.IsCompleteSentence {
			sentenceChunks++
		}
		if a.HasPartialWord {
			partialWordChunks++
		}
		chunkSizes = append(chunkSizes, a.CharCount)
	}

	avgChunkSize := float64(totalChars) / float64(totalChunks)
	avgWordsPerChunk := float64(totalWords) / float64(totalChunks)
	sentencePercentage := float64(sentenceChunks) / float64(totalChunks) * 100
	partialWordPercentage := float64(partialWordChunks) / float64(totalChunks) * 100

	fmt.Printf("\n\n%s SUMMARY STATISTICS %s\n", strings.Repeat("=", 35), strings.Repeat("=", 35))
	fmt.Printf("\nTotal chunks received: %d\n", totalChunks)
	fmt.Printf("Total time: %dms\n", totalTime)
	fmt.Printf("Total characters: %d\n", totalChars)
	fmt.Printf("Total words: %d\n", totalWords)
	fmt.Printf("\nAverage chunk size: %.2f characters\n", avgChunkSize)
	fmt.Printf("Average words per chunk: %.2f\n", avgWordsPerChunk)
	fmt.Printf("Chunks ending with complete sentence: %d (%.1f%%)\n", sentenceChunks, sentencePercentage)
	fmt.Printf("Chunks with partial words: %d (%.1f%%)\n", partialWordChunks, partialWordPercentage)
	fmt.Printf("\nChunk size distribution:\n")
	fmt.Printf("  Min: %d chars\n", minInt(chunkSizes))
	fmt.Printf("  Max: %d chars\n", maxInt(chunkSizes))
	fmt.Printf("  Median: %d chars\n", medianInt(chunkSizes))

	// Determine chunking behavior
	fmt.Printf("\n%s CHUNKING BEHAVIOR ANALYSIS %s\n", strings.Repeat("=", 30), strings.Repeat("=", 30))

	if avgChunkSize > 100 {
		fmt.Printf("✗ LARGE CHUNKS DETECTED\n")
		fmt.Printf("  Average chunk size: %.2f characters (too large for word-by-word)\n", avgChunkSize)
		fmt.Printf("  This indicates sentence-level or paragraph-level chunking\n")
	} else if avgChunkSize > 20 {
		fmt.Printf("⚠ MEDIUM-SIZED CHUNKS\n")
		fmt.Printf("  Average chunk size: %.2f characters\n", avgChunkSize)
		fmt.Printf("  This could be phrase-level chunking\n")
	} else {
		fmt.Printf("✓ SMALL CHUNKS DETECTED\n")
		fmt.Printf("  Average chunk size: %.2f characters (consistent with word-by-word)\n", avgChunkSize)
	}

	if sentencePercentage > 50 {
		fmt.Printf("\n✗ SENTENCE-LEVEL CHUNKING\n")
		fmt.Printf("  %.1f%% of chunks end with sentence punctuation\n", sentencePercentage)
		fmt.Printf("  This confirms sentence-by-sentence streaming\n")
	} else if partialWordPercentage > 50 {
		fmt.Printf("\n✓ WORD-LEVEL CHUNKING\n")
		fmt.Printf("  %.1f%% of chunks contain partial words\n", partialWordPercentage)
		fmt.Printf("  This confirms word-by-word streaming\n")
	} else {
		fmt.Printf("\n⚠ MIXED CHUNKING\n")
		fmt.Printf("  Neither sentence-level nor word-level dominant\n")
		fmt.Printf("  This could be phrase-level or variable chunking\n")
	}

	fmt.Printf("\n%s RECOMMENDATION %s\n", strings.Repeat("=", 35), strings.Repeat("=", 35))
	if avgChunkSize > 100 || sentencePercentage > 50 {
		fmt.Printf("The Gemini API is sending LARGE, sentence-level chunks.\n")
		fmt.Printf("\nTo achieve word-by-word streaming, you would need to:\n")
		fmt.Printf("  1. Receive the large chunks from Gemini\n")
		fmt.Printf("  2. Split them into smaller pieces (e.g., word-by-word)\n")
		fmt.Printf("  3. Send the smaller pieces with small delays\n")
		fmt.Printf("\nThis is called 'artificial streaming' and simulates word-by-word\n")
		fmt.Printf("output even though the API itself sends larger chunks.\n")
	} else {
		fmt.Printf("The Gemini API is already sending small chunks.\n")
		fmt.Printf("No artificial chunking is needed - just forward as-is.\n")
	}
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func minInt(slice []int) int {
	if len(slice) == 0 {
		return 0
	}
	min := slice[0]
	for _, v := range slice {
		if v < min {
			min = v
		}
	}
	return min
}

func maxInt(slice []int) int {
	if len(slice) == 0 {
		return 0
	}
	max := slice[0]
	for _, v := range slice {
		if v > max {
			max = v
		}
	}
	return max
}

func medianInt(slice []int) int {
	if len(slice) == 0 {
		return 0
	}
	// Simple median calculation (not sorting for efficiency in this case)
	return slice[len(slice)/2]
}
