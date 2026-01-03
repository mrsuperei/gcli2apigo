# Gemini API Streaming Chunk Analysis

This test program analyzes how the Gemini API chunks its streaming responses to determine if it sends:
- Sentence-by-sentence chunks
- Word-by-word chunks
- Phrase-level chunks
- Or some other pattern

## Purpose

The goal is to understand the actual chunking behavior of the Gemini API so we can decide whether to implement artificial word-by-word streaming.

## How to Run

### Prerequisites

1. Get a Gemini API key from https://makersuite.google.com/app/apikey
2. Set the environment variable:
   ```bash
   # Windows CMD
   set GEMINI_API_KEY=your_api_key_here

   # Windows PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"

   # Linux/Mac
   export GEMINI_API_KEY=your_api_key_here
   ```

### Running the Test

```bash
# Navigate to test directory
cd test_analysis

# Run the test
go run test_streaming_chunks.go
```

Or use the provided batch script (Windows):
```bash
run_test.bat
```

## What the Test Does

1. Makes a streaming request to Gemini API
2. Captures each chunk as it arrives
3. Analyzes each chunk for:
   - Character count
   - Word count
   - Whether it ends with sentence punctuation (., ?, !)
   - Whether it contains partial words
   - Time between chunks

4. Provides summary statistics:
   - Average chunk size
   - Chunk size distribution (min, max, median)
   - Percentage of chunks ending with complete sentences
   - Percentage of chunks with partial words

5. Makes a recommendation on whether artificial word-by-word chunking is needed

## Interpreting Results

### Large Chunks (>100 chars average)
- Indicates sentence-level or paragraph-level chunking
- Recommendation: Implement artificial word-by-word streaming

### Medium Chunks (20-100 chars average)
- Could be phrase-level chunking
- Recommendation: Depends on your use case

### Small Chunks (<20 chars average)
- Indicates word-by-word chunking
- Recommendation: No artificial chunking needed

### High Sentence Percentage (>50%)
- Confirms sentence-by-sentence streaming
- Recommendation: Implement artificial word-by-word streaming

### High Partial Word Percentage (>50%)
- Confirms word-by-word streaming
- Recommendation: No artificial chunking needed

## Example Output

```
Testing Gemini API streaming behavior...
Model: gemini-2.0-flash-exp
Prompt: Explain the concept of RESTful APIs in detail. Include at least 5 sentences.

================================================================================

=================================== CHUNK ANALYSIS ====================================

📦 CHUNK #1
--------------------------------------------------------------------------------
Time since start: 150ms | Time since last: 0ms
Character count: 293 | Word count: 45
Ends with period: true | Ends with question: false | Ends with exclamation: false
Has partial word: false | Is complete sentence: true

Text content (first 200 chars):
  **Framing the Inquiry** I'm starting by unpacking the user's core request: "the best way to design a Go REST API." The term "best" immediately flags this as subjective, contingent on various factors. Currently, I'm analyzing the implicit assumption

Last character: 'n' (ASCII: 110)

...

================================ SUMMARY STATISTICS ==================================

Total chunks received: 15
Total time: 2500ms
Total characters: 4234
Total words: 678

Average chunk size: 282.27 characters
Average words per chunk: 45.20
Chunks ending with complete sentence: 12 (80.0%)
Chunks with partial words: 0 (0.0%)

================================ CHUNKING BEHAVIOR ANALYSIS =================================

✗ LARGE CHUNKS DETECTED
  Average chunk size: 282.27 characters (too large for word-by-word)
  This indicates sentence-level or paragraph-level chunking

✗ SENTENCE-LEVEL CHUNKING
  80.0% of chunks end with sentence punctuation
  This confirms sentence-by-sentence streaming

================================ RECOMMENDATION =================================

The Gemini API is sending LARGE, sentence-level chunks.

To achieve word-by-word streaming, you would need to:
  1. Receive the large chunks from Gemini
  2. Split them into smaller pieces (e.g., word-by-word)
  3. Send the smaller pieces with small delays

This is called 'artificial streaming' and simulates word-by-word
output even though the API itself sends larger chunks.

================================================================================
```

## Notes

- The test uses the `gemini-2.0-flash-exp` model by default, but you can modify it in the code
- The prompt is designed to generate multiple sentences for better analysis
- Results may vary depending on the model and prompt used
- Run the test multiple times to get consistent results

## Next Steps

After running the test and reviewing the results:

1. If chunks are large/sentence-level: Consider implementing artificial word-by-word streaming
2. If chunks are already small/word-level: No changes needed
3. If results are mixed: Consider your use case and user experience goals
