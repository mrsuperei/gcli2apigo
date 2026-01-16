# Step 02: Shared Models

## Context

This step defines common data structures used across all providers. These models represent the canonical OpenAI-compatible format that all providers must support, ensuring consistent API responses regardless of the underlying provider.

## Objectives

1. Define OpenAI-compatible request/response models
2. Define common utility types
3. Ensure models are provider-agnostic
4. Support all features needed by Gemini, Copilot, Qwen, and Antigravity

## Design Pattern

**Data Transfer Object (DTO) Pattern**: Models represent pure data structures with no business logic, enabling easy serialization/deserialization and transformation.

## Files to Create

### 1. `internal/models/common.go`

**Purpose**: Common types and utilities shared across all models

**Full Implementation**:

```go
package models

import (
    "encoding/json"
    "time"
)

// JSONTime is a custom time type for JSON marshaling
type JSONTime struct {
    time.Time
}

// MarshalJSON implements json.Marshaler interface
func (jt JSONTime) MarshalJSON() ([]byte, error) {
    return json.Marshal(jt.Time.Unix())
}

// UnmarshalJSON implements json.Unmarshaler interface
func (jt *JSONTime) UnmarshalJSON(data []byte) error {
    var timestamp int64
    if err := json.Unmarshal(data, &timestamp); err != nil {
        return err
    }
    jt.Time = time.Unix(timestamp, 0)
    return nil
}

// String returns string representation
func (jt JSONTime) String() string {
    return jt.Time.String()
}
```

### 2. `internal/models/request.go`

**Purpose**: Request models for chat completions

**Full Implementation**:

```go
package models

// OpenAIChatCompletionRequest represents an OpenAI chat completion request
// This is the canonical request format that all providers accept
type OpenAIChatCompletionRequest struct {
    Model            string                  `json:"model"`
    Messages         []OpenAIChatMessage     `json:"messages"`
    Tools            []Tool                  `json:"tools,omitempty"`
    ToolChoice       interface{}             `json:"tool_choice,omitempty"`
    Stream           bool                    `json:"stream,omitempty"`
    Temperature      *float64                `json:"temperature,omitempty"`
    TopP             *float64                `json:"top_p,omitempty"`
    MaxTokens        *int                    `json:"max_tokens,omitempty"`
    Stop             interface{}             `json:"stop,omitempty"`
    FrequencyPenalty *float64                `json:"frequency_penalty,omitempty"`
    PresencePenalty  *float64                `json:"presence_penalty,omitempty"`
    N                *int                    `json:"n,omitempty"`
    Seed             *int                    `json:"seed,omitempty"`
    ResponseFormat   map[string]interface{}  `json:"response_format,omitempty"`
    ReasoningEffort  string                  `json:"reasoning_effort,omitempty"`
    ThinkingTokens   *int                    `json:"thinking_tokens,omitempty"`
    ThinkingEnabled  *bool                   `json:"thinking_enabled,omitempty"`
    GenerationConfig *map[string]interface{} `json:"generation_config,omitempty"`
}

// OpenAIChatMessage represents a chat message
type OpenAIChatMessage struct {
    Role             string      `json:"role"`
    Content          interface{} `json:"content"`
    ReasoningContent string      `json:"reasoning_content,omitempty"`
    ToolCalls        []ToolCall  `json:"tool_calls,omitempty"`
    ToolCallID       string      `json:"tool_call_id,omitempty"`
    Name             string      `json:"name,omitempty"`
}

// ContentPart represents a structured content part
type ContentPart struct {
    Type     string    `json:"type"`
    Text     string    `json:"text,omitempty"`
    ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image URL
type ImageURL struct {
    URL string `json:"url"`
}

// Tool represents a tool definition
type Tool struct {
    Type     string   `json:"type"`
    Function Function `json:"function"`
}

// Function represents a function definition
type Function struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description,omitempty"`
    Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ToolCall represents a tool call
type ToolCall struct {
    ID       string   `json:"id"`
    Type     string   `json:"type"`
    Function Function `json:"function"`
}

// ValidateRequest validates an OpenAI chat completion request
func (req *OpenAIChatCompletionRequest) ValidateRequest() error {
    if req.Model == "" {
        return &ValidationError{Field: "model", Message: "model is required"}
    }
    
    if len(req.Messages) == 0 {
        return &ValidationError{Field: "messages", Message: "at least one message is required"}
    }
    
    // Validate messages
    for i, msg := range req.Messages {
        if msg.Role == "" {
            return &ValidationError{
                Field:   "messages",
                Message: "message role is required",
                Index:   &i,
            }
        }
    }
    
    return nil
}

// ValidationError represents a validation error
type ValidationError struct {
    Field   string  `json:"field"`
    Message string  `json:"message"`
    Index   *int    `json:"index,omitempty"`
}

// Error implements error interface
func (ve *ValidationError) Error() string {
    if ve.Index != nil {
        return "validation error in " + ve.Field + " at index " + string(rune(*ve.Index)) + ": " + ve.Message
    }
    return "validation error in " + ve.Field + ": " + ve.Message
}
```

### 3. `internal/models/response.go`

**Purpose**: Response models for chat completions

**Full Implementation**:

```go
package models

import (
    "time"
)

// OpenAIChatCompletionResponse represents an OpenAI chat completion response
// This is the canonical response format that all providers return
type OpenAIChatCompletionResponse struct {
    ID      string                       `json:"id"`
    Object  string                       `json:"object"`
    Created JSONTime                     `json:"created"`
    Model   string                       `json:"model"`
    Choices []OpenAIChatCompletionChoice `json:"choices"`
    Usage   *Usage                       `json:"usage,omitempty"`
}

// OpenAIChatCompletionChoice represents a completion choice
type OpenAIChatCompletionChoice struct {
    Index        int               `json:"index"`
    Message      OpenAIChatMessage `json:"message"`
    FinishReason *string           `json:"finish_reason,omitempty"`
}

// OpenAIChatCompletionStreamResponse represents a streaming chat completion response
type OpenAIChatCompletionStreamResponse struct {
    ID      string                              `json:"id"`
    Object  string                              `json:"object"`
    Created JSONTime                            `json:"created"`
    Model   string                              `json:"model"`
    Choices []OpenAIChatCompletionStreamChoice `json:"choices"`
    Usage   *Usage                              `json:"usage,omitempty"`
}

// OpenAIChatCompletionStreamChoice represents a streaming completion choice
type OpenAIChatCompletionStreamChoice struct {
    Index        int        `json:"index"`
    Delta        OpenAIDelta `json:"delta"`
    FinishReason *string     `json:"finish_reason,omitempty"`
}

// OpenAIDelta represents a delta in a streaming response
type OpenAIDelta struct {
    Role             string     `json:"role,omitempty"`
    Content          string     `json:"content,omitempty"`
    ReasoningContent string     `json:"reasoning_content,omitempty"`
    ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

// Usage represents token usage statistics
type Usage struct {
    PromptTokens            int                      `json:"prompt_tokens"`
    CompletionTokens        int                      `json:"completion_tokens"`
    TotalTokens             int                      `json:"total_tokens"`
    CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// CompletionTokensDetails provides breakdown of completion tokens
type CompletionTokensDetails struct {
    ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
    AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
    RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
}

// AddUsage adds usage from another Usage struct
func (u *Usage) AddUsage(other *Usage) {
    if other == nil {
        return
    }
    u.PromptTokens += other.PromptTokens
    u.CompletionTokens += other.CompletionTokens
    u.TotalTokens += other.TotalTokens
    
    if u.CompletionTokensDetails == nil {
        u.CompletionTokensDetails = &CompletionTokensDetails{}
    }
    if other.CompletionTokensDetails != nil {
        u.CompletionTokensDetails.ReasoningTokens += other.CompletionTokensDetails.ReasoningTokens
        u.CompletionTokensDetails.AcceptedPredictionTokens += other.CompletionTokensDetails.AcceptedPredictionTokens
        u.CompletionTokensDetails.RejectedPredictionTokens += other.CompletionTokensDetails.RejectedPredictionTokens
    }
}

// ModelsListResponse represents a list of models
type ModelsListResponse struct {
    Object string       `json:"object"`
    Data   []ModelInfo `json:"data"`
}

// ModelInfo represents information about a model
type ModelInfo struct {
    ID                       string           `json:"id"`
    Object                   string           `json:"object"`
    Created                  int64            `json:"created"`
    OwnedBy                  string           `json:"owned_by"`
    Type                     string           `json:"type"`
    Name                     string           `json:"name"`
    Version                  string           `json:"version"`
    DisplayName              string           `json:"display_name"`
    Description              string           `json:"description"`
    InputTokenLimit         int              `json:"input_token_limit"`
    OutputTokenLimit        int              `json:"output_token_limit"`
    SupportedGenerationMethods []string         `json:"supported_generation_methods"`
    Thinking                *ThinkingSupport `json:"thinking,omitempty"`
}

// ThinkingSupport represents thinking/reasoning capabilities
type ThinkingSupport struct {
    Min            int      `json:"min"`
    Max            int      `json:"max"`
    ZeroAllowed    bool     `json:"zero_allowed"`
    DynamicAllowed bool     `json:"dynamic_allowed"`
    Levels         []string `json:"levels,omitempty"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
    Error ErrorDetail `json:"error"`
}

// ErrorDetail represents error details
type ErrorDetail struct {
    Message string      `json:"message"`
    Type    string      `json:"type,omitempty"`
    Param    string      `json:"param,omitempty"`
    Code    interface{} `json:"code,omitempty"`
}

// Error implements error interface
func (ed *ErrorDetail) Error() string {
    if ed.Code != nil {
        return ed.Type + ": " + ed.Message + " (code: " + string(ed.Code.(string)) + ")"
    }
    return ed.Type + ": " + ed.Message
}
```

## Dependencies

- **Step 01**: Core Interfaces (defines basic types)

## Verification

After completing this step, verify:

1. All model files compile without errors
2. Models are properly tagged for JSON serialization
3. Validation logic works correctly
4. Models are provider-agnostic
5. All required OpenAI fields are supported

## Next Steps

After completing this step, proceed to:
- **Step 03**: Proxy Infrastructure (implement proxy interfaces)
- **Step 04**: Proxy Manager (manage multiple proxies)
- **Step 05**: Provider Factory (create provider instances)
