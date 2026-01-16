package models

// Usage represents token usage statistics (OpenAI format)
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

// OpenAI Models

type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"` // Used in definitions (JSON Schema)
	Arguments   interface{}            `json:"arguments,omitempty"`  // Used in tool calls (JSON String)
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type ToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type OpenAIChatMessage struct {
	Role             string      `json:"role"`
	Content          interface{} `json:"content"` // Can be string or []ContentPart
	ReasoningContent string      `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string      `json:"tool_call_id,omitempty"`
	Name             string      `json:"name,omitempty"`
}

type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
	URL string `json:"url"`
}

type OpenAIChatCompletionRequest struct {
	Model            string                  `json:"model"`
	Messages         []OpenAIChatMessage     `json:"messages"`
	Tools            []Tool                  `json:"tools,omitempty"`
	ToolChoice       interface{}             `json:"tool_choice,omitempty"`
	Stream           bool                    `json:"stream,omitempty"`
	Temperature      *float64                `json:"temperature,omitempty"`
	TopP             *float64                `json:"top_p,omitempty"`
	MaxTokens        *int                    `json:"max_tokens,omitempty"`
	Stop             interface{}             `json:"stop,omitempty"` // Can be string or []string
	FrequencyPenalty *float64                `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64                `json:"presence_penalty,omitempty"`
	N                *int                    `json:"n,omitempty"`
	Seed             *int                    `json:"seed,omitempty"`
	ResponseFormat   map[string]interface{}  `json:"response_format,omitempty"`   // Supports type, json_schema
	ReasoningEffort  string                  `json:"reasoning_effort,omitempty"`  // low, medium, high
	ThinkingTokens   *int                    `json:"thinking_tokens,omitempty"`   // Direct token count
	ThinkingEnabled  *bool                   `json:"thinking_enabled,omitempty"`  // Boolean flag (helixrun compat)
	GenerationConfig *map[string]interface{} `json:"generation_config,omitempty"` // trpc-agent-go compatibility
}

type OpenAIChatCompletionChoice struct {
	Index        int               `json:"index"`
	Message      OpenAIChatMessage `json:"message"`
	FinishReason *string           `json:"finish_reason,omitempty"`
}

type OpenAIChatCompletionResponse struct {
	ID      string                       `json:"id"`
	Object  string                       `json:"object"`
	Created int64                        `json:"created"`
	Model   string                       `json:"model"`
	Choices []OpenAIChatCompletionChoice `json:"choices"`
	Usage   *Usage                       `json:"usage,omitempty"`
}

type OpenAIDelta struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

type OpenAIChatCompletionStreamChoice struct {
	Index        int         `json:"index"`
	Delta        OpenAIDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason,omitempty"`
}

type OpenAIChatCompletionStreamResponse struct {
	ID      string                             `json:"id"`
	Object  string                             `json:"object"`
	Created int64                              `json:"created"`
	Model   string                             `json:"model"`
	Choices []OpenAIChatCompletionStreamChoice `json:"choices"`
	Usage   *Usage                             `json:"usage,omitempty"`
}

// Gemini Models

type GeminiPart struct {
	Text             string                  `json:"text,omitempty"`
	Thought          bool                    `json:"thought,omitempty"`
	InlineData       *GeminiInlineData       `json:"inlineData,omitempty"`
	FunctionCall     *GeminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *GeminiFunctionResponse `json:"functionResponse,omitempty"`
}

type GeminiFunctionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

type GeminiFunctionResponse struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}

type GeminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type GeminiContent struct {
	Role  string       `json:"role"`
	Parts []GeminiPart `json:"parts"`
}

type GeminiRequest struct {
	Contents         []GeminiContent          `json:"contents"`
	Tools            []map[string]interface{} `json:"tools,omitempty"`
	GenerationConfig map[string]interface{}   `json:"generationConfig,omitempty"`
	SafetySettings   []map[string]interface{} `json:"safetySettings,omitempty"`
	Model            string                   `json:"model,omitempty"`
}

type GeminiCandidate struct {
	Content      GeminiContent `json:"content"`
	FinishReason string        `json:"finishReason,omitempty"`
	Index        int           `json:"index"`
}

type GeminiResponse struct {
	Candidates []GeminiCandidate `json:"candidates"`
}

// ThinkingSupport represents thinking/reasoning capabilities for a model
type ThinkingSupport struct {
	Min            int      `json:"min"`
	Max            int      `json:"max"`
	ZeroAllowed    bool     `json:"zeroAllowed"`
	DynamicAllowed bool     `json:"dynamicAllowed"`
	Levels         []string `json:"levels,omitempty"`
}

// ModelInfo represents detailed information about a model
type ModelInfo struct {
	ID                         string           `json:"id"`
	Object                     string           `json:"object"`
	Created                    int64            `json:"created"`
	OwnedBy                    string           `json:"owned_by"`
	Type                       string           `json:"type"`
	Name                       string           `json:"name"`
	Version                    string           `json:"version"`
	DisplayName                string           `json:"displayName"`
	Description                string           `json:"description"`
	InputTokenLimit            int              `json:"inputTokenLimit"`
	OutputTokenLimit           int              `json:"outputTokenLimit"`
	SupportedGenerationMethods []string         `json:"supportedGenerationMethods"`
	Thinking                   *ThinkingSupport `json:"thinking,omitempty"`
}
