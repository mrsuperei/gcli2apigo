# Test Examples for Enhanced gcli2apigo

## 1. Reasoning/Thinking Support Test

### Test with Low Reasoning Effort (1024 tokens)
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {"role": "user", "content": "Solve this complex math problem: What is the integral of x^2 * sin(x) dx?"}
    ],
    "response_format": {
      "reasoning_effort": "low"
    },
    "stream": false
  }'
```

### Test with Medium Reasoning Effort (4096 tokens)
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {"role": "user", "content": "Create a detailed business plan for a tech startup"}
    ],
    "response_format": {
      "reasoning_effort": "medium"
    },
    "stream": false
  }'
```

### Test with High Reasoning Effort (8192 tokens)
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {"role": "user", "content": "Analyze the implications of quantum computing on cryptography"}
    ],
    "response_format": {
      "reasoning_effort": "high"
    },
    "stream": false
  }'
```

### Expected Response Format
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gemini-2.5-pro",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The actual response content...",
        "reasoning_content": "Internal reasoning process visible here..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

## 2. Structured Output with JSON Schema

### Test JSON Schema Validation
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {"role": "user", "content": "Generate a user profile with name, age, and email"}
    ],
    "response_format": {
      "type": "json_object",
      "json_schema": {
        "name": "user_profile",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"}
          },
          "required": ["name", "age", "email"]
        }
      }
    },
    "stream": false
  }'
```

### Expected Response
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gemini-2.5-pro",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"name\":\"John Doe\",\"age\":30,\"email\":\"john@example.com\"}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

## 3. True Live Streaming Test

### Test Immediate Streaming (No Buffering)
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "Write a long story about a space adventure"}
    ],
    "stream": true
  }' \
  --no-buffer
```

### Expected Streaming Response (SSE Format)
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-flash","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-flash","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-flash","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-flash","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}

... (continues immediately as content is generated)

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## 4. Streaming with Reasoning Content

### Test Live Streaming with Reasoning
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {"role": "user", "content": "Explain quantum entanglement"}
    ],
    "response_format": {
      "reasoning_effort": "high"
    },
    "stream": true
  }' \
  --no-buffer
```

### Expected Streaming Response with Reasoning
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-pro","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-pro","choices":[{"index":0,"delta":{"reasoning_content":"Let me think about this..."},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-pro","choices":[{"index":0,"delta":{"content":"Quantum entanglement is"},"finish_reason":null}]}

... (continues with mix of reasoning_content and content)

data: [DONE]
```

## 5. Tool Calls with Live Streaming

### Test Tool Calls in Stream Mode
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather information for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "stream": true
  }' \
  --no-buffer
```

### Expected Response with Tool Calls
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-flash","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

... (text chunks streamed first if any)

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_xxx","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Tokyo\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gemini-2.5-flash","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

## 6. Combined Test: Reasoning + Structured Output + Streaming

### Ultimate Test
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {"role": "user", "content": "Analyze this data and provide insights in JSON format"}
    ],
    "response_format": {
      "type": "json_object",
      "reasoning_effort": "high",
      "json_schema": {
        "name": "analysis",
        "schema": {
          "type": "object",
          "properties": {
            "insights": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"}
          }
        }
      }
    },
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 2000
  }' \
  --no-buffer
```

## 7. Python Test Script

```python
import requests
import json

BASE_URL = "http://localhost:7860"
API_KEY = "YOUR_PASSWORD"

def test_live_streaming():
    """Test true live streaming with immediate chunk forwarding"""
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "gemini-2.5-flash",
        "messages": [
            {"role": "user", "content": "Count from 1 to 20 slowly"}
        ],
        "stream": True
    }
    
    print("Testing TRUE LIVE STREAMING (zero buffering):")
    print("-" * 50)
    
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                content = line[6:]
                if content == '[DONE]':
                    print("\nStream completed!")
                    break
                try:
                    chunk = json.loads(content)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            print(delta['content'], end='', flush=True)
                except json.JSONDecodeError:
                    pass

def test_reasoning():
    """Test reasoning/thinking support"""
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "gemini-2.5-pro",
        "messages": [
            {"role": "user", "content": "Solve: What is 123 * 456?"}
        ],
        "response_format": {
            "reasoning_effort": "medium"
        },
        "stream": False
    }
    
    print("\nTesting REASONING SUPPORT:")
    print("-" * 50)
    
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    
    if 'choices' in result and len(result['choices']) > 0:
        message = result['choices'][0]['message']
        print(f"Content: {message.get('content', 'N/A')}")
        print(f"Reasoning: {message.get('reasoning_content', 'N/A')}")

def test_structured_output():
    """Test structured JSON output"""
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "gemini-2.5-pro",
        "messages": [
            {"role": "user", "content": "Create a person profile"}
        ],
        "response_format": {
            "type": "json_object",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        },
        "stream": False
    }
    
    print("\nTesting STRUCTURED OUTPUT:")
    print("-" * 50)
    
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    
    if 'choices' in result and len(result['choices']) > 0:
        content = result['choices'][0]['message']['content']
        parsed = json.loads(content)
        print(f"Structured output: {json.dumps(parsed, indent=2)}")

if __name__ == "__main__":
    test_live_streaming()
    test_reasoning()
    test_structured_output()
```

## 8. Performance Verification

### Verify Zero Buffering
```bash
# This should show chunks appearing immediately, not in batches
time curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "Write 1000 words about AI"}
    ],
    "stream": true
  }' \
  --no-buffer | grep "data:" | head -20
```

You should see data appearing line by line immediately, not in groups.

## 9. Error Handling Test

### Test Invalid Reasoning Effort
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PASSWORD" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [
      {"role": "user", "content": "Test"}
    ],
    "response_format": {
      "reasoning_effort": "invalid"
    },
    "stream": false
  }'
```

Should still work - invalid values default to Gemini's default (-1).

## Success Criteria

✅ **Reasoning Support**: `reasoning_content` appears in responses when `reasoning_effort` is set
✅ **Live Streaming**: Chunks appear immediately, not in batches (verify with timestamps)
✅ **Structured Output**: JSON responses conform to provided schema
✅ **Tool Calls**: Tool calls are properly buffered and sent before finish_reason
✅ **Performance**: No noticeable buffering delay in streaming responses
✅ **Compatibility**: All existing OpenAI API features still work correctly
