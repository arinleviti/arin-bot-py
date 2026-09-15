response = groq_client.chat.completions.create(...)

# response (the WHOLE object) looks like this:
{
    "id": "chatcmpl-xyz789",
    "model": "openai/gpt-oss-120b",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {                        # ← this is response.choices[0].message
                "role": "assistant",             #   = response_message
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "notify_arin",
                            "arguments": '{"visitor_message": "wants to hire Arin", "visitor_contact": "jane@company.com"}'
                        }
                    }
                ]
            }
        }
    ],
    "usage": {"prompt_tokens": 412, "completion_tokens": 28, "total_tokens": 440}
}