# 1. 请求头
```python
headers = {  
    "Authorization": f"Bearer {API_KEY}",  
    "Content-Type": "application/json"  
}
```
# 2. 请求体
```python
payload = {  
    "model": "deepseek-chat",  
    "messages": messages,  
    "tools": tools,  
    "stream": False,  
    "temperature": 0.7,  
    "max_tokens": 1024  
}
```
**核心字段解析：**  
* **`model` (必填)**：指定要调用的模型。如 "deepseek-chat"（V3）或 "deepseek-reasoner"（R1）。。  
*   **`messages` (必填)**：对话历史数组。这是整个请求的灵魂，包含了上下文和用户的指令。  
		其中`messages`为消息对象，结构为：
```json
[
	{"role": "system", "content": SYSTEM_PROMPT},
	{"role": "role", "content": TASK},
	...
]
```
前两条消息必须为`system`和`role`，以告诉大模型自己的角色和当前任务。
其中`tools`为工具对象，结构为：
*   **`tools` (可选)**：提供给模型使用的工具列表（Function Calling）。如果不传，模型只能靠自己的知识回答。  
    * `type`：固定为 "function"。   
		* `function.name`：函数名（必须与本地函数名一致）。  
		* `function.description`：极为重要！ 模型完全靠这段文字来理解这个工具的作用。  
		* `function.parameters`：标准的 JSON Schema，告诉模型需要传什么参数。 *   **`temperature` (可选)**：控制随机性，0-2 之间。值越低（如 0.1）回答越严谨固定；值越高（如 1.0）越有创造力。  
```json
[
	{  
	    "type": "function",  
	    "function": {  
	        "name": 工具名,  
	        "description": 工具描述,  
	        "parameters": "parameters": {  
				"type": "object",  
				"properties": {参数名: {"type": 参数类型, "description": 参数描述}},  
				"required": []  
			},
    },
    ...  
]
```
*   **`max_tokens` (可选)**：限制模型生成的最大长度。  
*   **`stream` (可选)**：布尔值。true 表示流式输出（打字机效果），false 表示等待完整结果一次性返回。  

# 3. 返回结果
一个典型的普通文本回复，返回的 JSON 结构如下：  
```json  
{  
  "id": "88beaa90-17fd-45fc-939c-2ef75e2d913c",  
  "object": "chat.completion",  
  "created": 1793084842,  
  "model": "deepseek-chat",  
  "choices": [    
	  {      
		  "index": 0,      
		  "message": {        
			  "role": "assistant",        
			  "content": "Hello! How can I help you today?"      
		  },      
		  "logprobs": null,      
		  "finish_reason": "stop"    
	  }
  ],  
  "usage": {    
	  "prompt_tokens": 7,    
	  "completion_tokens": 11,    
	  "total_tokens": 18  
  },  
  "system_fingerprint": "e72d3c7f663d3e1383b00b3884cb4501"
}
```  
  
**核心字段解析：**  
*   **`choices`**：模型生成的回复列表（通常只有一个，即 `index: 0`）。  
    *   **`message`**：包含模型返回的完整消息。`role` 为 `assistant`，`content` 是核心的文本内容。  
    *   **`finish_reason`**：**判断模型是否结束回复的关键**。`stop` 表示正常结束；`length` 表示因 `max_tokens` 限制被截断；`tool_calls` 表示模型请求调用工具。  
*   **`usage`**：本次请求的 Token 消耗统计，用于计费和上下文管理。`prompt_tokens` 是输入消耗，`completion_tokens` 是生成消耗。

当模型认为需要调用工具时，`finish_reason` 会变为 `"tool_calls"`，并且 `message` 对象中会多出一个 **`tool_calls`** 数组，而 `content` 通常为 `null`。
```json
{  
  "choices": [{    
	  "index": 0,    
	  "message": {      
		  "role": "assistant",      
		  "content": null,      
		  "tool_calls": [{          
			  "id": "call_abc123",          
			  "type": "function",          
			  "function": {            
				  "name": "list_files",            
				  "arguments": "{\"dir\": \".\"}"          
			  }        
		  }]    
	  },    
	  "finish_reason": "tool_calls"  
  }]
}
```
**关键点：**  
*   **`id`**：这次工具调用的唯一标识。你在回传工具执行结果时，必须用 `tool_call_id` 字段原样带上它，模型才能知道结果对应哪次调用。  
*   **`function.arguments`**：这是一个 **JSON 字符串**，而非直接的 JSON 对象。你需要在代码里用 `json.loads()` 解析它，才能得到真正的参数字典。
```python
messages.append({  
    "role": "tool",  
    "tool_call_id": tool_call["id"],  
    "content": str(output)  
})
```