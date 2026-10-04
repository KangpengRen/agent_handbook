import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMClient:

    def __init__(self, model: str):
        api_key = os.getenv("CLOSEAI_API_KEY")
        base_url = os.getenv("CLOSEAI_BASE_URL")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        # 维护消息列表
        self.messages = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_response(self, system_prompt: str = None) -> str:
        """发送请求并返回响应"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )

        return response.choices[0].message.content

    def reset(self):
        """重置历史对话"""
        self.messages = []