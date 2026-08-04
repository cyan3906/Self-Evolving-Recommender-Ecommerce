import os
import time
from openai import OpenAI
import openai
import httpx
from langsmith import traceable
from langsmith.wrappers import wrap_openai # 使用监督模式
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from config.settings import Settings

settings = Settings()


class ChatGPT:
    def __init__(self,model_name: str = "gpt-4o-mini",is_langsmith: bool = False):
        self.api_key = settings.llm_api_key

        if is_langsmith:
            self.client = wrap_openai(OpenAI(
                api_key=self.api_key,
                base_url=settings.llm_base_url, # 中转站
            ))
            
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=settings.llm_base_url, # 中转站
            )

        self.model_name = model_name
        
    def generate(self,sys_prompt: str, prompt: str, temperature: float = 0.0):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                # {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=temperature
        )
        # print(response)
        return response.choices[0].message.content
    

    