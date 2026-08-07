import os
from dotenv import load_dotenv
import requests

from config.settings import Settings

settings = Settings()

def generate_embedding(
    texts:list[str]
):
    
    API_KEY, BASE_URL = settings.embedding_api_key, settings.embedding_base_url
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    ans = []
    
    for text in texts:
        if type(text) == dict:
            _text = ""
            for key, value in text.items():
                _text += f"{key}: {value}\n"
            text = _text
        
        resp = requests.post(f"{BASE_URL}/v1/embeddings", headers=headers, json={
            "model": settings.embedding_model,
            "input": text,
            "encoding_format": "float"
        }).json()
        ans.append(resp["data"][0]["embedding"])
    
    return ans




if __name__ == "__main__":
    
    print(generate_response("你好", "你好"))
