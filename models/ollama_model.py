import requests
import json
from typing import Optional, Dict
from .base import BaseModelInference, GenerationConfig

class OllamaModel(BaseModelInference):
    """Ollama模型实现"""
    
    def __init__(self, model_name: str = "llama3.1",
                 generation_config: Optional[GenerationConfig] = None,
                 base_url: str = "http://localhost:11434"):
        super().__init__(model_name, generation_config)
        self.base_url = base_url.rstrip("/")
        
    def _generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """生成文本实现"""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens
            }
        }
        
        try:
            # 使用stream=True处理流式响应
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            # 收集所有响应文本
            full_response = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                    
                try:
                    # 解析每行JSON
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        full_response.append(chunk['response'])
                        
                    # 如果生成完成，退出循环
                    if chunk.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
            
            # 拼接所有响应文本
            return ''.join(full_response).strip()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")