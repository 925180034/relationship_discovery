from vllm import LLM, SamplingParams
from typing import Optional, Dict
from .base import BaseModelInference, GenerationConfig

class VLLMModel(BaseModelInference):
    """vLLM模型实现"""
    
    def __init__(self, model_name: str,
                 generation_config: Optional[GenerationConfig] = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9):
        super().__init__(model_name, generation_config)
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )
        
    def _generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """生成文本实现"""
        # 转换为vLLM采样参数
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences,
            top_k=config.top_k,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        
        # 生成
        outputs = self.llm.generate(prompt, sampling_params)
        
        # 提取响应
        if outputs:
            response = outputs[0].outputs[0].text.strip()
            return response
        return ""
        
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'llm'):
            del self.llm