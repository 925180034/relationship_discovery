from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, List
from .base import BaseModelInference, GenerationConfig

class HuggingFaceModel(BaseModelInference):
    """HuggingFace模型实现"""
    
    def __init__(self, model_name: str, 
                 generation_config: Optional[GenerationConfig] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model_name, generation_config)
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)
        
        # 确保模型有必要的token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """生成文本实现"""
        # Tokenize输入
        inputs = self.tokenizer(prompt, return_tensors='pt', 
                              padding=True, truncation=True).to(self.device)
        
        # 设置生成参数
        gen_kwargs = {
            'max_length': config.max_tokens,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'top_k': config.top_k,
            'num_return_sequences': config.num_return_sequences,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        if config.stop_sequences:
            gen_kwargs['stopping_criteria'] = self._create_stopping_criteria(
                config.stop_sequences)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
        # 解码并返回响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除prompt
        response = response[len(prompt):].strip()
        
        return response

    def _create_stopping_criteria(self, stop_sequences: List[str]):
        """创建停止条件"""
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopOnSequences(StoppingCriteria):
            def __init__(self, stops = [], tokenizer = None):
                super().__init__()
                self.stops = [tokenizer.encode(stop) for stop in stops]

            def __call__(self, input_ids, scores, **kwargs):
                for stop in self.stops:
                    if torch.all((input_ids[0][-len(stop):] == torch.tensor(stop))):
                        return True
                return False
                
        return StoppingCriteriaList([
            StopOnSequences(stop_sequences, self.tokenizer)
        ])