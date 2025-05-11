# models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import logging
from core.interfaces import ModelInference

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """生成文本的配置参数"""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_return_sequences: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repeat_penalty: float = 1.1  # 添加repeat_penalty参数
    stop_sequences: Optional[List[str]] = None

    def __post_init__(self):
        """验证参数值的合法性"""
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("Top_p must be between 0 and 1")
        if self.top_k < 0:
            raise ValueError("Top_k must be positive")
        if self.presence_penalty < -2 or self.presence_penalty > 2:
            raise ValueError("Presence penalty must be between -2 and 2")
        if self.frequency_penalty < -2 or self.frequency_penalty > 2:
            raise ValueError("Frequency penalty must be between -2 and 2")
        if self.repeat_penalty < 0:
            raise ValueError("Repeat penalty must be positive")

class ResponseParser:
    """Parse model responses into structured format"""
    
    @staticmethod
    def parse_matches(response: str) -> List[Dict[str, Any]]:
        """Parse matching results from model response"""
        matches = []
        
        # Split response into lines
        lines = response.strip().split('\n')
        
        for line in lines:
            try:
                # Extract source and target columns
                if '->' not in line:
                    continue
                    
                # Parse basic components
                match_part, reasoning = line.split(':', 1)
                source_target, conf_part = match_part.split('(confidence:', 1)
                source_col, target_col = source_target.split('->')
                conf_score = float(conf_part.strip(' )'))
                
                matches.append({
                    'source_column': source_col.strip(),
                    'target_column': target_col.strip(),
                    'confidence': conf_score,
                    'reasoning': reasoning.strip()
                })
            except Exception as e:
                logger.warning(f"Failed to parse line: {line}. Error: {str(e)}")
                continue
                
        return matches

class BaseModelInference:
    def __init__(self, model_name: str, generation_config: Optional[Union[Dict, GenerationConfig]] = None):
        self.model_name = model_name
        if isinstance(generation_config, dict):
            # 如果是字典,转换为 GenerationConfig
            self.generation_config = GenerationConfig(**generation_config) if generation_config else GenerationConfig()
        else:
            # 如果已经是 GenerationConfig 或 None
            self.generation_config = generation_config or GenerationConfig()

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        try:
            # 更新生成配置
            config_dict = {
                field: getattr(self.generation_config, field) 
                for field in self.generation_config.__dataclass_fields__
            }
            config_dict.update(kwargs)
            config = GenerationConfig(**config_dict)
            
            # 实际生成
            return self._generate_text(prompt, config)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

