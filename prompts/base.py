# 新建文件 prompts/base.py:

from typing import Dict, Any
from core.interfaces import TableFeatures

class ColumnDescriptionHandler:
    """统一处理列描述的工具类"""
    
    @staticmethod
    def get_description(features: TableFeatures, column: str) -> str:
        """获取列描述,处理可能不存在的情况"""
        if not hasattr(features, 'structural_features'):
            return "No description available"
            
        struct_features = features.structural_features
        if not struct_features or 'descriptions' not in struct_features:
            return "No description available"
            
        return struct_features['descriptions'].get(column, "No description available")

class BasePromptBuilder:
    """基础提示词构建器"""
    
    def __init__(self):
        self.system_context = {
            "role": "You are an AI assistant specialized in schema matching.",
            "expertise": "Your expertise lies in identifying semantically equivalent columns between database tables.",
            "capability": "You can analyze column names, descriptions, data types, and content to determine semantic equivalence."
        }
        self.description_handler = ColumnDescriptionHandler()
    
    def _build_system_prompt(self) -> str:
        return f"{self.system_context['role']} {self.system_context['expertise']} {self.system_context['capability']}"

    def _build_semantic_guidance(self) -> str:
        return """
        Semantic equivalence means that two columns represent the same real-world concept or information,
        even if they use different names or formats.
        """
        
    def _build_output_format(self) -> str:
        return """
        Please conclude with a clear Yes/No answer based on semantic equivalence.
        Provide your decision in a new line containing only 'Yes' or 'No'.
        """