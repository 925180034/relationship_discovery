# prompts/builders.py

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.interfaces import PromptBuilder, TableFeatures

class BasePromptBuilder:
    """基础提示词构建器"""
    
    def __init__(self, template_dir: str = "prompts/templates"):
        self.template_dir = Path(template_dir)
        
    def _load_template(self, template_name: str) -> str:
        """加载模板文件"""
        template_path = self.template_dir / template_name
        if not template_path.exists():
            raise ValueError(f"Template not found: {template_name}")
            
        with open(template_path, 'r') as f:
            return f.read()
            
    def _get_column_info(self, features: TableFeatures, column: str) -> Dict[str, Any]:
        """获取列的基本信息"""
        return {
            'type': features.structural_features.get('dtypes', {}).get(column, 'unknown'),
            'desc': features.text_features.get('descriptions', {}).get(column, 'No description available')
        }

class MetadataPromptBuilder(BasePromptBuilder):
    """元数据场景的提示词构建器"""
    
    def build_prompt(self, 
                    source_features: TableFeatures,
                    target_features: TableFeatures,
                    column_pair: tuple,
                    similarity_score: float = 0.0) -> str:
        template = self._load_template("metadata_prompt.txt")
        
        source_col, target_col = column_pair
        source_info = self._get_column_info(source_features, source_col)
        target_info = self._get_column_info(target_features, target_col)
        
        return template.format(
            source_column=source_col,
            source_desc=source_info['desc'],
            source_type=source_info['type'],
            target_column=target_col,
            target_desc=target_info['desc'],
            target_type=target_info['type'],
            similarity_score=similarity_score
        )

class FewInstancesPromptBuilder(BasePromptBuilder):
    """少量样本场景的提示词构建器"""
    
    def build_prompt(self,
                    source_features: TableFeatures,
                    target_features: TableFeatures,
                    column_pair: tuple,
                    sample_data: Optional[Dict[str, List]] = None,
                    similarity_score: float = 0.0) -> str:
        """构建提示词"""
        template = self._load_template("few_instances_prompt.txt")
        
        source_col, target_col = column_pair
        source_info = self._get_column_info(source_features, source_col)
        target_info = self._get_column_info(target_features, target_col)
        
        # 安全获取样本数据
        source_samples = []
        target_samples = []
        if sample_data:
            if isinstance(sample_data, dict):
                source_samples = sample_data.get('source', [])
                target_samples = sample_data.get('target', [])
        
        return template.format(
            source_column=source_col,
            source_desc=source_info['desc'],
            source_type=source_info['type'],
            source_samples=", ".join(map(str, source_samples[:5])),
            target_column=target_col,
            target_desc=target_info['desc'],
            target_type=target_info['type'],
            target_samples=", ".join(map(str, target_samples[:5])),
            similarity_score=similarity_score
        )

class RichInstancesPromptBuilder(BasePromptBuilder):
    """丰富数据场景的提示词构建器"""
    
    def build_prompt(self,
                    source_features: TableFeatures,
                    target_features: TableFeatures,
                    column_pair: tuple,
                    sample_data: Optional[Dict[str, List]] = None,
                    stats: Optional[Dict[str, Dict]] = None,
                    similarity_score: float = 0.0) -> str:
        """构建提示词
        
        Args:
            source_features: 源表特征
            target_features: 目标表特征
            column_pair: 列对(source_column, target_column)
            sample_data: 采样数据 {"source": {...}, "target": {...}}
            stats: 统计信息 {"source": {...}, "target": {...}}
            similarity_score: 相似度分数
        """
        template = self._load_template("rich_instances_prompt.txt")
        
        source_col, target_col = column_pair
        source_info = self._get_column_info(source_features, source_col)
        target_info = self._get_column_info(target_features, target_col)
        
        # 获取统计信息
        source_stats = stats.get('source', {}).get(source_col, {}) if stats else {}
        target_stats = stats.get('target', {}).get(target_col, {}) if stats else {}
        
        # 获取样本数据
        source_samples = sample_data.get('source', {}).get(source_col, []) if sample_data else []
        target_samples = sample_data.get('target', {}).get(target_col, []) if sample_data else []
        
        return template.format(
            source_column=source_col,
            source_desc=source_info['desc'],
            source_type=source_info['type'],
            source_stats=self._format_stats(source_stats),
            source_samples=", ".join(map(str, source_samples[:5])),
            target_column=target_col,
            target_desc=target_info['desc'],
            target_type=target_info['type'],
            target_stats=self._format_stats(target_stats),
            target_samples=", ".join(map(str, target_samples[:5])),
            similarity_score=similarity_score
        )
        
    def _format_stats(self, stats: Dict) -> str:
        """格式化统计信息"""
        if not stats:
            return "No statistics available"
            
        formatted = []
        if 'null_count' in stats:
            formatted.append(f"null rate: {stats['null_count']}")
        if 'unique_count' in stats:
            formatted.append(f"unique values: {stats['unique_count']}")
        if 'value_counts' in stats:
            top_values = list(stats['value_counts'].items())[:3]
            formatted.append(f"top values: {top_values}")
            
        return "; ".join(formatted)