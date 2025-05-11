# validation/validators.py
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from core.interfaces import ResultValidator

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results"""
    precision: float
    recall: float
    f1_score: float
    type_compatibility: Dict[Tuple[str, str], bool]
    value_similarity: Dict[Tuple[str, str], float]
    additional_metrics: Dict[str, float]

class BaseValidator(ResultValidator):
    """Base validator implementing common validation logic"""
    
    def validate(self, matches: List[tuple],
                source_table: pd.DataFrame,
                target_table: pd.DataFrame,  
                ground_truth: Optional[List[tuple]] = None) -> Dict[str, float]:
        """修改验证方法确保返回原生类型"""
        metrics = {}
        
        try:
            if ground_truth:
                match_set = set(matches)
                truth_set = set(ground_truth)
                
                true_positives = len(match_set.intersection(truth_set))
                false_positives = len(match_set - truth_set)
                false_negatives = len(truth_set - match_set)
                
                precision = float(true_positives / (true_positives + false_positives) 
                            if true_positives + false_positives > 0 else 0)
                recall = float(true_positives / (true_positives + false_negatives)
                        if true_positives + false_negatives > 0 else 0)
                f1 = float(2 * precision * recall / (precision + recall)
                        if precision + recall > 0 else 0)
                        
                metrics = {
                    'precision': precision,
                    'recall': recall, 
                    'f1_score': f1,
                    'type_compatibility_rate': float(self._check_type_compatibility(matches, source_table, target_table)),
                    'avg_value_similarity': float(self._compute_avg_similarity(matches, source_table, target_table)),
                    'source_coverage': float(len(set(s for s,t in matches)) / len(source_table.columns)),
                    'target_coverage': float(len(set(t for s,t in matches)) / len(target_table.columns)),
                    'one_to_one_ratio': float(self._compute_one_to_one_ratio(matches))
                }
                
            return metrics
        
        except Exception as e:
            logger.error(f"计算评估指标时出错: {str(e)}")
            return metrics
    
    def _compute_metrics_with_ground_truth(self,
                                         matches: List[tuple],
                                         ground_truth: List[tuple]) -> Dict[str, float]:
        """使用ground truth计算指标"""
        match_set = set(matches)
        truth_set = set(ground_truth)
        
        true_positives = len(match_set.intersection(truth_set))
        false_positives = len(match_set - truth_set)
        false_negatives = len(truth_set - match_set)
        
        precision = (true_positives / (true_positives + false_positives) 
                    if true_positives + false_positives > 0 else 0)
        recall = (true_positives / (true_positives + false_negatives)
                 if true_positives + false_negatives > 0 else 0)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    def _compute_estimated_metrics(self,
                                 matches: List[tuple],
                                 source_table: pd.DataFrame,
                                 target_table: pd.DataFrame) -> Dict[str, float]:
        """在没有ground truth时估算指标"""
        # 计算列覆盖率
        source_cols = set(source_table.columns)
        target_cols = set(target_table.columns)
        matched_source = {src for src, _ in matches}
        matched_target = {tgt for _, tgt in matches}
        
        source_coverage = len(matched_source) / len(source_cols)
        target_coverage = len(matched_target) / len(target_cols)
        
        # 计算一对一匹配比例
        source_counts = pd.Series([src for src, _ in matches]).value_counts()
        target_counts = pd.Series([tgt for _, tgt in matches]).value_counts()
        one_to_one_ratio = (
            sum((source_counts == 1) & (target_counts == 1)) / len(matches)
            if matches else 0
        )
        
        # 基于覆盖率和一对一比例估算指标
        estimated_precision = one_to_one_ratio * min(source_coverage, target_coverage)
        estimated_recall = source_coverage * target_coverage
        estimated_f1 = (2 * estimated_precision * estimated_recall /
                       (estimated_precision + estimated_recall)
                       if estimated_precision + estimated_recall > 0 else 0)
        
        return {
            'precision': estimated_precision,
            'recall': estimated_recall,
            'f1_score': estimated_f1,
            'source_coverage': source_coverage,
            'target_coverage': target_coverage,
            'one_to_one_ratio': one_to_one_ratio
        }
    
    def _check_type_compatibility(self,
                                matches: List[tuple],
                                source_table: pd.DataFrame,
                                target_table: pd.DataFrame) -> Dict[Tuple[str, str], bool]:
        """检查数据类型兼容性"""
        compatibility = {}
        
        for source_col, target_col in matches:
            source_type = source_table[source_col].dtype
            target_type = target_table[target_col].dtype
            
            # 检查类型是否兼容
            compatible = self._are_types_compatible(source_type, target_type)
            compatibility[(source_col, target_col)] = compatible
            
        return compatibility
    
    def _are_types_compatible(self, type1: np.dtype, type2: np.dtype) -> bool:
        """检查两个数据类型是否兼容"""
        
        def is_numeric(dtype):
            return np.issubdtype(dtype, np.number)
            
        def is_string_like(dtype):
            return dtype == object or pd.api.types.is_string_dtype(dtype)
            
        def is_datetime(dtype):
            return pd.api.types.is_datetime64_any_dtype(dtype)
            
        # 同类型兼容
        if type1 == type2:
            return True
            
        # 数值类型兼容
        if is_numeric(type1) and is_numeric(type2):
            return True
            
        # 字符串类型兼容
        if is_string_like(type1) and is_string_like(type2):
            return True
            
        # 日期时间类型兼容
        if is_datetime(type1) and is_datetime(type2):
            return True
            
        return False
    
    def _compute_value_similarity(self,
                                matches: List[tuple],
                                source_table: pd.DataFrame,
                                target_table: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """计算匹配列的值相似度"""
        similarities = {}
        
        for source_col, target_col in matches:
            source_values = source_table[source_col].dropna()
            target_values = target_table[target_col].dropna()
            
            if len(source_values) == 0 or len(target_values) == 0:
                similarities[(source_col, target_col)] = 0
                continue
                
            # 根据数据类型选择相似度计算方法
            if (np.issubdtype(source_values.dtype, np.number) and 
                np.issubdtype(target_values.dtype, np.number)):
                sim = self._compute_numeric_similarity(source_values, target_values)
            else:
                sim = self._compute_categorical_similarity(source_values, target_values)
                
            similarities[(source_col, target_col)] = sim
            
        return similarities
    
    def _compute_numeric_similarity(self, 
                                 source_values: pd.Series,
                                 target_values: pd.Series) -> float:
        """计算数值类型列的相似度"""
        try:
            # 标准化
            source_norm = (source_values - source_values.mean()) / source_values.std()
            target_norm = (target_values - target_values.mean()) / target_values.std()
            
            # 比较分布
            source_hist = np.histogram(source_norm, bins=10)[0]
            target_hist = np.histogram(target_norm, bins=10)[0]
            
            # 归一化直方图
            source_hist = source_hist / source_hist.sum()
            target_hist = target_hist / target_hist.sum()
            
            # 计算相似度
            similarity = 1 - np.mean(np.abs(source_hist - target_hist))
            return similarity
            
        except Exception:
            # 如果数值计算失败，降级为分类比较
            return self._compute_categorical_similarity(
                source_values.astype(str), target_values.astype(str))
    
    def _compute_categorical_similarity(self,
                                    source_values: pd.Series,
                                    target_values: pd.Series) -> float:
        """计算分类/字符串类型列的相似度"""
        # 获取值分布
        source_dist = source_values.value_counts(normalize=True)
        target_dist = target_values.value_counts(normalize=True)
        
        # 获取唯一值
        all_values = set(source_dist.index) | set(target_dist.index)
        
        # 计算Jaccard相似度
        intersection = len(set(source_dist.index) & set(target_dist.index))
        union = len(all_values)
        
        jaccard_sim = intersection / union if union > 0 else 0
        
        # 计算分布相似度
        common_values = set(source_dist.index) & set(target_dist.index)
        if common_values:
            dist_diff = sum(abs(source_dist.get(val, 0) - target_dist.get(val, 0))
                         for val in common_values) / len(common_values)
            dist_sim = 1 - dist_diff
        else:
            dist_sim = 0
        
        # 综合相似度
        return 0.5 * jaccard_sim + 0.5 * dist_sim

class ComprehensiveValidator(ResultValidator):
    """Enhanced validator with case-insensitive matching"""
    
    def __init__(self):
        self.name_mapping = {}  # 原始列名到标准化列名的映射
        
    def _normalize_column_name(self, column: str) -> str:
        """标准化列名,保存原始映射"""
        norm_name = column.lower()
        self.name_mapping[norm_name] = column
        return norm_name
    
    def validate(self,
                matches: List[tuple],
                source_table: pd.DataFrame,
                target_table: pd.DataFrame,
                ground_truth: Optional[List[tuple]] = None) -> Dict[str, float]:
        """使用大小写不敏感的方式验证结果"""
        # 重置映射
        self.name_mapping.clear()
        
        # 标准化预测的匹配
        norm_matches = set()
        for src, tgt in matches:
            norm_src = self._normalize_column_name(src)
            norm_tgt = self._normalize_column_name(tgt)
            norm_matches.add((norm_src, norm_tgt))
            
        # 标准化ground truth
        if ground_truth:
            norm_truth = set()
            for src, tgt in ground_truth:
                norm_src = self._normalize_column_name(src)
                norm_tgt = self._normalize_column_name(tgt)
                norm_truth.add((norm_src, norm_tgt))
                
            # 计算精确度
            true_positives = len(norm_matches & norm_truth)
            if len(norm_matches) > 0:
                precision = true_positives / len(norm_matches)
            else:
                precision = 0.0
                
            # 计算召回率
            if len(norm_truth) > 0:
                recall = true_positives / len(norm_truth)
            else:
                recall = 0.0
                
            # 计算F1分数
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
                
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            
        # 类型兼容性
        type_compat = self._check_type_compatibility(matches, source_table, target_table)
        metrics['type_compatibility_rate'] = (
            sum(type_compat.values()) / len(type_compat) if type_compat else 0)
            
        # 值相似度
        value_sim = self._compute_value_similarity(matches, source_table, target_table)
        metrics['avg_value_similarity'] = (
            sum(value_sim.values()) / len(value_sim) if value_sim else 0)
            
        # 计算覆盖率
        norm_source_cols = {self._normalize_column_name(col) for col in source_table.columns}
        norm_target_cols = {self._normalize_column_name(col) for col in target_table.columns}
        
        matched_source = {src for src, _ in norm_matches}
        matched_target = {tgt for _, tgt in norm_matches}
        
        metrics['source_coverage'] = len(matched_source) / len(norm_source_cols)
        metrics['target_coverage'] = len(matched_target) / len(norm_target_cols)
        
        # 一对一比率
        if matches:
            source_counts = pd.Series([src for src, _ in norm_matches]).value_counts()
            target_counts = pd.Series([tgt for _, tgt in norm_matches]).value_counts()
            one_to_one = sum((source_counts == 1) & (target_counts == 1))
            metrics['one_to_one_ratio'] = one_to_one / len(matches)
        else:
            metrics['one_to_one_ratio'] = 0.0
            
        return metrics
        
    def _check_type_compatibility(self,
                                matches: List[tuple],
                                source_table: pd.DataFrame,
                                target_table: pd.DataFrame) -> Dict[Tuple[str, str], bool]:
        """检查匹配列的数据类型兼容性"""
        compatibility = {}
        
        for source_col, target_col in matches:
            source_type = source_table[source_col].dtype
            target_type = target_table[target_col].dtype
            
            # 检查类型是否兼容
            compatible = self._are_types_compatible(source_type, target_type)
            compatibility[(source_col, target_col)] = compatible
            
        return compatibility
        
    def _are_types_compatible(self, type1: np.dtype, type2: np.dtype) -> bool:
        """检查两个数据类型是否兼容"""
        
        def is_numeric(dtype):
            return np.issubdtype(dtype, np.number)
            
        def is_string_like(dtype):
            return dtype == object or pd.api.types.is_string_dtype(dtype)
            
        def is_datetime(dtype):
            return pd.api.types.is_datetime64_any_dtype(dtype)
            
        # 同类型兼容
        if type1 == type2:
            return True
            
        # 数值类型兼容
        if is_numeric(type1) and is_numeric(type2):
            return True
            
        # 字符串类型兼容
        if is_string_like(type1) and is_string_like(type2):
            return True
            
        # 日期时间类型兼容
        if is_datetime(type1) and is_datetime(type2):
            return True
            
        return False
        
    def _compute_value_similarity(self,
                                matches: List[tuple],
                                source_table: pd.DataFrame,
                                target_table: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """计算匹配列的值相似度"""
        similarities = {}
        
        for source_col, target_col in matches:
            # 获取列数据
            source_values = source_table[source_col]
            target_values = target_table[target_col]
            
            # 计算相似度
            try:
                if pd.api.types.is_numeric_dtype(source_values) and pd.api.types.is_numeric_dtype(target_values):
                    # 数值列用分布相似度
                    similarity = self._compute_numeric_similarity(source_values, target_values)
                else:
                    # 非数值列用分类相似度
                    similarity = self._compute_categorical_similarity(source_values, target_values)
                    
                similarities[(source_col, target_col)] = similarity
                
            except Exception as e:
                logger.warning(f"计算值相似度失败: {str(e)}")
                similarities[(source_col, target_col)] = 0.0
                
        return similarities
        
    def _compute_numeric_similarity(self, col1: pd.Series, col2: pd.Series) -> float:
        """计算数值列的相似度"""
        try:
            # 计算基本统计量
            stats1 = col1.describe()
            stats2 = col2.describe()
            
            # 标准化数据
            norm1 = (col1 - stats1['mean']) / stats1['std']
            norm2 = (col2 - stats2['mean']) / stats2['std']
            
            # 计算分布相似度
            similarity = 1 - abs(norm1.mean() - norm2.mean())
            return max(0, min(similarity, 1))
            
        except Exception:
            return 0.0
            
    def _compute_categorical_similarity(self, col1: pd.Series, col2: pd.Series) -> float:
        """计算分类列的相似度"""
        try:
            # 计算值分布
            dist1 = col1.value_counts(normalize=True)
            dist2 = col2.value_counts(normalize=True)
            
            # 计算共同值
            common_values = set(dist1.index) & set(dist2.index)
            
            if not common_values:
                return 0.0
                
            # 计算分布相似度
            similarity = 1 - sum(abs(dist1.get(val, 0) - dist2.get(val, 0)) 
                               for val in common_values) / len(common_values)
            
            return max(0, min(similarity, 1))
            
        except Exception:
            return 0.0