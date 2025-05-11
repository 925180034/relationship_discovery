import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import logging
import re

logger = logging.getLogger(__name__)

class SimilarityAnalyzer:
    """相似度分析器类"""
    
    def __init__(self, semantic_model: str = "paraphrase-MiniLM-L3-v2"):
        """初始化相似度分析器
        
        Args:
            semantic_model: 语义模型名称
        """
        self.semantic_model = SentenceTransformer(semantic_model)
        logger.info(f"初始化SimilarityAnalyzer，使用模型: {semantic_model}")
        
    def compute_similarities(self,
                           source_features: Dict[str, Any],
                           target_features: Dict[str, Any],
                           name_weight: float = 0.4,
                           semantic_weight: float = 0.3,
                           structural_weight: float = 0.3) -> Dict[str, np.ndarray]:
        """计算相似度矩阵
        
        Args:
            source_features: 源表特征
            target_features: 目标表特征 
            name_weight: 列名相似度权重
            semantic_weight: 语义相似度权重
            structural_weight: 结构相似度权重
            
        Returns:
            包含各类相似度矩阵的字典
        """
        try:
            # 获取列名列表
            source_cols = source_features.text_features.get('column_names', [])
            target_cols = target_features.text_features.get('column_names', [])
            
            if not source_cols or not target_cols:
                raise ValueError("未找到列名")
                
            # 计算列名相似度矩阵  
            name_sim = self._compute_name_similarity(source_cols, target_cols)
            
            # 计算语义相似度矩阵
            semantic_sim = self._compute_semantic_similarity(
                source_features, target_features)
                
            # 计算结构相似度矩阵
            structural_sim = self._compute_structural_similarity(
                source_features, target_features)
                
            # 加权组合相似度
            combined_sim = (
                name_weight * name_sim +
                semantic_weight * semantic_sim + 
                structural_weight * structural_sim
            )
            
            return {
                'name': name_sim,
                'semantic': semantic_sim,
                'structural': structural_sim,
                'combined': combined_sim
            }
            
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            raise
            
    def _compute_name_similarity(self,
                               source_cols: List[str],
                               target_cols: List[str]) -> np.ndarray:
        """计算列名相似度
        
        Args:
            source_cols: 源表列名列表
            target_cols: 目标表列名列表
            
        Returns:
            列名相似度矩阵
        """
        try:
            n_source = len(source_cols)
            n_target = len(target_cols)
            name_sim = np.zeros((n_source, n_target))
            
            for i, src_col in enumerate(source_cols):
                for j, tgt_col in enumerate(target_cols):
                    # 预处理列名
                    src_name = self._preprocess_column_name(src_col)
                    tgt_name = self._preprocess_column_name(tgt_col)
                    
                    # 计算编辑距离相似度
                    lev_sim = 1 - Levenshtein.distance(src_name, tgt_name) / \
                             max(len(src_name), len(tgt_name))
                             
                    # 计算Jaccard相似度
                    src_tokens = set(src_name.split())
                    tgt_tokens = set(tgt_name.split())
                    intersection = len(src_tokens & tgt_tokens)
                    union = len(src_tokens | tgt_tokens)
                    jaccard_sim = intersection / union if union > 0 else 0
                    
                    # 组合两种相似度
                    name_sim[i, j] = 0.7 * lev_sim + 0.3 * jaccard_sim
                    
            return name_sim
            
        except Exception as e:
            logger.error(f"计算列名相似度失败: {str(e)}")
            raise
            
    def _compute_semantic_similarity(self,
                                   source_features: Dict[str, Any],
                                   target_features: Dict[str, Any]) -> np.ndarray:
        """计算语义相似度
        
        Args:
            source_features: 源表特征
            target_features: 目标表特征
            
        Returns:
            语义相似度矩阵
        """
        try:
            # 获取源表和目标表的列名
            source_cols = source_features.text_features.get('column_names', [])
            target_cols = target_features.text_features.get('column_names', [])
            
            if not source_cols or not target_cols:
                raise ValueError("未找到列名")
                
            # 准备列名和描述文本
            source_texts = []
            for col in source_cols:
                desc = source_features.structural_features.get('descriptions', {}).get(col, '')
                text = f"{col} {desc}".strip()
                source_texts.append(text)
                
            target_texts = []
            for col in target_cols:
                desc = target_features.structural_features.get('descriptions', {}).get(col, '')
                text = f"{col} {desc}".strip()
                target_texts.append(text)
                
            # 计算文本embedding
            source_embeddings = self.semantic_model.encode(source_texts)
            target_embeddings = self.semantic_model.encode(target_texts)
            
            # 如果有语义特征，加入考虑
            if hasattr(source_features, 'semantic_features') and \
               hasattr(target_features, 'semantic_features'):
                source_semantic = source_features.semantic_features
                target_semantic = target_features.semantic_features
                
                if source_semantic and target_semantic:
                    # 获取样本嵌入
                    source_samples = source_semantic.get('sample_embeddings', {})
                    target_samples = target_semantic.get('sample_embeddings', {})
                    
                    # 计算样本相似度
                    sample_sim = np.zeros((len(source_cols), len(target_cols)))
                    for i, src_col in enumerate(source_cols):
                        for j, tgt_col in enumerate(target_cols):
                            if src_col in source_samples and tgt_col in target_samples:
                                src_embed = source_samples[src_col]
                                tgt_embed = target_samples[tgt_col]
                                if len(src_embed) > 0 and len(tgt_embed) > 0:
                                    sim = cosine_similarity(
                                        [np.mean(src_embed, axis=0)],
                                        [np.mean(tgt_embed, axis=0)]
                                    )[0, 0]
                                    sample_sim[i, j] = sim
                                    
                    # 组合列名和样本相似度
                    semantic_sim = cosine_similarity(source_embeddings, target_embeddings)
                    return 0.6 * semantic_sim + 0.4 * sample_sim
                    
            # 如果没有语义特征，只返回列名相似度
            return cosine_similarity(source_embeddings, target_embeddings)
            
        except Exception as e:
            logger.error(f"计算语义相似度失败: {str(e)}")
            raise
            
    def _compute_structural_similarity(self,
                                     source_features: Dict[str, Any],
                                     target_features: Dict[str, Any]) -> np.ndarray:
        """计算结构相似度
        
        Args:
            source_features: 源表特征
            target_features: 目标表特征
            
        Returns:
            结构相似度矩阵
        """
        try:
            # 获取列名列表
            source_cols = source_features.text_features.get('column_names', [])
            target_cols = target_features.text_features.get('column_names', [])
            
            if not source_cols or not target_cols:
                raise ValueError("未找到列名")
                
            n_source = len(source_cols)
            n_target = len(target_cols)
            struct_sim = np.zeros((n_source, n_target))
            
            # 获取数据类型信息
            source_dtypes = source_features.structural_features.get('dtypes', {})
            target_dtypes = target_features.structural_features.get('dtypes', {})
            
            # 获取统计信息
            source_stats = source_features.structural_features.get('basic_stats', {})
            target_stats = target_features.structural_features.get('basic_stats', {})
            
            for i, src_col in enumerate(source_cols):
                for j, tgt_col in enumerate(target_cols):
                    similarity = 0.0
                    count = 0
                    
                    # 比较数据类型
                    if src_col in source_dtypes and tgt_col in target_dtypes:
                        src_type = source_dtypes[src_col]
                        tgt_type = target_dtypes[tgt_col]
                        type_sim = self._compare_types(src_type, tgt_type)
                        similarity += type_sim
                        count += 1
                        
                    # 比较统计特征
                    if src_col in source_stats and tgt_col in target_stats:
                        src_stats = source_stats[src_col]
                        tgt_stats = target_stats[tgt_col]
                        stats_sim = self._compare_stats(src_stats, tgt_stats)
                        if stats_sim is not None:
                            similarity += stats_sim
                            count += 1
                            
                    struct_sim[i, j] = similarity / max(count, 1)
                    
            return struct_sim
            
        except Exception as e:
            logger.error(f"计算结构相似度失败: {str(e)}")
            raise
            
    def _preprocess_column_name(self, column: str) -> str:
        """预处理列名
        
        Args:
            column: 原始列名
            
        Returns:
            处理后的列名
        """
        # 转换为小写
        name = column.lower()
        
        # 分割驼峰命名
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        
        # 替换特殊字符为空格
        name = re.sub(r'[_\-.]', ' ', name)
        
        # 移除非字母数字字符
        name = re.sub(r'[^a-z0-9\s]', '', name)
        
        # 合并多个空格
        name = re.sub(r'\s+', ' ', name)
        
        return name.strip()
        
    def _compare_types(self, type1: str, type2: str) -> float:
        """比较数据类型相似度
        
        Args:
            type1: 第一个类型
            type2: 第二个类型
            
        Returns:
            类型相似度分数
        """
        # 转换为小写并移除空格
        type1 = type1.lower().strip()
        type2 = type2.lower().strip()
        
        # 如果类型完全相同
        if type1 == type2:
            return 1.0
            
        # 数值类型的相似度
        numeric_types = {'int', 'float', 'double', 'decimal', 'numeric'}
        if type1 in numeric_types and type2 in numeric_types:
            return 0.8
            
        # 字符串类型的相似度
        string_types = {'str', 'string', 'text', 'varchar', 'char'}
        if type1 in string_types and type2 in string_types:
            return 0.8
            
        # 日期时间类型的相似度
        date_types = {'date', 'time', 'datetime', 'timestamp'}
        if type1 in date_types and type2 in date_types:
            return 0.8
            
        # 默认返回较低相似度
        return 0.1
        
    def _compare_stats(self, stats1: Dict[str, Any], 
                      stats2: Dict[str, Any]) -> Optional[float]:
        """比较统计特征相似度
        
        Args:
            stats1: 第一列统计信息
            stats2: 第二列统计信息
            
        Returns:
            统计特征相似度分数
        """
        try:
            similarity = 0.0
            count = 0
            
            # 比较空值比例
            if 'null_count' in stats1 and 'null_count' in stats2:
                null_sim = 1 - abs(
                    stats1['null_count'] / stats1.get('total_count', 1) -
                    stats2['null_count'] / stats2.get('total_count', 1)
                )
                similarity += null_sim
                count += 1

            # 比较唯一值比例
            if 'unique_count' in stats1 and 'unique_count' in stats2:
                unique_sim = 1 - abs(
                    stats1['unique_count'] / stats1.get('total_count', 1) -
                    stats2['unique_count'] / stats2.get('total_count', 1)
                )
                similarity += unique_sim
                count += 1

            # 如果是数值类型，比较数值统计特征
            numeric_stats = {'mean', 'std', 'min', 'max'}
            if all(key in stats1 for key in numeric_stats) and \
               all(key in stats2 for key in numeric_stats):
                # 计算标准化后的差异
                for stat in numeric_stats:
                    if stats1['std'] != 0 and stats2['std'] != 0:
                        norm1 = (stats1[stat] - stats1['mean']) / stats1['std']
                        norm2 = (stats2[stat] - stats2['mean']) / stats2['std']
                        stat_sim = 1 - min(abs(norm1 - norm2), 1)
                        similarity += stat_sim
                        count += 1

            # 比较分布特征
            if 'value_counts' in stats1 and 'value_counts' in stats2:
                src_dist = pd.Series(stats1['value_counts'])
                tgt_dist = pd.Series(stats2['value_counts'])
                
                # 计算分布相似度
                all_values = set(src_dist.index) | set(tgt_dist.index)
                total_diff = 0
                for value in all_values:
                    src_prob = src_dist.get(value, 0) / src_dist.sum()
                    tgt_prob = tgt_dist.get(value, 0) / tgt_dist.sum()
                    total_diff += abs(src_prob - tgt_prob)
                    
                dist_sim = 1 - min(total_diff / 2, 1)  # 归一化到[0,1]
                similarity += dist_sim
                count += 1

            return similarity / max(count, 1) if count > 0 else None
                
        except Exception as e:
            logger.warning(f"比较统计特征失败: {str(e)}")
            return None