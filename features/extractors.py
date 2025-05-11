# features/extractors.py
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from core.interfaces import FeatureExtractor, TableFeatures
import logging

logger = logging.getLogger(__name__)

class BaseFeatureExtractor(FeatureExtractor):
    """Base feature extractor implementing common functionality"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer()
    
    def _get_column_samples(self, table: pd.DataFrame, max_samples: int = 5) -> Dict[str, List]:
        """Get sample values for each column"""
        samples = {}
        for col in table.columns:
            # Get non-null values
            values = table[col].dropna().astype(str).head(max_samples).tolist()
            samples[col] = values
        return samples
    
    def _get_basic_stats(self, table: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate basic statistics for each column"""
        stats = {}
        for col in table.columns:
            col_stats = {
                'null_count': table[col].isnull().sum(),
                'unique_count': table[col].nunique(),
                'value_counts': table[col].value_counts().head().to_dict()
            }
            
            # Add numeric statistics if applicable
            if pd.api.types.is_numeric_dtype(table[col]):
                desc = table[col].describe().to_dict()
                col_stats.update(desc)
                
            stats[col] = col_stats
        return stats

class ComprehensiveFeatureExtractor(FeatureExtractor):
    """提取全面的特征集,支持开关不同特征类型"""
    
    def __init__(self, 
                text: bool = True,
                structural: bool = True, 
                semantic: bool = False,  # 默认关闭语义特征
                semantic_model: str = "paraphrase-MiniLM-L3-v2",
                cache_enabled: bool = False,
                **kwargs):
        """初始化特征提取器"""
        
        # 加载配置
        self.extract_text = text
        self.extract_structural = structural
        self.extract_semantic = semantic
        self.semantic_model_name = semantic_model
        self.cache_enabled = cache_enabled
        
        logger.info(f"特征提取器配置:")
        logger.info(f"  text: {text}")
        logger.info(f"  structural: {structural}")
        logger.info(f"  semantic: {semantic}")
        logger.info(f"  semantic_model: {semantic_model}")
        logger.info(f"  cache_enabled: {cache_enabled}")
        
        # 初始化组件
        if self.extract_text:
            self.tfidf = TfidfVectorizer()
            
        if self.extract_semantic:
            self.semantic_model = SentenceTransformer(self.semantic_model_name)
    
    def extract_features(self, table: pd.DataFrame) -> TableFeatures:
        """确保特征都是原生类型"""
        
        def convert_value(v):
            if isinstance(v, (np.integer, np.int64)):
                return int(v)
            elif isinstance(v, (np.floating, np.float64)):
                return float(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            return v
            
        # 文本特征
        text_features = {
            'column_names': list(table.columns)
        }
        
        # 结构特征
        structural_features = {
            'dtypes': {col: str(dtype) for col, dtype in table.dtypes.items()},
            'null_counts': {col: convert_value(val) for col, val in table.isnull().sum().items()},
            'unique_counts': {col: convert_value(val) for col, val in table.nunique().items()}
        }
        
        # 语义特征
        semantic_features = {}
        if self.extract_semantic:
            semantic_features = {
                k: convert_value(v) for k, v in self._extract_semantic_features(table).items()
            }
            
        return TableFeatures(
            text_features=text_features,
            structural_features=structural_features,
            semantic_features=semantic_features
        )
        
    def extract(self, table: pd.DataFrame) -> TableFeatures:
        """为了向后兼容的别名方法"""
        return self.extract_features(table)
        
    def _extract_text_features(self, table: pd.DataFrame) -> Dict[str, Any]:
        """提取文本特征"""
        features = {}
        
        # TF-IDF特征
        col_name_text = [' '.join(col.split('_')) for col in table.columns]
        col_name_matrix = self.tfidf.fit_transform(col_name_text)
        features['column_name_tfidf'] = col_name_matrix
        features['tfidf_vocabulary'] = self.tfidf.vocabulary_
        
        # 采样值
        features['column_samples'] = self._get_column_samples(table)
        
        return features
    
    def _extract_value_patterns(self, table: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """提取每列的值模式特征"""
        patterns = {}
        
        for col in table.columns:
            try:
                col_patterns = {}
                series = table[col]
                
                # 基本统计
                col_patterns['length'] = {
                    'min': len(str(series.min())) if series.any() else 0,
                    'max': len(str(series.max())) if series.any() else 0
                }
                
                # 数据类型相关模式
                if pd.api.types.is_numeric_dtype(series):
                    col_patterns['numeric'] = {
                        'min': float(series.min()) if not series.empty else 0,
                        'max': float(series.max()) if not series.empty else 0,
                        'mean': float(series.mean()) if not series.empty else 0,
                        'is_integer': all(series.dropna().apply(float.is_integer))
                    }
                
                # 改进日期检测
                elif pd.api.types.is_string_dtype(series):
                    # 尝试多种常见日期格式
                    date_formats = [
                        '%Y-%m-%d',
                        '%d/%m/%Y',
                        '%Y/%m/%d',
                        '%d-%m-%Y',
                        '%Y%m%d'
                    ]
                    
                    is_date = False
                    for date_format in date_formats:
                        try:
                            pd.to_datetime(series.dropna().head(), format=date_format)
                            is_date = True
                            break
                        except:
                            continue
                            
                    col_patterns['is_date'] = is_date
                        
                    # 文本模式
                    sample = series.dropna().head(100)
                    col_patterns['text'] = {
                        'has_numbers': any(char.isdigit() for s in sample for char in str(s)),
                        'has_special': any(not char.isalnum() for s in sample for char in str(s)),
                        'avg_length': sum(len(str(s)) for s in sample) / len(sample) if not sample.empty else 0
                    }
                
                patterns[col] = col_patterns
                
            except Exception as e:
                logger.warning(f"提取列 {col} 的模式时出错: {str(e)}")
                patterns[col] = {}
                
        return patterns

    def _get_basic_stats(self, table: pd.DataFrame) -> Dict[str, Dict]:
        """计算基础统计信息"""
        stats = {}
        for col in table.columns:
            try:
                col_stats = {
                    'null_count': 0,  # 初始化默认值
                    'unique_count': 1
                }
                
                # 计算空值数量
                null_count = table[col].isnull().sum()
                if not pd.isna(null_count):
                    col_stats['null_count'] = int(null_count)
                    
                # 计算唯一值数量
                unique_count = table[col].nunique()
                if not pd.isna(unique_count):
                    col_stats['unique_count'] = int(unique_count)
                    
                # 值分布
                try:
                    value_counts = table[col].value_counts().head()
                    col_stats['value_counts'] = {
                        str(k): int(v) for k, v in value_counts.items()
                    }
                except:
                    col_stats['value_counts'] = {}
                
                # 数值列的统计
                if pd.api.types.is_numeric_dtype(table[col]):
                    desc = table[col].describe()
                    for k, v in desc.items():
                        if not pd.isna(v):
                            col_stats[k] = float(v)
                    
                stats[col] = col_stats
                
            except Exception as e:
                logger.warning(f"Error calculating stats for column {col}: {str(e)}")
                stats[col] = {
                    'null_count': 0,
                    'unique_count': 1,
                    'value_counts': {}
                }
                
        return stats
    
        
    # def _extract_text_features(self, table: pd.DataFrame) -> Dict[str, Any]:
    #     """提取文本特征"""
    #     features = {}
        
    #     # 列名
    #     features['column_names'] = list(table.columns)
        
    #     # TF-IDF特征
    #     col_name_text = [' '.join(col.split('_')) for col in table.columns]
    #     col_name_matrix = self.tfidf.fit_transform(col_name_text)
    #     features['column_name_tfidf'] = col_name_matrix
    #     features['tfidf_vocabulary'] = self.tfidf.vocabulary_
        
    #     # 采样值
    #     features['column_samples'] = self._get_column_samples(table)
        
    #     return features
    
    def _extract_structural_features(self, table: pd.DataFrame) -> Dict[str, Any]:
        """提取结构特征"""
        features = {}
        
        # 字符串列的长度统计
        str_columns = table.select_dtypes(include=['object']).columns
        value_lengths = {}
        for col in str_columns:
            try:
                length_stats = table[col].str.len().describe().to_dict()
                # 确保统计值是数值类型
                length_stats = {
                    k: float(v) if not pd.isna(v) else 0.0 
                    for k, v in length_stats.items()
                }
                value_lengths[col] = length_stats
            except:
                value_lengths[col] = {}
                
        features['value_lengths'] = value_lengths
        
        return features
    
    def _extract_semantic_features(self, table: pd.DataFrame) -> Dict[str, Any]:
        """提取语义特征"""
        features = {}
        
        # 列名编码
        col_name_text = [' '.join(col.split('_')) for col in table.columns]
        features['column_embeddings'] = self.semantic_model.encode(col_name_text)
        
        # 采样值编码
        sample_embeddings = {}
        for col, samples in self._get_column_samples(table).items():
            if samples:
                try:
                    sample_embeddings[col] = self.semantic_model.encode(samples)
                except:
                    continue
        features['sample_embeddings'] = sample_embeddings
        
        return features
        
    def _get_column_samples(self, table: pd.DataFrame, max_samples: int = 5) -> Dict[str, List]:
        """获取列采样值"""
        samples = {}
        for col in table.columns:
            try:
                values = (table[col]
                         .dropna()
                         .astype(str)
                         .head(max_samples)
                         .tolist())
                samples[col] = values
            except:
                samples[col] = []
        return samples

class CachedFeatureExtractor(FeatureExtractor):
    """Feature extractor with caching support"""
    
    def __init__(self, base_extractor: FeatureExtractor):
        self.base_extractor = base_extractor
        self.cache = {}
    
    def extract_features(self, table: pd.DataFrame) -> TableFeatures:
        cache_key = self._generate_cache_key(table)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        features = self.base_extractor.extract_features(table)
        self.cache[cache_key] = features
        return features
    
    def _generate_cache_key(self, table: pd.DataFrame) -> str:
        """Generate cache key based on table content"""
        return pd.util.hash_pandas_object(table).sum()