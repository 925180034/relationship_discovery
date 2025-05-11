# schema_matching/utils/data.py

import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载和预处理类"""
    
    @staticmethod
    def load_table(file_path: str, **kwargs) -> pd.DataFrame:
        """加载表格数据"""
        try:
            path = Path(file_path)
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(
                    file_path,
                    keep_default_na=False,  
                    na_values=[],           
                    dtype=str,              
                    **kwargs
                )
                
                # 转换数值类型列,但跳过nulls列
                for col in df.columns:
                    if col != 'nulls':
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except:
                            pass
                        
            elif path.suffix.lower() == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif path.suffix.lower() in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
            
            return DataCleaner.basic_clean(df)
            
        except Exception as e:
            logger.error(f"Failed to load table {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_ground_truth(file_path: str) -> List[Tuple[str, str]]:
        """加载Ground Truth映射,保持原始大小写"""
        try:
            with open(file_path, 'r') as f:
                mapping_data = json.load(f)
            
            matches = []
            for match in mapping_data.get('matches', []):
                source_col = match.get('source_column')
                target_col = match.get('target_column')
                if source_col and target_col:
                    # 保持原始大小写
                    matches.append((source_col, target_col))
            
            logger.info(f"加载了 {len(matches)} 个ground truth映射")
            return matches
            
        except Exception as e:
            logger.error(f"加载ground truth失败 {file_path}: {str(e)}")
            raise

class DataCleaner:
    """数据清洗和标准化类"""
    
    # 特殊空值标记列表
    SPECIAL_NULL_VALUES = {'NA', 'null', 'N/A', 'NULL'}
    
    @staticmethod
    def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
        """基础数据清洗"""
        # 复制以避免修改原始数据
        df = df.copy()
        
        # 列名标准化
        df.columns = df.columns.str.strip().str.lower()
        
        # 处理nulls列的特殊逻辑
        if 'nulls' in df.columns:
            # 先处理字符串的前后空白
            df['nulls'] = df['nulls'].str.strip()
            
            # 将空字符串和纯空白转换为None
            df['nulls'] = df['nulls'].replace(r'^\s*$', None, regex=True)
            
            # 保留特殊空值标记和其他值不变
            def process_null_value(x):
                if pd.isna(x) or x is None:
                    return None
                if x in DataCleaner.SPECIAL_NULL_VALUES:
                    return x
                return x
            
            df['nulls'] = df['nulls'].apply(process_null_value)
        
        # 处理其他列
        for col in df.columns:
            if col != 'nulls' and df[col].dtype == object:
                df[col] = df[col].str.strip()
        
        # 移除全空的行和列
        df = df.dropna(how='all', axis=1)
        df = df.dropna(how='all', axis=0)
        
        return df
    

class DataSampler:
    """数据采样类"""
    
    @staticmethod
    def sample_tables(source_df: pd.DataFrame, target_df: pd.DataFrame,
                     n_samples: int = 5) -> Dict[str, Dict[str, List]]:
        """对两个表格进行采样
        
        Args:
            source_df: 源表格
            target_df: 目标表格
            n_samples: 每列的采样数量
            
        Returns:
            包含采样结果的字典，格式为：
            {
                'source': {'colA': [values...], 'colB': [values...]},
                'target': {'colX': [values...], 'colY': [values...]}
            }
        """
        try:
            logger.info(f"开始表格采样，样本数: {n_samples}")
            
            # 初始化结果字典
            samples = {
                'source': {},
                'target': {}
            }
            
            # 采样源表的每一列
            for col in source_df.columns:
                try:
                    samples['source'][col] = DataSampler.sample_column(source_df[col], n_samples)
                    logger.debug(f"源表列 {col} 采样完成，获取 {len(samples['source'][col])} 个样本")
                except Exception as e:
                    logger.warning(f"源表列 {col} 采样失败: {str(e)}")
                    samples['source'][col] = []

            # 采样目标表的每一列
            for col in target_df.columns:
                try:
                    samples['target'][col] = DataSampler.sample_column(target_df[col], n_samples)
                    logger.debug(f"目标表列 {col} 采样完成，获取 {len(samples['target'][col])} 个样本")
                except Exception as e:
                    logger.warning(f"目标表列 {col} 采样失败: {str(e)}")
                    samples['target'][col] = []

            # 验证结果结构
            for table_type in ['source', 'target']:
                for col, values in samples[table_type].items():
                    if not isinstance(values, list):
                        logger.warning(f"{table_type} 表 {col} 列的样本不是列表类型，进行转换")
                        samples[table_type][col] = list(values) if values is not None else []

            return samples
            
        except Exception as e:
            logger.error(f"表格采样失败: {str(e)}")
            # 返回空结构而不是抛出异常
            return {'source': {}, 'target': {}}

    @staticmethod
    def sample_column(series: pd.Series, n_samples: int = 5) -> List:
        """对列数据进行采样，返回字符串列表"""
        if len(series) <= n_samples:
            return [str(x) for x in series.dropna().tolist()]
            
        try:
            # 获取非空值
            non_null = series.dropna()
            if len(non_null) == 0:
                return []
                
            # 数值类型列：按分位数采样
            if pd.api.types.is_numeric_dtype(series):
                quantiles = np.linspace(0, 1, n_samples+2)[1:-1]
                samples = non_null.quantile(quantiles)
                return [f"{x:.2f}" if isinstance(x, float) else str(x) for x in samples]
                
            # 其他类型：按频率采样    
            value_counts = non_null.value_counts()
            if len(value_counts) <= n_samples:
                return [str(x) for x in value_counts.index]
            return [str(x) for x in value_counts.head(n_samples).index]
            
        except Exception as e:
            logger.warning(f"列采样失败: {str(e)}")
            return []