# schema_matching/tests/test_basic_components.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from utils.data import DataLoader, DataCleaner, DataSampler
from features.extractors import ComprehensiveFeatureExtractor
from prompts.builders import (
    MetadataPromptBuilder, 
    FewInstancesPromptBuilder,
    RichInstancesPromptBuilder
)
from core.interfaces import TableFeatures

class TestBasicComponents:
    """基础组件单元测试"""
    
    @pytest.fixture
    def sample_tables(self):
        """创建测试用的表格数据"""
        # 源表
        source_df = pd.DataFrame({
            'user_id': range(1, 11),
            'username': [f'user_{i}' for i in range(1, 11)],
            'age': np.random.randint(18, 80, 10),
            'nulls': ["", "NA", "null", "N/A", "", "", "test", "test2", "", "NULL"]
        })
        
        # 目标表
        target_df = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'user_{i}' for i in range(1, 11)],
            'user_age': np.random.randint(18, 80, 10),
            'category': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B']
        })
        
        return {'source': source_df, 'target': target_df}
    
    @pytest.fixture
    def test_files(self, sample_tables, tmp_path):
        """准备测试文件"""
        # 创建CSV和映射文件
        files = {}
        
        # 保存源表和目标表
        files['source_path'] = tmp_path / "source.csv"
        files['target_path'] = tmp_path / "target.csv"
        sample_tables['source'].to_csv(files['source_path'], index=False)
        sample_tables['target'].to_csv(files['target_path'], index=False)
        
        # 创建映射文件
        mapping = {
            "matches": [
                {"source_column": "user_id", "target_column": "id"},
                {"source_column": "username", "target_column": "name"}
            ]
        }
        files['mapping_path'] = tmp_path / "mapping.json"
        with open(files['mapping_path'], 'w') as f:
            json.dump(mapping, f)
            
        return files
    
    @pytest.fixture
    def extracted_features(self, sample_tables):
        """提取特征用于prompt测试"""
        extractor = ComprehensiveFeatureExtractor()
        source_features = extractor.extract_features(sample_tables['source'])
        target_features = extractor.extract_features(sample_tables['target'])
        return source_features, target_features
    
    def test_data_utils(self, test_files, sample_tables):
        """测试数据加载和处理工具"""
        # 1. 测试 DataLoader
        # 测试 CSV 加载
        source_df = DataLoader.load_table(test_files['source_path'])
        assert isinstance(source_df, pd.DataFrame)

        # 匹配清洗逻辑调整期望值
        source_expected = sample_tables['source'].copy()
        source_expected['nulls'] = source_expected['nulls'].replace({"": None, " ": None})
        source_expected['nulls'] = source_expected['nulls'].where(
            source_expected['nulls'].isin(['NA', 'null', 'N/A', 'NULL']), source_expected['nulls']
        )

        # 验证 DataFrame 是否相等
        pd.testing.assert_frame_equal(source_df, source_expected, check_dtype=False)

        # 测试映射加载
        matches = DataLoader.load_ground_truth(test_files['mapping_path'])
        assert len(matches) == 2
        assert matches[0] == ('user_id', 'id')

        # 测试错误处理
        with pytest.raises(ValueError):
            DataLoader.load_table("invalid.txt")
            
        # 2. 测试DataCleaner
        df = sample_tables['source']
        cleaned_df = DataCleaner.basic_clean(df)
        assert all(col.islower() for col in cleaned_df.columns)
        assert cleaned_df['nulls'].isna().sum() == 6
        
        # 3. 测试DataSampler
        n_samples = 5
        sampled_tables = DataSampler.sample_tables(
            sample_tables['source'],
            sample_tables['target'],
            n_samples
        )
        assert len(sampled_tables['source']) == n_samples
        
    def test_feature_extractor(self, sample_tables):
        """测试特征提取器"""
        extractor = ComprehensiveFeatureExtractor()
        
        # 测试源表特征提取
        source_features = extractor.extract_features(sample_tables['source'])
        assert isinstance(source_features, TableFeatures)
        
        # 验证文本特征
        assert 'column_names' in source_features.text_features
        assert len(source_features.text_features['column_names']) == len(sample_tables['source'].columns)
        
        # 验证结构特征
        assert 'dtypes' in source_features.structural_features
        assert all(col in source_features.structural_features['dtypes'] 
                  for col in sample_tables['source'].columns)
        
        # 验证语义特征(如果启用)
        if source_features.semantic_features:
            assert 'column_embeddings' in source_features.semantic_features
            
        # 测试目标表特征提取
        target_features = extractor.extract_features(sample_tables['target'])
        assert isinstance(target_features, TableFeatures)
    
    def test_prompt_builders(self, extracted_features):
        """测试提示词构建器"""
        source_features, target_features = extracted_features
        column_pair = ('user_id', 'id')
        
        # 1. 测试元数据提示词构建器
        metadata_builder = MetadataPromptBuilder()
        metadata_prompt = metadata_builder.build_prompt(
            source_features, target_features, column_pair)
        assert isinstance(metadata_prompt, str)
        assert 'user_id' in metadata_prompt
        assert 'id' in metadata_prompt
        
        # 2. 测试少量样本提示词构建器
        few_instances_builder = FewInstancesPromptBuilder()
        sample_data = {
            'source': ['1', '2', '3'],
            'target': ['1', '2', '3']
        }
        few_prompt = few_instances_builder.build_prompt(
            source_features, target_features,
            column_pair, sample_data
        )
        assert isinstance(few_prompt, str)
        assert all(str(i) in few_prompt for i in range(1, 4))
        
        # 3. 测试丰富数据提示词构建器
        rich_instances_builder = RichInstancesPromptBuilder()
        rich_prompt = rich_instances_builder.build_prompt(
            source_features, target_features,
            column_pair, 0.95
        )
        assert isinstance(rich_prompt, str)
        assert '0.95' in rich_prompt

