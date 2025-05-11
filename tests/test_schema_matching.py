# tests/test_schema_matching.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from unittest.mock import Mock, patch

from core.workflow import SchemaMatchingWorkflow
from core.interfaces import (
    FeatureExtractor, PlanGenerator, 
    PromptBuilder, ModelInference, ResultValidator,
    TableFeatures, MatchingPlan
)
from utils.data import DataLoader, DataCleaner, DataSampler
from features.extractors import ComprehensiveFeatureExtractor
from prompts.builders import MetadataPromptBuilder
from planning.plan_generator import BasePlanGenerator
from models.ollama_model import OllamaModel
from validation.validators import BaseValidator

class TestBasicComponents:
    """基础组件单元测试"""
    
    @pytest.fixture
    def sample_tables(self):
        """创建测试用表格数据"""
        source_df = pd.DataFrame({
            'user_id': range(1, 11),
            'username': [f'user_{i}' for i in range(1, 11)],
            'age': np.random.randint(18, 80, 10),
            'email': [f'user_{i}@example.com' for i in range(1, 11)]
        })
        
        target_df = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'user_{i}' for i in range(1, 11)],
            'user_age': source_df['age'].values,
            'contact': [f'user_{i}@example.com' for i in range(1, 11)]
        })
        
        return source_df, target_df
        
    @pytest.fixture
    def test_files(self, sample_tables, tmp_path):
        """准备测试文件"""
        source_df, target_df = sample_tables
        files = {}
        
        # 保存源表和目标表
        files['source_path'] = tmp_path / "source.csv"
        files['target_path'] = tmp_path / "target.csv"
        source_df.to_csv(files['source_path'], index=False)
        target_df.to_csv(files['target_path'], index=False)
        
        # 创建映射文件
        mapping = {
            "matches": [
                {"source_column": "user_id", "target_column": "id"},
                {"source_column": "username", "target_column": "name"},
                {"source_column": "age", "target_column": "user_age"}
            ]
        }
        files['mapping_path'] = tmp_path / "mapping.json"
        with open(files['mapping_path'], 'w') as f:
            json.dump(mapping, f)
            
        return files
    
    def test_data_utils(self, test_files, sample_tables):
        """测试数据加载和处理工具"""
        # 1. 测试DataLoader
        source_df = DataLoader.load_table(test_files['source_path'])
        assert isinstance(source_df, pd.DataFrame)
        
        # 测试映射加载
        matches = DataLoader.load_ground_truth(test_files['mapping_path'])
        assert len(matches) == 3  # user_id->id, username->name, age->user_age
        
        # 测试错误处理
        with pytest.raises(ValueError):
            DataLoader.load_table("invalid.txt")
            
        # 2. 测试DataCleaner
        df = sample_tables[0]  # 使用source表
        cleaned_df = DataCleaner.basic_clean(df)
        assert all(col.islower() for col in cleaned_df.columns)
        
        # 3. 测试DataSampler
        n_samples = 5
        sampled_data = DataSampler.sample_tables(
            sample_tables[0], sample_tables[1], n_samples)
        assert isinstance(sampled_data, dict)
        assert 'source' in sampled_data
        assert 'target' in sampled_data
        for samples in sampled_data.values():
            assert isinstance(samples, dict)
            
    def test_feature_extractor(self, sample_tables):
        """测试特征提取器"""
        source_df, target_df = sample_tables
        extractor = ComprehensiveFeatureExtractor()
        
        # 测试源表特征提取
        source_features = extractor.extract_features(source_df)
        assert isinstance(source_features, TableFeatures)
        assert 'column_names' in source_features.text_features
        assert len(source_features.text_features['column_names']) == len(source_df.columns)
        
        # 测试目标表特征提取
        target_features = extractor.extract_features(target_df)
        assert isinstance(target_features, TableFeatures)
        assert 'column_names' in target_features.text_features
        assert len(target_features.text_features['column_names']) == len(target_df.columns)

class TestWorkflowIntegration:
    """工作流集成测试"""
    
    @pytest.fixture
    def sample_tables(self):
        """创建测试表格"""
        source_df = pd.DataFrame({
            'user_id': range(1, 11),
            'username': [f'user_{i}' for i in range(1, 11)],
            'age': np.random.randint(18, 80, 10),
            'email': [f'user_{i}@example.com' for i in range(1, 11)]
        })
        
        target_df = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'user_{i}' for i in range(1, 11)],
            'user_age': source_df['age'].values,
            'contact': [f'user_{i}@example.com' for i in range(1, 11)]
        })
        
        return source_df, target_df
        
    @pytest.fixture
    def mock_components(self, sample_tables):
        """创建模拟组件"""
        source_df, target_df = sample_tables
        
        feature_extractor = Mock(spec=FeatureExtractor)
        feature_extractor.extract_features.side_effect = [
            TableFeatures(
                text_features={'column_names': list(source_df.columns)},
                structural_features={'dtypes': source_df.dtypes.to_dict()}
            ),
            TableFeatures(
                text_features={'column_names': list(target_df.columns)},
                structural_features={'dtypes': target_df.dtypes.to_dict()}
            )
        ]
        
        similarity_matrix = np.random.rand(
            len(source_df.columns), 
            len(target_df.columns)
        )
        
        plan_generator = Mock(spec=PlanGenerator)
        plan_generator.generate_plan.return_value = MatchingPlan(
            steps=['step1', 'step2'],
            rules={'rule1': 'value1'},
            metadata={
                'similarity_scores': {
                    'combined': similarity_matrix,
                    'name': similarity_matrix,
                    'semantic': similarity_matrix
                }
            }
        )
        
        prompt_builder = Mock(spec=PromptBuilder)
        prompt_builder.build_prompt.return_value = "Test prompt"
        
        model = Mock(spec=ModelInference)
        model.generate.return_value = "Yes"
        
        validator = Mock(spec=ResultValidator)
        validator.validate.return_value = {
            'precision': 0.8,
            'recall': 0.7,
            'f1_score': 0.75,
            'type_compatibility_rate': 1.0,
            'avg_value_similarity': 0.9
        }
        
        return {
            'feature_extractor': feature_extractor,
            'plan_generator': plan_generator,
            'prompt_builder': prompt_builder,
            'model': model,
            'validator': validator
        }
        
    @pytest.fixture
    def workflow_config(self):
        """创建工作流配置"""
        return {
            'experiment': {
                'scenario': 'metadata',  # 默认场景
                'name': 'test_workflow'
            },
            'feature_extractor': {
                'type': 'comprehensive'
            },
            'plan_generator': {
                'type': 'base'
            },
            'model': {
                'type': 'mock'
            },
            'validator': {
                'type': 'base'
            }
        }
        
    @pytest.mark.parametrize("scenario", [
        pytest.param('metadata', id='test_metadata_scenario'),
        pytest.param('few_instances', id='test_few_instances_scenario'),
        pytest.param('rich_instances', id='test_rich_instances_scenario')
    ])
    def test_scenarios(self, sample_tables, mock_components,
                     workflow_config, scenario):
        """测试不同场景的执行"""
        source_df, target_df = sample_tables
        workflow_config['experiment']['scenario'] = scenario
        if scenario == 'few_instances':
            workflow_config['data'] = {'sample_size': 3}
            
        # 预先设置mock组件的行为
        mock_components['model'].generate.return_value = "Yes"
        
        workflow = SchemaMatchingWorkflow(
            feature_extractor=mock_components['feature_extractor'],
            plan_generator=mock_components['plan_generator'],
            prompt_builder=mock_components['prompt_builder'],
            model=mock_components['model'],
            validator=mock_components['validator'],
            config=workflow_config
        )
        
        result = workflow.execute(source_df, target_df)
        
        # 验证结果
        assert 'matches' in result
        assert 'metrics' in result
        assert 'plan' in result
        assert 'features' in result
        
        # 验证组件调用
        assert mock_components['prompt_builder'].build_prompt.call_count > 0
        mock_components['model'].generate.assert_called()
        mock_components['validator'].validate.assert_called_once()
    
    def test_error_handling(self, mock_components, workflow_config):
        """测试错误处理"""
        # 无效场景
        workflow_config['experiment']['scenario'] = 'invalid'
        with pytest.raises(ValueError, match="Unknown scenario: invalid"):
            workflow = SchemaMatchingWorkflow(
                feature_extractor=mock_components['feature_extractor'],
                plan_generator=mock_components['plan_generator'],
                prompt_builder=mock_components['prompt_builder'],
                model=mock_components['model'],
                validator=mock_components['validator'],
                config=workflow_config
            )
            workflow.execute(pd.DataFrame(), pd.DataFrame())
            
        # 特征提取失败
        workflow_config['experiment']['scenario'] = 'metadata'
        mock_components['feature_extractor'].extract_features.side_effect = \
            Exception("Feature extraction failed")
        workflow = SchemaMatchingWorkflow(
            feature_extractor=mock_components['feature_extractor'],
            plan_generator=mock_components['plan_generator'],
            prompt_builder=mock_components['prompt_builder'],
            model=mock_components['model'],
            validator=mock_components['validator'],
            config=workflow_config
        )
        with pytest.raises(Exception, match="Feature extraction failed"):
            workflow.execute(pd.DataFrame(), pd.DataFrame())
    
    @pytest.mark.integration
    def test_end_to_end(self, tmp_path, mock_components):
        """测试完整的端到端流程"""
        # 创建测试数据
        source_df = pd.DataFrame({
            'user_id': range(1, 5),
            'username': ['user1', 'user2', 'user3', 'user4'],
            'age': [25, 30, 35, 40]
        })
        
        target_df = pd.DataFrame({
            'id': range(1, 5),
            'name': ['user1', 'user2', 'user3', 'user4'],
            'user_age': [25, 30, 35, 40]
        })
        
        # 创建测试文件
        source_path = tmp_path / "source.csv"
        target_path = tmp_path / "target.csv"
        source_df.to_csv(source_path, index=False)
        target_df.to_csv(target_path, index=False)
        
        mapping = {
            "matches": [
                {"source_column": "user_id", "target_column": "id"},
                {"source_column": "username", "target_column": "name"},
                {"source_column": "age", "target_column": "user_age"}
            ]
        }
        
        mapping_path = tmp_path / "mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f)
            
        # 配置工作流
        config = {
            'experiment': {
                'scenario': 'metadata',
                'name': 'end_to_end_test'
            },
            'data': {
                'source_table': str(source_path),
                'target_table': str(target_path),
                'ground_truth': str(mapping_path)
            }
        }
        
        try:
            workflow = SchemaMatchingWorkflow(
                feature_extractor=ComprehensiveFeatureExtractor(),
                plan_generator=BasePlanGenerator(),
                prompt_builder=MetadataPromptBuilder(),
                model=mock_components['model'],  # 使用mock model
                validator=BaseValidator(),
                config=config
            )
            
            result = workflow.execute(source_df, target_df)
            
            # 验证结果
            assert isinstance(result, dict)
            assert 'matches' in result
            assert 'metrics' in result
            assert len(result['matches']) > 0
            assert result['metrics']['precision'] >= 0
            assert result['metrics']['recall'] >= 0
            
        finally:
            # 清理测试文件
            source_path.unlink()
            target_path.unlink()
            mapping_path.unlink()