# tests/test_advanced_components.py
import pytest
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from models.ollama_model import OllamaModel
from unittest.mock import Mock, patch

from core.interfaces import TableFeatures
from planning.plan_generator import BasePlanGenerator, AdaptivePlanGenerator
from models.base import BaseModelInference
from models.hf_model import HuggingFaceModel
from models.vllm_model import VLLMModel
from validation.validators import BaseValidator, ComprehensiveValidator
from features.similarity import SimilarityAnalyzer

class TestPlanGenerator:
    """测试计划生成器"""
    
    @pytest.fixture
    def sample_features(self):
        """创建测试用的特征数据"""
        # 创建一个简单的表格
        df = pd.DataFrame({
            'user_id': range(1, 11),
            'username': [f'user_{i}' for i in range(1, 11)],
            'age': np.random.randint(18, 80, 10)
        })
        
        # 提取文本特征
        column_names = list(df.columns)
        # 创建一个简单的TF-IDF矩阵
        tfidf_matrix = np.random.rand(len(column_names), 5)
        
        # 创建特征对象
        features = TableFeatures(
            text_features={
                'column_names': column_names,
                'column_name_tfidf': tfidf_matrix
            },
            structural_features={
                'dtypes': df.dtypes.to_dict(),
                'statistics': {
                    col: {'unique_count': df[col].nunique()}
                    for col in df.columns
                }
            }
        )
        
        return features, df
    
    def test_base_plan_generator(self, sample_features):
        """测试基础计划生成器"""
        features, _ = sample_features
        generator = BasePlanGenerator()
        plan = generator.generate_plan(features, features)
        
        # 验证生成的计划
        assert hasattr(plan, 'steps')
        assert hasattr(plan, 'rules')
        assert hasattr(plan, 'metadata')
        
        # 验证计划包含必要的步骤
        assert len(plan.steps) > 0
        assert any('analyze' in step.lower() for step in plan.steps)
        
    def test_adaptive_plan_generator(self, sample_features):
        """测试自适应计划生成器"""
        features, _ = sample_features
        generator = AdaptivePlanGenerator()
        plan = generator.generate_plan(features, features)
        
        # 验证自适应特性
        assert len(plan.steps) > 0
        assert any('adaptive' in step.lower() for step in plan.steps)
        
    def test_similarity_analyzer(self, sample_features):
        """测试相似度分析器"""
        features, _ = sample_features
        analyzer = SimilarityAnalyzer()
        similarities = analyzer.compute_similarities(
            features, features,
            name_weight=0.3,
            semantic_weight=0.4,
            structural_weight=0.3
        )
        
        # 验证相似度计算结果
        assert 'name' in similarities
        assert 'structural' in similarities
        assert 'combined' in similarities

# 把模型测试暂时跳过
# @pytest.mark.skip(reason="Model implementation not ready")
class TestModelInference:
    """测试模型推理"""
    
    @pytest.fixture
    def sample_prompt(self):
        return "Test prompt for schema matching"
    
    @patch('requests.post')
    def test_ollama_model(self, mock_post, sample_prompt):
        """测试Ollama模型"""
        # 设置mock响应
        mock_response = Mock()
        mock_response.status_code = 200
        
        # 模拟流式响应数据
        response_lines = [
            '{"model":"llama3.1","response":"Generated ","done":false}',
            '{"model":"llama3.1","response":"schema ","done":false}',
            '{"model":"llama3.1","response":"matching ","done":false}',
            '{"model":"llama3.1","response":"response","done":true}'
        ]
        
        # 设置iter_lines方法的返回值
        mock_response.iter_lines.return_value = response_lines
        
        mock_post.return_value = mock_response
        
        # 初始化模型
        model = OllamaModel()
        
        # 测试生成
        response = model.generate(sample_prompt)
        
        # 验证结果
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == "Generated schema matching response"
        
        # 验证API调用
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert 'json' in call_args
        assert call_args['json']['model'] == 'llama3.1'
        assert call_args['json']['prompt'] == sample_prompt
        assert call_args['stream'] == True
    
    @pytest.mark.integration
    def test_ollama_model_integration(self, sample_prompt):
        """集成测试Ollama模型（需要运行中的Ollama服务）"""
        if not self._is_ollama_available():
            pytest.skip("Ollama service not available")
            
        model = OllamaModel()
        response = model.generate(sample_prompt)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def _is_ollama_available(self):
        """检查Ollama服务是否可用"""
        try:
            response = requests.get("http://localhost:11434/api/version")
            return response.status_code == 200
        except:
            return False
        
class TestResultValidator:
    """测试结果验证器"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试用的匹配结果和数据"""
        matches = [
            ('user_id', 'id'),
            ('username', 'name'),
            ('age', 'user_age')
        ]
        
        source_df = pd.DataFrame({
            'user_id': range(1, 11),
            'username': [f'user_{i}' for i in range(1, 11)],
            'age': np.random.randint(18, 80, 10)
        })
        
        target_df = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'user_{i}' for i in range(1, 11)],
            'user_age': source_df['age'].values
        })
        
        ground_truth = [
            ('user_id', 'id'),
            ('username', 'name'),
            ('age', 'user_age')
        ]
        
        return {
            'matches': matches,
            'source_df': source_df,
            'target_df': target_df,
            'ground_truth': ground_truth
        }
    
    def test_base_validator(self, sample_data):
        """测试基础验证器"""
        validator = BaseValidator()
        metrics = validator.validate(
            matches=sample_data['matches'],
            source_table=sample_data['source_df'],
            target_table=sample_data['target_df'],
            ground_truth=sample_data['ground_truth']
        )
        
        # 验证基础指标
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # 验证指标的合理性
        assert all(0 <= v <= 1 for v in metrics.values())
        
    def test_comprehensive_validator(self, sample_data):
        """测试全面验证器"""
        validator = ComprehensiveValidator()
        metrics = validator.validate(
            matches=sample_data['matches'],
            source_table=sample_data['source_df'],
            target_table=sample_data['target_df'],
            ground_truth=sample_data['ground_truth']
        )
        
        # 验证额外的指标
        assert 'type_compatibility_rate' in metrics
        assert 'avg_value_similarity' in metrics
        
        # 验证所有指标的合理性
        assert all(0 <= v <= 1 for v in metrics.values())