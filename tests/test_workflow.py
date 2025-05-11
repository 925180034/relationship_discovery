# tests/test_workflow.py

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
from utils.data import DataLoader, DataSampler
from features.extractors import ComprehensiveFeatureExtractor
from planning.plan_generator import BasePlanGenerator
from models.ollama_model import OllamaModel
from validation.validators import BaseValidator

class TestWorkflowIntegration:
    """Schema matching workflow integration tests"""
    
    @pytest.fixture
    def sample_tables(self):
        """Create test tables fixture"""
        # Source table
        source_df = pd.DataFrame({
            'user_id': range(1, 11),
            'username': [f'user_{i}' for i in range(1, 11)],
            'age': np.random.randint(18, 80, 10),
            'email': [f'user_{i}@example.com' for i in range(1, 11)]
        })
        
        # Target table
        target_df = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'user_{i}' for i in range(1, 11)],
            'user_age': source_df['age'].values,
            'contact': [f'user_{i}@example.com' for i in range(1, 11)]
        })
        
        return source_df, target_df
        
    @pytest.fixture
    def mock_components(self, sample_tables):
        """Create mocked workflow components with realistic data"""
        source_df, target_df = sample_tables
        
        # Mock feature extractor
        feature_extractor = Mock(spec=FeatureExtractor)
        feature_extractor.extract_features.return_value = TableFeatures(
            text_features={'column_names': list(source_df.columns)},
            structural_features={'dtypes': source_df.dtypes.to_dict()}
        )
        
        # Mock plan generator with similarity scores
        plan_generator = Mock(spec=PlanGenerator)
        similarity_matrix = np.random.rand(len(source_df.columns), len(target_df.columns))
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
        
        # Mock prompt builder
        prompt_builder = Mock(spec=PromptBuilder)
        prompt_builder.build_prompt.return_value = "Test prompt"
        
        # Mock model
        model = Mock(spec=ModelInference)
        model.generate.return_value = "Yes"
        
        # Mock validator
        validator = Mock(spec=ResultValidator)
        validator.validate.return_value = {
            'precision': 0.8,
            'recall': 0.7,
            'f1_score': 0.75
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
        """Create test workflow configuration"""
        return {
            'experiment': {
                'scenario': 'metadata',
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

    def test_metadata_scenario(self, sample_tables, mock_components, workflow_config):
        """Test metadata-only matching scenario"""
        source_df, target_df = sample_tables
        workflow = SchemaMatchingWorkflow(
            feature_extractor=mock_components['feature_extractor'],
            plan_generator=mock_components['plan_generator'],
            prompt_builder=mock_components['prompt_builder'],
            model=mock_components['model'],
            validator=mock_components['validator'],
            config=workflow_config
        )
        
        result = workflow.execute(source_df, target_df)
        
        # Verify workflow execution
        assert 'matches' in result
        assert 'metrics' in result
        assert 'plan' in result
        assert 'features' in result
        
        # Verify component interactions
        mock_components['feature_extractor'].extract_features.assert_called()
        mock_components['plan_generator'].generate_plan.assert_called()
        mock_components['model'].generate.assert_called()
        mock_components['validator'].validate.assert_called()

    def test_few_instances_scenario(self, sample_tables, mock_components, workflow_config):
        """Test few instances matching scenario"""
        source_df, target_df = sample_tables
        workflow_config['experiment']['scenario'] = 'few_instances'
        workflow_config['data'] = {'sample_size': 3}
        
        workflow = SchemaMatchingWorkflow(
            feature_extractor=mock_components['feature_extractor'],
            plan_generator=mock_components['plan_generator'],
            prompt_builder=mock_components['prompt_builder'],
            model=mock_components['model'],
            validator=mock_components['validator'],
            config=workflow_config
        )
        
        result = workflow.execute(source_df, target_df)
        
        # Verify sample data handling
        mock_components['prompt_builder'].build_prompt.assert_called()
        args = mock_components['prompt_builder'].build_prompt.call_args[0]
        assert len(args) >= 3  # Should include sample data

    def test_rich_instances_scenario(self, sample_tables, mock_components, workflow_config):
        """Test rich instances matching scenario"""
        source_df, target_df = sample_tables
        workflow_config['experiment']['scenario'] = 'rich_instances'
        
        workflow = SchemaMatchingWorkflow(
            feature_extractor=mock_components['feature_extractor'],
            plan_generator=mock_components['plan_generator'],
            prompt_builder=mock_components['prompt_builder'],
            model=mock_components['model'],
            validator=mock_components['validator'],
            config=workflow_config
        )
        
        result = workflow.execute(source_df, target_df)
        
        # Verify similarity score handling
        mock_components['prompt_builder'].build_prompt.assert_called()
        args = mock_components['prompt_builder'].build_prompt.call_args[0]
        assert len(args) >= 3  # Should include similarity score

    def test_error_handling(self, mock_components, workflow_config):
        """Test error handling scenarios"""
        workflow_config['experiment']['scenario'] = 'metadata'  # Start with valid scenario
        
        workflow = SchemaMatchingWorkflow(
            feature_extractor=mock_components['feature_extractor'],
            plan_generator=mock_components['plan_generator'],
            prompt_builder=mock_components['prompt_builder'],
            model=mock_components['model'],
            validator=mock_components['validator'],
            config=workflow_config
        )
        
        # Test invalid scenario
        workflow_config['experiment']['scenario'] = 'invalid'
        with pytest.raises(ValueError, match="Unknown scenario: invalid"):
            workflow.execute(pd.DataFrame(), pd.DataFrame())
        
        # Test feature extraction failure
        workflow_config['experiment']['scenario'] = 'metadata'
        mock_components['feature_extractor'].extract_features.side_effect = Exception("Feature extraction failed")
        with pytest.raises(Exception, match="Feature extraction failed"):
            workflow.execute(pd.DataFrame(), pd.DataFrame())
            
        # Reset mock
        mock_components['feature_extractor'].extract_features.side_effect = None
        
        # Test model failure
        mock_components['model'].generate.side_effect = Exception("Model inference failed")
        with pytest.raises(Exception, match="Model inference failed"):
            workflow.execute(pd.DataFrame(), pd.DataFrame())

    @pytest.mark.integration
    def test_end_to_end_workflow(self, tmp_path):
        """Test complete end-to-end workflow with real components"""
        # Create test files
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
        
        source_path = tmp_path / "source.csv"
        target_path = tmp_path / "target.csv"
        source_df.to_csv(source_path, index=False)
        target_df.to_csv(target_path, index=False)
        
        # Create ground truth mapping
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
            
        # Initialize real components
        feature_extractor = ComprehensiveFeatureExtractor()
        plan_generator = BasePlanGenerator()
        model = OllamaModel(model_name="llama2", base_url="http://localhost:11434")
        validator = BaseValidator()
        
        # Create workflow configuration
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
        
        # Create workflow
        workflow = SchemaMatchingWorkflow(
            feature_extractor=feature_extractor,
            plan_generator=plan_generator,
            prompt_builder=PromptBuilder(),
            model=model,
            validator=validator,
            config=config
        )
        
        try:
            # Execute workflow
            result = workflow.execute(source_df, target_df)
            
            # Verify results
            assert isinstance(result, dict)
            assert 'matches' in result
            assert 'metrics' in result
            assert len(result['matches']) > 0
            assert result['metrics']['precision'] >= 0
            assert result['metrics']['recall'] >= 0
        finally:
            # Clean up
            source_path.unlink()
            target_path.unlink()
            mapping_path.unlink()