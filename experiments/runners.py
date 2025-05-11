# schema_matching/experiments/runners.py
import importlib
from typing import Dict, Any, List, Optional, Type
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dataclasses import dataclass
import yaml
import logging
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from core.interfaces import (
    FeatureExtractor, PlanGenerator, 
    PromptBuilder, ModelInference, ResultValidator
)

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for schema matching experiment"""
    experiment_name: str
    data_path: str
    output_path: str
    feature_extractor_config: Dict[str, Any]
    plan_generator_config: Dict[str, Any]
    prompt_builder_config: Dict[str, Any]
    model_config: Dict[str, Any]
    validator_config: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
class ComponentFactory:
    """Factory class for creating component instances"""
    
    @staticmethod
    def create_instance(component_type: str, config: Dict[str, Any], base_class: Type):
        """Create component instance based on type and config"""
        try:
            # Import the correct module based on component type
            module_path = f"schema_matching.{base_class.__module__.split('.')[0]}"
            module = importlib.import_module(module_path)
            
            # Get the class
            class_name = f"{component_type.capitalize()}{base_class.__name__}"
            component_class = getattr(module, class_name)
            
            # Create instance
            return component_class(**config)
        except Exception as e:
            logger.error(f"Failed to create {base_class.__name__} instance: {str(e)}")
            raise

class ExperimentRunner:
    """Main class for running schema matching experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_logging()
        self.factory = ComponentFactory()
        
        # Initialize components
        self.feature_extractor = self._init_feature_extractor()
        self.plan_generator = self._init_plan_generator()
        self.prompt_builder = self._init_prompt_builder()
        self.model = self._init_model()
        self.validator = self._init_validator()
    
    def _init_feature_extractor(self) -> FeatureExtractor:
        """Initialize feature extractor based on config"""
        return self.factory.create_instance(
            self.config.feature_extractor_config['type'],
            self.config.feature_extractor_config,
            FeatureExtractor
        )
    
    def _init_plan_generator(self) -> PlanGenerator:
        """Initialize plan generator based on config"""
        return self.factory.create_instance(
            self.config.plan_generator_config['type'],
            self.config.plan_generator_config,
            PlanGenerator
        )
    
    def _init_prompt_builder(self) -> PromptBuilder:
        """Initialize prompt builder based on config"""
        return self.factory.create_instance(
            self.config.prompt_builder_config['type'],
            self.config.prompt_builder_config,
            PromptBuilder
        )
    
    def _init_model(self) -> ModelInference:
        """Initialize model based on config"""
        return self.factory.create_instance(
            self.config.model_config['type'],
            self.config.model_config,
            ModelInference
        )
    
    def _init_validator(self) -> ResultValidator:
        """Initialize validator based on config"""
        return self.factory.create_instance(
            self.config.validator_config['type'],
            self.config.validator_config,
            ResultValidator
        )

    def load_data(self, data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load source and target tables from paths"""
        try:
            data = {}
            for key, path in data_paths.items():
                data[key] = pd.read_csv(path)
                logger.info(f"Loaded {key} table with shape {data[key].shape}")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results"""
        analysis = {
            'summary_metrics': {},
            'detailed_analysis': {},
            'visualizations': {}
        }
        
        # Calculate summary metrics
        for metric, value in results['metrics'].items():
            if isinstance(value, (int, float)):
                analysis['summary_metrics'][metric] = value
        
        # Analyze matches
        matches = results['matches']
        analysis['detailed_analysis']['match_count'] = len(matches)
        analysis['detailed_analysis']['confidence_stats'] = {
            'mean': np.mean([m['confidence'] for m in matches]),
            'std': np.std([m['confidence'] for m in matches])
        }
        
        # Generate visualizations
        self._generate_visualizations(results, analysis['visualizations'])
        
        return analysis
    
    def _generate_visualizations(self, results: Dict[str, Any], 
                               viz_dict: Dict[str, plt.Figure]):
        """Generate visualization figures"""
        # Confidence distribution
        fig_conf = plt.figure(figsize=(10, 6))
        confidence_scores = [m['confidence'] for m in results['matches']]
        sns.histplot(confidence_scores, bins=20)
        plt.title('Distribution of Match Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        viz_dict['confidence_dist'] = fig_conf
        
        # Metrics comparison
        fig_metrics = plt.figure(figsize=(10, 6))
        metrics = results['metrics']
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Evaluation Metrics')
        plt.xticks(rotation=45)
        viz_dict['metrics_comparison'] = fig_metrics
    
    def generate_report(self, results: Dict[str, Any], 
                       output_path: Optional[str] = None) -> str:
        """Generate detailed experiment report"""
        if output_path is None:
            output_path = Path(self.config.output_path) / 'reports'
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Create report content
        report = [
            f"# Schema Matching Experiment Report\n",
            f"## Experiment Details",
            f"- Name: {self.config.experiment_name}",
            f"- Timestamp: {results['timestamp']}",
            f"\n## Summary Metrics",
        ]
        
        # Add metrics
        for metric, value in analysis['summary_metrics'].items():
            report.append(f"- {metric}: {value:.4f}")
        
        # Add detailed analysis
        report.extend([
            f"\n## Detailed Analysis",
            f"- Total Matches: {analysis['detailed_analysis']['match_count']}",
            f"- Average Confidence: {analysis['detailed_analysis']['confidence_stats']['mean']:.4f}",
            f"- Confidence Std: {analysis['detailed_analysis']['confidence_stats']['std']:.4f}",
        ])
        
        # Add matches
        report.extend([
            f"\n## Matches",
            "| Source Column | Target Column | Confidence | Reasoning |",
            "|---------------|---------------|------------|-----------|"
        ])
        
        for match in results['matches']:
            report.append(
                f"| {match['source_column']} | {match['target_column']} | "
                f"{match['confidence']:.4f} | {match['reasoning']} |"
            )
        
        # Save visualizations
        viz_path = Path(output_path) / 'figures'
        viz_path.mkdir(exist_ok=True)
        
        for name, fig in analysis['visualizations'].items():
            fig_path = viz_path / f"{name}.png"
            fig.savefig(fig_path)
            report.append(f"\n![{name}]({fig_path})")
        
        # Save report
        report_content = "\n".join(report)
        report_path = Path(output_path) / f"report_{self.config.experiment_name}.md"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        return report_path