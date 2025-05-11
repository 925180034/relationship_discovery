from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str
    source_path: str
    target_path: str
    ground_truth_path: str
    output_dir: str
    feature_config: Dict  
    model_config: Dict    
    scenario: str        
    metrics: List[str]