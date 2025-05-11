# core/interfaces.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class TableFeatures:
    """Data class for extracted table features"""
    text_features: Dict[str, Any]
    structural_features: Dict[str, Any]
    semantic_features: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'text_features': self.text_features,
            'structural_features': self.structural_features,
            'semantic_features': self.semantic_features
        }
    
@dataclass
class MatchingPlan:
    """匹配计划数据类"""
    steps: List[str]
    rules: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'steps': self.steps,
            'rules': self.rules,
            'metadata': self.metadata
        }
    
@dataclass
class MatchingResult:
    """Data class for matching results"""
    matches: List[tuple]  # List of matched column pairs
    confidence_scores: Dict[tuple, float]  # Confidence score for each match
    metadata: Dict[str, Any]  # Additional result metadata

class FeatureExtractor(ABC):
    """Interface for table feature extraction"""
    
    @abstractmethod
    def extract_features(self, table: pd.DataFrame) -> TableFeatures:
        """Extract features from a table
        
        Args:
            table: Input DataFrame to extract features from
            
        Returns:
            TableFeatures object containing extracted features
        """
        pass

class PlanGenerator(ABC):
    """Interface for matching plan generation"""
    
    @abstractmethod
    def generate_plan(self, source_features: TableFeatures, 
                     target_features: TableFeatures) -> MatchingPlan:
        """Generate matching plan based on table features
        
        Args:
            source_features: Features extracted from source table
            target_features: Features extracted from target table
            
        Returns:
            MatchingPlan object containing matching steps and rules
        """
        pass

class PromptBuilder(ABC):
    """Interface for prompt construction"""
    
    @abstractmethod
    def build_prompt(self, 
                    source_features: TableFeatures,
                    target_features: TableFeatures,
                    plan: MatchingPlan) -> str:
        """Build model prompt based on features and plan
        
        Args:
            source_features: Features from source table
            target_features: Features from target table
            plan: Generated matching plan
            
        Returns:
            Constructed prompt string
        """
        pass

class ModelInference(ABC):
    """Interface for model inference"""
    
    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        """Initialize model with given configuration"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate model response for given prompt
        
        Args:
            prompt: Input prompt string
            **kwargs: Additional generation parameters
            
        Returns:
            Generated model response
        """
        pass

class ResultValidator(ABC):
    """Interface for result validation"""
    
    @abstractmethod
    def validate(self, 
                matches: List[tuple],
                source_table: pd.DataFrame,
                target_table: pd.DataFrame,
                ground_truth: Optional[List[tuple]] = None) -> Dict[str, float]:
        """Validate matching results and compute metrics
        
        Args:
            matches: List of predicted column matches
            source_table: Source DataFrame
            target_table: Target DataFrame
            ground_truth: Optional ground truth matches for evaluation
            
        Returns:
            Dictionary of validation metrics
        """
        pass