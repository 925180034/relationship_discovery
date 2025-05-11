# schema_matching/planning/plan_generator.py

import json
from typing import Dict, Any
import numpy as np
from core.interfaces import MatchingPlan, TableFeatures
from features.similarity import SimilarityAnalyzer
from .strategies import BaseMatchingStrategy, AdaptiveMatchingStrategy
import logging

logger = logging.getLogger(__name__)

class BasePlanGenerator:
    def __init__(self, config: Dict[str, Any]):
        """使用配置初始化生成器"""
        self.config = config
        
        # 获取当前场景
        self.scenario = config.get('experiment', {}).get('scenario', 'metadata')
        logger.info(f"初始化PlanGenerator，场景: {self.scenario}")
        
        # 获取全局默认配置
        default_thresholds = config.get('plan_generator', {}).get('thresholds', {})
        default_weights = config.get('plan_generator', {}).get('weights', {})
        
        # 获取场景特定配置
        scenarios_config = config.get('scenarios', {})  # 注意这里的路径修改
        scenario_config = scenarios_config.get(self.scenario, {})
        logger.debug(f"场景配置: {scenario_config}")
        
        # 设置阈值 - 先使用默认值，再由场景配置覆盖
        scenario_thresholds = scenario_config.get('thresholds', {})
        self.thresholds = {
            'name': scenario_thresholds.get('name', default_thresholds.get('name')),
            'structural': scenario_thresholds.get('structural', default_thresholds.get('structural')),
            'semantic': scenario_thresholds.get('semantic', default_thresholds.get('semantic')),
            'combined': scenario_thresholds.get('combined', default_thresholds.get('combined'))
        }
        
        # 设置权重
        scenario_weights = scenario_config.get('weights', {})
        self.weights = {
            'name': scenario_weights.get('name', default_weights.get('name')),
            'structural': scenario_weights.get('structural', default_weights.get('structural')),
            'semantic': scenario_weights.get('semantic', default_weights.get('semantic'))
        }
        
        # 记录实际使用的配置
        logger.info("使用的配置:")
        logger.info(f"阈值: {self.thresholds}")
        logger.info(f"权重: {self.weights}")
        
        self.similarity_analyzer = SimilarityAnalyzer()
        self.strategy = BaseMatchingStrategy(
            name_weight=self.weights['name'],
            semantic_weight=self.weights['semantic'],
            structural_weight=self.weights['structural'],
            threshold=self.thresholds['combined']
        )

    def generate_plan(self, source_features: TableFeatures,
                    target_features: TableFeatures) -> MatchingPlan:
        """生成匹配计划"""
        try:
            logger.info("开始生成匹配计划...")
            
            # 计算相似度
            logger.info("计算相似度...")
            similarity_scores = self.similarity_analyzer.compute_similarities(
                source_features, target_features,
                self.weights['name'],
                self.weights['semantic'], 
                self.weights['structural']
            )
            logger.info("相似度计算完成")
            
            # 生成步骤
            steps = self.strategy.generate_steps(similarity_scores)
            logger.info(f"生成了 {len(steps)} 个步骤")
            
            # 生成规则
            rules = {
                'thresholds': self.thresholds,
                'weights': self.weights
            }
            
            # 准备元数据
            metadata = {
                'similarity_scores': similarity_scores,
                'source_columns': source_features.text_features['column_names'],
                'target_columns': target_features.text_features['column_names']
            }
            
            # 创建计划
            plan = MatchingPlan(
                steps=steps,
                rules=rules,
                metadata=metadata
            )
            
            logger.info("匹配计划生成完成")
            logger.info(f"Plan包含:\n- {len(steps)}个步骤\n" + 
                      f"- Rules: {json.dumps(rules, indent=2)}\n" +
                      f"- Metadata包含 {len(metadata)} 项")
            
            return plan
            
        except Exception as e:
            logger.error(f"生成计划失败: {str(e)}")
            raise
    
class AdaptivePlanGenerator(BasePlanGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strategy = AdaptiveMatchingStrategy(
            name_weight=self.name_weight,
            semantic_weight=self.semantic_weight,
            structural_weight=self.structural_weight,
            threshold=self.threshold
        )