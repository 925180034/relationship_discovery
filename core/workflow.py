# schema_matching/core/workflow.py

import json
from typing import Dict, Any, List, Optional, Tuple
import logging
from experiments.experiment_runner import ExperimentConfig
import numpy as np
import pandas as pd
from pathlib import Path
from core.config import ExperimentConfig 
from core.interfaces import (
    FeatureExtractor, PlanGenerator, 
    PromptBuilder, ModelInference, ResultValidator,
    TableFeatures, MatchingPlan
)
from utils.data import DataLoader, DataSampler

logger = logging.getLogger(__name__)

VALID_SCENARIOS = {'metadata', 'few_instances', 'rich_instances'}

class SchemaMatchingWorkflow:
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 plan_generator: PlanGenerator,
                 prompt_builder: PromptBuilder,
                 model: ModelInference,
                 validator: ResultValidator,
                 config: Dict[str, Any]):  # 改为接收字典类型
        self.feature_extractor = feature_extractor
        self.plan_generator = plan_generator
        self.prompt_builder = prompt_builder
        self.model = model
        self.validator = validator
        self.config = config
        
    @property
    def scenario(self) -> str:
        """获取当前场景"""
        return self.config['experiment']['scenario']  # 修改访问方式
        
    def _validate_scenario(self):
        """验证当前场景的有效性"""
        if self.scenario not in VALID_SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")
            
    def execute(self, source_table: pd.DataFrame, target_table: pd.DataFrame,
            ground_truth: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        try:
            # 特征提取
            logger.info("开始特征提取...")
            source_features = self.feature_extractor.extract_features(source_table)
            target_features = self.feature_extractor.extract_features(target_table)
            logger.info("特征提取完成")
            
            # 生成匹配计划
            logger.info("生成匹配计划...")
            plan = self.plan_generator.generate_plan(source_features, target_features)
            logger.info("匹配计划生成完成")
            
            # 确保从配置中正确读取场景
            scenario = self.config.get('experiment', {}).get('scenario') 
            if not scenario:
                scenario = 'metadata'  # 设置默认值
                
            logger.info(f"使用场景: {scenario}")
            
            # 根据场景执行匹配
            logger.info(f"开始执行{scenario}场景匹配...")
            matches = []
            if scenario == 'metadata':
                logger.info("执行基于元数据的匹配...")
                matches = self._execute_metadata_matching(
                    source_features, target_features, plan)
            elif scenario == 'few_instances':
                matches = self._execute_few_instances_matching(
                    source_features, target_features, plan,
                    source_table, target_table) # 传入源表和目标表
            elif scenario == 'rich_instances':
                matches = self._execute_rich_instances_matching(
                    source_features, target_features, plan,
                    source_table, target_table) # 传入源表和目标表
            else:
                raise ValueError(f"未知场景: {scenario}")
                
            logger.info("匹配完成,开始验证结果...")
            metrics = self.validator.validate(matches, source_table, target_table, ground_truth)
            logger.info("验证完成")
            
            logger.info("保存结果...")
            # 创建计划
            result = {
                'matches': matches,
                'metrics': metrics,
                'plan': plan.to_dict() if hasattr(plan, 'to_dict') else plan,
                'features': {
                    'source': source_features.to_dict() if hasattr(source_features, 'to_dict') else source_features,
                    'target': target_features.to_dict() if hasattr(target_features, 'to_dict') else target_features
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            raise
                
    def _validate_features(self, features: TableFeatures, table_name: str):
        """验证特征提取结果的有效性"""
        if not features.text_features or 'column_names' not in features.text_features:
            raise ValueError(f"Missing column names in {table_name} features")
        if not features.structural_features or 'dtypes' not in features.structural_features:
            raise ValueError(f"Missing data types in {table_name} features")
            
    def _execute_metadata_matching(self,
                            source_features: TableFeatures,
                            target_features: TableFeatures,
                            plan: MatchingPlan) -> List[Tuple[str, str]]:
        """优化后的元数据匹配执行"""
        matches = []
        logger.info("开始基于元数据的匹配...")
        
        try:
            source_cols = source_features.text_features['column_names']
            target_cols = target_features.text_features['column_names']
            
            # 获取相似度分数
            logger.info("获取相似度分数...")
            similarity_scores = plan.metadata.get('similarity_scores', {})
            combined_scores = similarity_scores.get('combined', [])
            
            if not combined_scores:
                logger.error("未找到相似度分数")
                return matches
                
            # 获取阈值
            threshold = plan.rules.get('thresholds', {}).get('combined', 0.2)
            
            # 预处理 - 找出所有超过阈值的候选对
            candidates = []
            for i, source_col in enumerate(source_cols):
                scores = combined_scores[i]
                for j, target_col in enumerate(target_cols):
                    if j < len(scores):
                        similarity = float(scores[j])
                        if similarity >= threshold:
                            candidates.append((source_col, target_col, similarity))
            
            # 按相似度降序排序
            candidates.sort(key=lambda x: x[2], reverse=True)
            logger.info(f"找到 {len(candidates)} 个潜在匹配")
            
            # 跟踪已匹配的列
            used_source = set()
            used_target = set()
            
            # 只对高相似度的候选对调用模型
            for source_col, target_col, similarity in candidates:
                if source_col in used_source or target_col in used_target:
                    continue
                    
                try:
                    # 构建prompt
                    prompt = self.prompt_builder.build_prompt(
                        source_features,
                        target_features,
                        (source_col, target_col),
                        similarity_score=similarity
                    )
                    
                    # 模型推理
                    response = self.model.generate(
                        prompt,
                        temperature=0.35,
                        top_p=0.9,
                        top_k=10
                    )
                    
                    # 如果模型确认匹配
                    if self._parse_response(response):
                        matches.append((source_col, target_col))
                        used_source.add(source_col)
                        used_target.add(target_col)
                        logger.info(f"{source_col} -> {target_col}")
                        
                except Exception as e:
                    logger.warning(f"处理列对 {source_col}-{target_col} 出错: {str(e)}")
                    continue
            
            return matches
            
        except Exception as e:
            logger.error(f"元数据匹配失败: {str(e)}")
            return matches

    def _execute_few_instances_matching(self,
                                    source_features: TableFeatures,
                                    target_features: TableFeatures,
                                    plan: MatchingPlan,
                                    source_table: pd.DataFrame,
                                    target_table: pd.DataFrame) -> List[Tuple[str, str]]:
        """执行基于少量样本的匹配"""
        matches = []
        
        try:
            # 获取采样数据
            logger.info("开始获取少量样本数据...")
            sample_size = 5
            samples = DataSampler.sample_tables(source_table, target_table, sample_size)
            
            # 验证samples的结构
            if not isinstance(samples, dict) or 'source' not in samples or 'target' not in samples:
                logger.error(f"samples格式错误: {samples}")
                return matches

            if not isinstance(samples['source'], dict) or not isinstance(samples['target'], dict):
                logger.error(f"samples内部格式错误: source={type(samples['source'])}, target={type(samples['target'])}")
                return matches

            logger.info(f"成功获取样本数据: source列数={len(samples['source'])}, target列数={len(samples['target'])}")

            # 获取相似度分数
            similarity_scores = plan.metadata.get('similarity_scores', {})
            combined_scores = similarity_scores.get('combined', [])
            
            source_cols = source_features.text_features['column_names']
            target_cols = target_features.text_features['column_names']
            
            # 处理每对列
            for i, src_col in enumerate(source_cols):
                for j, tgt_col in enumerate(target_cols):
                    try:
                        similarity = float(combined_scores[i][j])
                        if similarity >= 0.2:
                            # 安全获取样本数据
                            source_samples = []
                            target_samples = []
                            
                            if src_col in samples['source']:
                                source_samples = samples['source'][src_col]
                            if tgt_col in samples['target']:
                                target_samples = samples['target'][tgt_col]
                                
                            # 构建样本数据字典
                            column_samples = {
                                'source': source_samples,
                                'target': target_samples
                            }
                            
                            # 记录样本数据信息
                            logger.debug(f"列对 {src_col}-{tgt_col} 样本: "
                                    f"source={len(source_samples)}, "
                                    f"target={len(target_samples)}")
                            
                            prompt = self.prompt_builder.build_prompt(
                                source_features,
                                target_features,
                                (src_col, tgt_col),
                                sample_data=column_samples,
                                similarity_score=similarity
                            )
                            
                            response = self.model.generate(
                                prompt,
                                temperature=0.7,
                                top_p=0.9,
                                top_k=40
                            )
                            
                            if self._parse_response(response):
                                logger.info(f"✓ 确认匹配: {src_col} -> {tgt_col}")
                                matches.append((src_col, tgt_col))
                            else:
                                logger.info(f"✗ 拒绝匹配: {src_col} -> {tgt_col}")
                                
                    except Exception as e:
                        logger.warning(f"处理列对 {src_col}-{tgt_col} 失败: {repr(e)}")
                        continue
                        
            return matches
                
        except Exception as e:
            logger.error(f"Few instances matching失败: {repr(e)}")
            raise

    def _execute_rich_instances_matching(self,
                                        source_features: TableFeatures,
                                        target_features: TableFeatures,
                                        plan: MatchingPlan,
                                        source_table: pd.DataFrame,
                                        target_table: pd.DataFrame) -> List[Tuple[str, str]]:
        """执行基于丰富数据的匹配"""
        matches = []
        logger.info("开始丰富数据场景匹配...")

        try:
            # 获取采样和统计数据
            sample_size = 50  # rich instances场景使用更多样本
            samples = DataSampler.sample_tables(
                source_table, target_table, sample_size
            )
            
            # 计算统计信息
            source_stats = {}
            target_stats = {}
            for col in source_table.columns:
                source_stats[col] = self._compute_column_stats(source_table[col])
            for col in target_table.columns:
                target_stats[col] = self._compute_column_stats(target_table[col])

            stats = {
                'source': source_stats,
                'target': target_stats
            }

            similarity_scores = plan.metadata.get('similarity_scores', {})
            combined_scores = similarity_scores.get('combined', [])
            
            source_cols = source_features.text_features['column_names']
            target_cols = target_features.text_features['column_names']
            
            for i, src_col in enumerate(source_cols):
                for j, tgt_col in enumerate(target_cols):
                    try:
                        # 获取相似度
                        similarity = float(combined_scores[i][j])
                        if similarity >= 0.1:  # rich instances场景使用最低阈值
                            # 构建包含完整信息的prompt
                            prompt = self.prompt_builder.build_prompt(
                                source_features,
                                target_features,
                                (src_col, tgt_col),
                                sample_data=samples,  
                                stats=stats,
                                similarity_score=similarity
                            )
                            
                            # 模型推理
                            response = self.model.generate(
                                prompt,
                                temperature=0.7,
                                top_p=0.9,
                                top_k=40
                            )
                            
                            if self._parse_response(response):
                                logger.info(f"✓ 确认匹配: {src_col} -> {tgt_col}")
                                matches.append((src_col, tgt_col))
                            else:
                                logger.info(f"✗ 拒绝匹配: {src_col} -> {tgt_col}")
                                
                    except Exception as e:
                        logger.warning(f"处理匹配失败 {src_col}-{tgt_col}: {str(e)}")
                        continue
                        
            return matches
            
        except Exception as e:
            logger.error(f"Rich instances matching失败: {str(e)}")
            raise

    def _compute_column_stats(self, column: pd.Series) -> Dict[str, Any]:
        """计算列的统计信息"""
        stats = {}
        try:
            # 基础统计
            stats['null_count'] = column.isnull().sum()
            stats['unique_count'] = column.nunique()
            
            # 值分布
            value_counts = column.value_counts()
            stats['value_counts'] = {
                str(k): int(v) for k, v in value_counts.head().items()
            }
            
            # 数值型列的统计
            if pd.api.types.is_numeric_dtype(column):
                desc = column.describe()
                for k, v in desc.items():
                    if not pd.isna(v):
                        stats[k] = float(v)
                        
            # 字符串列的统计
            elif pd.api.types.is_string_dtype(column):
                length_stats = column.str.len().describe()
                stats['length_stats'] = {
                    k: float(v) for k, v in length_stats.items()
                    if not pd.isna(v)
                }
                
        except Exception as e:
            logger.warning(f"计算统计信息失败: {str(e)}")
            
        return stats 

    def _parse_response(self, response: str) -> bool:
        """解析模型响应"""
        # 清理响应文本
        response = response.strip().lower()
        
        # 检查常见的肯定回答模式
        positive_patterns = ['yes', 'true', 'equivalent', 'match', 'similar']
        for pattern in positive_patterns:
            if pattern in response:
                return True
                
        # 获取最后一行作为决策
        last_line = response.split('\n')[-1].strip()
        return last_line in positive_patterns
    
    def _save_result(self, result: Dict):
        """保存实验结果"""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存详细匹配结果
            result_file = output_dir / f"{self.config.experiment_name}_results.json"
            detailed_results = {
                'matches': result['matches'],
                'metrics': result['metrics'],
                'matching_details': {
                    'source_columns': list(self.source_df.columns),
                    'target_columns': list(self.target_df.columns),
                    'matched_pairs': [
                        {
                            'source': src,
                            'target': tgt,
                            'source_type': str(self.source_df[src].dtype),
                            'target_type': str(self.target_df[tgt].dtype)
                        }
                        for src, tgt in result['matches']
                    ]
                }
            }
            
            with open(result_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
                
            # 创建人类可读的报告
            report_file = output_dir / f"{self.config.experiment_name}_report.txt"
            with open(report_file, 'w') as f:
                f.write(f"Schema Matching 实验报告\n")
                f.write(f"实验名称: {self.config.experiment_name}\n")
                f.write(f"场景: {self.config.scenario}\n\n")
                
                f.write("匹配结果:\n")
                for src, tgt in result['matches']:
                    f.write(f"{src} -> {tgt}\n")
                
                f.write("\n评估指标:\n")
                for metric, value in result['metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                    
            logger.info(f"结果已保存到: {output_dir}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            raise