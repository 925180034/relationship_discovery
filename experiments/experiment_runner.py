# experiments/experiment_runner.py
from models.base import GenerationConfig
import logging
import json
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import copy
from core.config import ExperimentConfig
from core.workflow import SchemaMatchingWorkflow
from core.interfaces import TableFeatures, MatchingPlan 
from features.extractors import ComprehensiveFeatureExtractor 
from prompts.builders import MetadataPromptBuilder, FewInstancesPromptBuilder, RichInstancesPromptBuilder
from planning.plan_generator import BasePlanGenerator
from models.ollama_model import OllamaModel
from models.hf_model import HuggingFaceModel
from validation.validators import ComprehensiveValidator
from utils.data import DataLoader

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    experiment_name: str
    source_path: str
    target_path: str
    ground_truth_path: str
    output_dir: str
    feature_config: Dict[str, Any]  
    model_config: Dict[str, Any]    
    scenario: str        
    metrics: List[str]  

def dict_to_experiment_config(config: Dict[str, Any]) -> ExperimentConfig:
    """将配置字典转换为ExperimentConfig对象"""
    return ExperimentConfig(
        experiment_name=config.get('experiment', {}).get('name', 'default_experiment'),
        source_path=config.get('data', {}).get('source_table', ''),
        target_path=config.get('data', {}).get('target_table', ''),
        ground_truth_path=config.get('data', {}).get('ground_truth', ''),
        output_dir=config.get('data', {}).get('output_dir', 'experiments/results'),
        feature_config=config.get('feature_extractor', {}),
        model_config=config.get('model', {}),
        scenario=config.get('experiment', {}).get('scenario', 'metadata'),
        metrics=config.get('metrics', [])
    )

class ExperimentRunner:
    """实验运行器"""
    def __init__(self, config: Dict[str, Any]):
        """初始化实验运行器
        
        Args:
            config: 配置字典
        """
        try:
            # 处理配置
            self.config = self._process_config(config)
            logger.info("配置处理完成")
            
            # 设置日志
            self.setup_logging()
            logger.info("日志系统设置完成")
            
            # 加载数据
            self._load_data()
            logger.info("数据加载完成")
            
            # 初始化组件
            self.setup_components()
            logger.info("组件初始化完成")
            
        except Exception as e:
            logger.error(f"实验运行器初始化失败: {str(e)}")
            raise


    def setup_logging(self):
        """增强的日志设置"""
        log_dir = Path("experiments/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每种实验类型创建单独的日志文件
        self.loggers = {
            'feature': logging.getLogger('feature_ablation'),
            'scenario': logging.getLogger('scenario_comparison'), 
            'model': logging.getLogger('model_comparison')
        }
        
        for name, logger in self.loggers.items():
            handler = logging.FileHandler(f"experiments/logs/{name}_experiments.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理和验证配置"""
        processed_config = copy.deepcopy(config)
        
        # 确保基本结构存在
        processed_config.setdefault('experiment', {})
        processed_config.setdefault('data', {})
        processed_config.setdefault('feature_extractor', {})
        processed_config.setdefault('model', {})
        processed_config.setdefault('plan_generator', {})
        
        # 设置默认值
        if 'experiment' in processed_config:
            processed_config['experiment'].setdefault('name', 'default_experiment')
            processed_config['experiment'].setdefault('scenario', 'metadata')
            
        if 'model' in processed_config:
            processed_config['model'].setdefault('type', 'ollama')
            processed_config['model'].setdefault('name', 'llama3.1')
            processed_config['model'].setdefault('generation_config', {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_tokens': 2048
            })
            
        logger.info("配置处理完成")
        return processed_config

    def setup_components(self):
        """初始化所有组件"""
        try:
            # 1. 初始化特征提取器
            feature_config = self.config.get('feature_extractor', {})
            self.feature_extractor = ComprehensiveFeatureExtractor(**feature_config)
            logger.info("特征提取器初始化完成")
            
            # 2. 初始化计划生成器
            self.plan_generator = BasePlanGenerator(config=self.config)
            logger.info("计划生成器初始化完成")
            
            # 3. 初始化提示词构建器
            scenario = self.config.get('experiment', {}).get('scenario', 'metadata')
            if scenario == 'metadata':
                self.prompt_builder = MetadataPromptBuilder()
            elif scenario == 'few_instances':
                self.prompt_builder = FewInstancesPromptBuilder()
            else:
                self.prompt_builder = RichInstancesPromptBuilder()
            logger.info(f"提示词构建器初始化完成: {scenario}场景")
            
            # 4. 初始化模型
            model_config = self.config.get('model', {})
            model_type = model_config.get('type', 'ollama')
            
            # 处理生成配置
            generation_config_dict = model_config.get('generation_config', {})
            generation_config = GenerationConfig(**generation_config_dict)
            
            if model_type == 'ollama':
                self.model = OllamaModel(
                    model_name=model_config.get('name', 'llama3.1'),
                    base_url=model_config.get('base_url', 'http://localhost:11434'),
                    generation_config=generation_config
                )
            else:
                self.model = HuggingFaceModel(
                    model_name=model_config.get('name', 'bert-base-uncased'),
                    device=model_config.get('device', 'cpu'),
                    generation_config=generation_config
                )
            logger.info(f"模型初始化完成: {model_type}")

            # 5. 初始化验证器
            self.validator = ComprehensiveValidator()
            logger.info("验证器初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {str(e)}")
            raise

    def _save_ablation_result(self, feature_name: str, result: Dict):
        output_dir = Path(self.config.output_dir) / 'ablation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_path = output_dir / f"ablation_{feature_name}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
            
    def _save_scenario_result(self, scenario: str, result: Dict):
        output_dir = Path(self.config.output_dir) / 'scenarios'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_path = output_dir / f"scenario_{scenario}.json" 
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    def _load_data(self):
        """加载数据"""
        try:
            logger.info("加载数据...")
            
            # 从配置的data部分获取路径
            data_config = self.config.get('data', {})
            source_path = data_config.get('source_table')
            target_path = data_config.get('target_table')
            ground_truth_path = data_config.get('ground_truth')
            
            if not source_path or not target_path:
                raise ValueError("数据路径未在配置中指定")
            
            # 加载源表和目标表
            self.source_df = DataLoader.load_table(source_path)
            self.target_df = DataLoader.load_table(target_path)
            
            logger.info(f"已加载源表: {len(self.source_df)} 行, {len(self.source_df.columns)} 列")
            logger.info(f"源表列类型: {self.source_df.dtypes.to_dict()}")
            
            logger.info(f"已加载目标表: {len(self.target_df)} 行, {len(self.target_df.columns)} 列")
            logger.info(f"目标表列类型: {self.target_df.dtypes.to_dict()}")

            # 如果有ground truth，加载它
            self.ground_truth = None
            if ground_truth_path:
                self.ground_truth = DataLoader.load_ground_truth(ground_truth_path)
                logger.info(f"已加载 {len(self.ground_truth)} 个ground truth映射")
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
        
    def run_feature_ablation(self, scenario: str = None, features: List[Dict] = None):
        """运行特征消融实验"""
        logger = self.loggers['feature']
        logger.info(f"开始场景的特征消融: {scenario}")
        
        if features is None:
            # 默认的特征组合
            features = [
                {'text': True, 'structural': False, 'semantic': False},
                {'text': False, 'structural': True, 'semantic': False},
                {'text': False, 'structural': False, 'semantic': True},
                {'text': True, 'structural': True, 'semantic': False},
                {'text': True, 'structural': False, 'semantic': True},
                {'text': False, 'structural': True, 'semantic': True},
                {'text': True, 'structural': True, 'semantic': True}
            ]
                    
        results = {}
        for feature_config in features:
            # 生成特征名称
            feature_name = '_'.join(k for k, v in feature_config.items() if v)
            logger.info(f"测试特征组合: {feature_name}")
            
            try:
                # 更新此次运行的配置
                run_config = copy.deepcopy(self.config)
                if scenario:
                    run_config['experiment']['scenario'] = scenario
                
                # 更新特征提取器配置
                run_config['feature_extractor'].update(feature_config)
                
                # 创建新的实验运行器
                temp_runner = ExperimentRunner(run_config)
                
                # 运行单次实验
                result = temp_runner.run_single_experiment()
                results[feature_name] = result
                
                # 保存结果
                self._save_ablation_result(feature_name, result)
                logger.info(f"特征组合 {feature_name} 测试完成")
                    
            except Exception as e:
                logger.error(f"特征消融 {feature_name} 出错: {str(e)}")
                continue
                        
        return results

    def _save_ablation_result(self, feature_name: str, result: Dict):
        """保存消融实验结果"""
        try:
            output_dir = Path(self.config.output_dir) / 'ablation'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result_path = output_dir / f"ablation_{feature_name}.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
            logger.info(f"已保存 {feature_name} 的结果到 {result_path}")
        except Exception as e:
            logger.error(f"保存消融实验结果失败: {str(e)}")
        
    def run_scenario_comparison(self):
        """场景对比实验"""
        results = {}
        scenarios = ['metadata', 'few_instances', 'rich_instances']

        try:
            for scenario in scenarios:
                logger.info(f"Running experiment for {scenario} scenario")
                
                # 更新场景配置
                self.config.scenario = scenario
                
                # 更新 prompt_builder
                if scenario == 'metadata':
                    self.prompt_builder = MetadataPromptBuilder()
                elif scenario == 'few_instances':
                    self.prompt_builder = FewInstancesPromptBuilder()
                else:
                    self.prompt_builder = RichInstancesPromptBuilder()

                # 创建 workflow
                workflow = SchemaMatchingWorkflow(
                    feature_extractor=self.feature_extractor,
                    plan_generator=self.plan_generator,
                    prompt_builder=self.prompt_builder,
                    model=self.model,
                    validator=self.validator,
                    config={'experiment': {'scenario': scenario}}
                )

                # 执行实验
                result = workflow.execute(self.source_df, self.target_df)
                results[scenario] = result

                # 保存每个场景的结果
                self._save_scenario_result(scenario, result)

        except Exception as e:
            logger.error(f"Error in scenario comparison: {str(e)}")
            raise

        return results
            
    def run_model_comparison(self):
        """模型对比实验"""
        results = {}
        models = [
            {
                'type': 'ollama',
                'name': 'llama3.1',
                'temperature': 0.7
            },
            {
                'type': 'huggingface',
                'name': 'bert-base-uncased',
                'temperature': 0.7  
            }
        ]
        
        for model_config in models:
            self.config.model_config = model_config
            model_name = f"{model_config['type']}_{model_config['name']}"
            result = self._run_single_experiment()
            results[model_name] = result
            
        return results
        
    def run_validation_analysis(self):
        """验证方法实验"""
        results = {}
        
        try:
            # 完整ground truth
            result = self._run_single_experiment()
            results['full_ground_truth'] = result
            
            # 部分ground truth
            if self.ground_truth:
                halfway = len(self.ground_truth) // 2
                partial_ground_truth = self.ground_truth[:halfway]
                result = self._run_single_experiment(ground_truth=partial_ground_truth)
                results['partial_ground_truth'] = result
            
            # 无ground truth
            result = self._run_single_experiment(ground_truth=[])
            results['no_ground_truth'] = result
            
            return results
            
        except Exception as e:
            logger.error(f"Validation analysis failed: {str(e)}")
            raise
        
    def run_single_experiment(self) -> Dict[str, Any]:
        """运行单个实验"""
        try:
            experiment_name = self.config.get('experiment', {}).get('name', 'default_experiment')
            scenario = self.config.get('experiment', {}).get('scenario', 'metadata')
            
            logger.info(f"开始运行 {experiment_name} 实验...")
            logger.info(f"场景: {scenario}")
            
            # 验证所需组件
            required_attrs = ['feature_extractor', 'plan_generator', 
                            'prompt_builder', 'model', 'validator']
            for attr in required_attrs:
                if not hasattr(self, attr):
                    raise RuntimeError(f"缺少必要组件: {attr}")
            
            # 创建workflow
            workflow = SchemaMatchingWorkflow(
                feature_extractor=self.feature_extractor,
                plan_generator=self.plan_generator,
                prompt_builder=self.prompt_builder,
                model=self.model,
                validator=self.validator,
                config=self.config
            )
            
            # 执行匹配并获取结果
            result = workflow.execute(
                self.source_df,
                self.target_df,
                ground_truth=self.ground_truth
            )
            
            # 打印评估结果
            logger.info("评估结果:")
            for metric, value in result['metrics'].items():
                logger.info(f"{metric}: {value:.4f}")
                
            logger.info(f"找到 {len(result['matches'])} 个匹配:")
            for src, tgt in result['matches']:
                logger.info(f"{src} -> {tgt}")
            
            # 保存结果
            self._save_result(result)
            
            return result

        except Exception as e:
            logger.error(f"实验失败: {str(e)}")
            raise

            
    def _convert_features(self, features: TableFeatures) -> Dict[str, Any]:
        """转换TableFeatures对象为可序列化的字典格式"""
        if features is None:
            return None

        # 使用字典推导式处理可能的numpy数组
        def convert_value(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                return float(v)
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            return v

        # 转换每个特征类别
        return {
            'text_features': {
                k: convert_value(v) for k, v in features.text_features.items()
            } if features.text_features else None,
            'structural_features': {
                k: convert_value(v) for k, v in features.structural_features.items()
            } if features.structural_features else None,
            'semantic_features': {
                k: convert_value(v) for k, v in features.semantic_features.items()
            } if features.semantic_features else None
        }

    def _save_result(self, result: Dict[str, Any]):
        """保存实验结果"""
        try:
            # 从配置字典中获取输出目录
            output_dir = Path(self.config.get('data', {}).get('output_dir', 'experiments/results'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取实验名称
            experiment_name = self.config.get('experiment', {}).get('name', 'default_experiment')
            
            # 构建结果文件路径
            result_file = output_dir / f"{experiment_name}_results.json"
            
            # 保存结果
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            logger.info(f"结果已保存到: {result_file}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            raise


class BatchExperimentRunner:
    def __init__(self, base_dir: str, config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.config = config
        self.results = {}
        
        # 确保输出目录存在
        self.output_dir = Path(config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建individual_results目录
        self.individual_results_dir = self.output_dir / 'individual_results'
        self.individual_results_dir.mkdir(parents=True, exist_ok=True)

    def _validate_config(self):
        return True

    @staticmethod
    def convert_to_serializable(obj: Any) -> Any:
        """增强的序列化转换方法"""
        if obj is None:
            return None
            
        # 处理numpy数值类型    
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8, np.uint16,
            np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
            
        # 处理numpy数组    
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # 处理基本类型
        if isinstance(obj, (str, int, float, bool)):
            return obj
            
        # 处理pandas类型    
        if isinstance(obj, pd.core.dtypes.base.ExtensionDtype):
            return str(obj)
            
        # 处理可转换对象
        if hasattr(obj, 'to_dict'):
            result = obj.to_dict()
            return BatchExperimentRunner.convert_to_serializable(result)
            
        # 处理字典
        if isinstance(obj, dict):
            return {k: BatchExperimentRunner.convert_to_serializable(v) 
                for k, v in obj.items()}
                
        # 处理列表/元组    
        if isinstance(obj, (list, tuple)):
            return [BatchExperimentRunner.convert_to_serializable(item) 
                    for item in obj]
                    
        # 其他类型转字符串
        try:
            return str(obj)
        except:
            return None

    @staticmethod  # 注意装饰器位置在这里
    def convert_numpy_to_python(obj):
        """将numpy类型转换为Python原生类型"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: BatchExperimentRunner.convert_numpy_to_python(value) 
                    for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BatchExperimentRunner.convert_numpy_to_python(item) 
                    for item in obj]
        return obj

    
    def discover_experiments(self):
        """发现所有实验"""
        experiments = []
        
        # 修改为从data_dir/experiments目录读取配置
        experiment_dir = Path(self.base_dir) / 'experiments'
        logger.info(f"Searching for experiments in: {experiment_dir}")
        
        if not experiment_dir.exists():
            logger.error(f"实验目录不存在: {experiment_dir}")
            return experiments
            
        for experiment_file in os.listdir(experiment_dir):
            if experiment_file.endswith('.json'):
                file_path = experiment_dir / experiment_file
                try:
                    if file_path.stat().st_size == 0:
                        logger.warning(f"跳过空文件: {file_path}")
                        continue
                        
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logger.debug(f"Reading file {file_path}: {content[:200]}...")
                        
                        experiment_data = json.loads(content)
                        if self._validate_experiment_data(experiment_data):
                            experiments.append(experiment_data)
                        else:
                            logger.warning(f"实验数据格式无效: {file_path}")
                            
                except Exception as e:
                    logger.error(f"处理文件时出错 {file_path}: {str(e)}")
                    continue
        
        logger.info(f"发现 {len(experiments)} 个有效实验")
        return experiments

    def _validate_experiment_data(self, data):
        """验证实验数据格式"""
        required_fields = ['source', 'target', 'mapping', 'metadata']
        return all(field in data for field in required_fields)

    def format_result(self, result: Dict) -> Dict:
        """将结果转换为可JSON序列化的格式"""
        formatted = {}
        # 保存metrics
        formatted['metrics'] = result['metrics']
        # 保存matches
        formatted['matches'] = result['matches']
        # 复制plan字典(已经在workflow中转换)
        if 'plan' in result:
            formatted['plan'] = result['plan']
        return formatted

    def run_batch(self):
        """运行所有实验"""
        # 验证配置
        if not self._validate_config():
            logger.error("配置无效，终止批处理实验")
            return
            
        experiments = self.discover_experiments()
        if not experiments:
            logger.warning("未找到有效的实验配置，终止执行")
            return
            
        logger.info(f"开始执行 {len(experiments)} 个实验...")
        
        for exp in experiments:
            try:
                # 更新配置
                exp_config = copy.deepcopy(self.config)
                
                # 使用正确的路径拼接方式
                base_path = Path(self.base_dir)
                exp_config['data'].update({
                    'source_table': str(base_path / exp['source']),
                    'target_table': str(base_path / exp['target']),
                    'ground_truth': str(base_path / exp['mapping']),
                    'metadata': str(base_path / exp['metadata'])
                })
                
                # 运行单个实验
                runner = ExperimentRunner(exp_config)
                result = runner.run_single_experiment()
                
                # 保存单个实验结果
                self.save_individual_result(exp['name'], result)
                
            except Exception as e:
                logger.error(f"实验 {exp['name']} 执行失败: {str(e)}")
                continue
        
        # 生成并保存汇总报告
        if self.results:  # 只在有结果时生成报告
            self.generate_summary_report()
        else:
            logger.warning("没有成功的实验结果，跳过生成汇总报告")

    def _validate_config(self):
        """验证批处理配置"""
        if 'data' not in self.config:
            logger.error("配置中缺少 'data' 部分")
            return False
            
        if 'experiment_dir' not in self.config['data']:
            logger.error("配置中缺少 'experiment_dir' 设置")
            return False
            
        experiment_dir = Path(self.config['data']['experiment_dir'])
        if not experiment_dir.exists():
            logger.error(f"实验目录不存在: {experiment_dir}")
            return False
            
        return True
    
    def save_individual_result(self, dataset_name: str, result: Dict):
        """完善的结果保存方法"""
        try:
            # 深度转换所有数值类型
            def deep_convert(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: deep_convert(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [deep_convert(item) for item in obj]
                elif hasattr(obj, 'to_dict'):
                    return deep_convert(obj.to_dict())
                return obj
                
            # 转换结果
            converted_result = deep_convert(result)
            
            # 保存前验证
            try:
                json.dumps(converted_result)
            except TypeError as e:
                logger.error(f"结果验证失败，存在不可序列化的类型: {str(e)}")
                # 打印问题数据的类型信息
                def print_types(obj, path=""):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            print_types(v, f"{path}.{k}" if path else k)
                    elif isinstance(obj, (list, tuple)):
                        for i, v in enumerate(obj):
                            print_types(v, f"{path}[{i}]")
                    else:
                        logger.debug(f"Path: {path}, Type: {type(obj)}")
                        
                print_types(converted_result)
                raise
                
            # 保存结果
            output_path = self.individual_results_dir / f"{dataset_name}_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_result, f, indent=2, ensure_ascii=False)
                
            self.results[dataset_name] = converted_result
            
            # 打印指标
            if 'metrics' in converted_result:
                metrics = converted_result['metrics']
                logger.info(f"\n{dataset_name} Results:")
                logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
                logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
                logger.info(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
                
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            logger.debug("错误详情:", exc_info=True)

    def generate_summary_report(self):
        """生成汇总报告"""
        try:
            if not self.results:
                logger.warning("没有结果可供汇总")
                return
                
            summary = {
                'overall_metrics': {
                    'avg_precision': np.mean([r['metrics']['precision'] for r in self.results.values()]).item(),
                    'avg_recall': np.mean([r['metrics']['recall'] for r in self.results.values()]).item(),
                    'avg_f1': np.mean([r['metrics']['f1_score'] for r in self.results.values()]).item()
                },
                'dataset_metrics': {
                    name: result['metrics']
                    for name, result in self.results.items()
                }
            }
            
            # 保存汇总报告
            out_dir = Path(self.config['data']['output_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)
            
            with open(out_dir / 'summary_report.json', 'w') as f:
                json.dump(summary, f, indent=2)
                
            # 打印汇总结果
            logger.info("\nOverall Results:")
            logger.info(f"Average Precision: {summary['overall_metrics']['avg_precision']:.4f}")
            logger.info(f"Average Recall: {summary['overall_metrics']['avg_recall']:.4f}")
            logger.info(f"Average F1 Score: {summary['overall_metrics']['avg_f1']:.4f}")
        except Exception as e:
            logger.error(f"生成汇总报告失败: {str(e)}")

if __name__ == "__main__":
    try:
        # 加载配置
        config = ExperimentConfig(
            experiment_name="schema_matching_test",
            source_path="data/source.csv",
            target_path="data/target.csv", 
            ground_truth_path="data/mapping.json",
            output_dir="experiments/results",
            feature_config={
                'text': True,
                'structural': True,
                'semantic': True
            },
            model_config={
                'type': 'ollama',
                'name': 'llama3.1',
                'temperature': 0.7
            },
            scenario='metadata',
            metrics=['precision', 'recall', 'f1_score']
        )
        
        # 创建实验运行器
        runner = ExperimentRunner(config)
        
        # 运行特征消融实验
        logger.info("Running feature ablation experiments...")
        feature_results = runner.run_feature_ablation()
        
        # 运行场景对比实验
        logger.info("Running scenario comparison experiments...") 
        scenario_results = runner.run_scenario_comparison()
        
        # 运行模型对比实验
        logger.info("Running model comparison experiments...")
        model_results = runner.run_model_comparison()
        
        # 运行验证方法实验
        logger.info("Running validation analysis experiments...")
        validation_results = runner.run_validation_analysis()
        
        logger.info("All experiments completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise