# utils/config.py

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import copy

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, default_config_path: str = "config/default_config.yaml"):
        """Initialize configuration manager"""
        self.default_config_path = Path(default_config_path)
        self.config = self._load_default_config()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        try:
            with open(self.default_config_path) as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded default config: {self.default_config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Default config not found: {self.default_config_path}")
            return {}
            
    def load_experiment_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load and merge experiment configuration"""
        if not config_path:
            logger.info("No experiment config specified, using default config")
            return self.config.copy()
            
        try:
            with open(config_path) as f:
                exp_config = yaml.safe_load(f)
                logger.info(f"Loaded experiment config: {config_path}")
                
                # Deep merge configs
                config = self._deep_merge(self.config.copy(), exp_config)
                
                # Apply scenario-specific config if specified
                scenario = config.get('experiment', {}).get('scenario', 'metadata')
                config = self._apply_scenario_config(config, scenario)
                
                return config
                
        except FileNotFoundError:
            logger.warning(f"Experiment config not found: {config_path}")
            return self.config.copy()
        except Exception as e:
            logger.error(f"Failed to load experiment config: {str(e)}")
            return self.config.copy()
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Recursively merge two dictionaries with proper update behavior
        
        Args:
            base: Base dictionary to merge into
            update: Dictionary with update values
            
        Returns:
            Merged dictionary
        """
        for key, value in update.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    base[key] = self._deep_merge(base[key], value)
                elif isinstance(base[key], list) and isinstance(value, list):
                    # For lists, extend base list with update list
                    base[key].extend(value)
                else:
                    # For all other types, update overwrites base
                    base[key] = copy.deepcopy(value)
            else:
                # New key, add it
                base[key] = copy.deepcopy(value)
        return base
        
    def _apply_scenario_config(self, config: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Apply scenario-specific configuration"""
        result = copy.deepcopy(config)
        
        # 获取场景配置
        scenario_config = config.get('scenarios', {}).get(scenario, {})
        if not scenario_config:
            return result
            
        # 更新特征提取器配置
        if 'feature_extractor' in scenario_config:
            result.setdefault('feature_extractor', {})
            result['feature_extractor'].update(scenario_config['feature_extractor'])
        
        # 更新阈值配置
        if 'thresholds' in scenario_config:
            # 确保不会使用硬编码的默认值
            result.setdefault('plan_generator', {})
            result['plan_generator'].setdefault('thresholds', {})
            result['plan_generator']['thresholds'].update(scenario_config['thresholds'])
        
        return result

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and required values
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Required top-level sections
            required_sections = ['experiment', 'data', 'feature_extractor', 'model']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required config section: {section}")
                    return False
            
            # Validate experiment section
            experiment = config['experiment']
            if 'scenario' not in experiment:
                logger.error("Missing scenario in experiment config")
                return False
            if experiment['scenario'] not in ['metadata', 'few_instances', 'rich_instances']:
                logger.error(f"Invalid scenario: {experiment['scenario']}")
                return False
            
            # Validate data section
            data = config['data']
            required_data = ['source_table', 'target_table', 'output_dir']
            for field in required_data:
                if field not in data:
                    logger.error(f"Missing required data field: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed: {str(e)}")
            return False
        
# # utils/config.py

# import yaml
# from pathlib import Path
# from typing import Dict, Any, Optional
# import logging
# import copy

# logger = logging.getLogger(__name__)

# class ConfigManager:
#     def __init__(self, default_config_path: str = "config/default_config.yaml"):
#         """初始化配置管理器
        
#         Args:
#             default_config_path: 默认配置文件路径
#         """
#         self.default_config_path = Path(default_config_path)
#         self.config = self._load_default_config()
        
#     def _load_default_config(self) -> Dict[str, Any]:
#         """加载默认配置"""
#         try:
#             with open(self.default_config_path) as f:
#                 config = yaml.safe_load(f)
#                 logger.info(f"加载默认配置: {self.default_config_path}")
#                 return config
#         except FileNotFoundError:
#             logger.warning(f"默认配置文件未找到: {self.default_config_path}")
#             return {}
            
#     def load_experiment_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
#         """加载并合并实验配置
        
#         Args:
#             config_path: 实验配置文件路径
            
#         Returns:
#             合并后的配置字典
#         """
#         if not config_path:
#             logger.info("未指定实验配置，使用默认配置")
#             return self.config.copy()
            
#         try:
#             with open(config_path) as f:
#                 exp_config = yaml.safe_load(f)
#                 logger.info(f"加载实验配置: {config_path}")
                
#                 # 合并基础配置
#                 config = self._deep_merge(self.config.copy(), exp_config)
                
#                 # 处理场景特定配置
#                 scenario = config.get('experiment', {}).get('scenario', 'metadata')
#                 config = self._apply_scenario_config(config, scenario)
                
#                 return config
                
#         except FileNotFoundError:
#             logger.warning(f"实验配置文件未找到: {config_path}")
#             return self.config.copy()
#         except Exception as e:
#             logger.error(f"加载实验配置失败: {str(e)}")
#             return self.config.copy()
    
#     def _deep_merge(self, base: Dict, update: Dict) -> Dict:
#         """递归合并两个字典，确保更新字典的值覆盖基础字典的值
        
#         Args:
#             base: 基础字典
#             update: 更新字典
            
#         Returns:
#             合并后的字典
#         """
#         for key, value in update.items():
#             if key in base and isinstance(base[key], dict) and isinstance(value, dict):
#                 base[key] = self._deep_merge(base[key], value)
#             else:
#                 base[key] = copy.deepcopy(value)
#         return base
        
#     def _apply_scenario_config(self, config: Dict[str, Any], scenario: str) -> Dict[str, Any]:
#         """应用场景特定配置
        
#         Args:
#             config: 当前配置字典
#             scenario: 场景名称
            
#         Returns:
#             更新后的配置字典
#         """
#         # 获取场景配置
#         scenarios_config = config.get('experiments', {}).get('scenarios', {})
#         if scenario not in scenarios_config:
#             return config
            
#         scenario_config = scenarios_config[scenario]
#         logger.info(f"应用场景 {scenario} 的特定配置")
        
#         # 复制配置以避免修改原始配置
#         result = copy.deepcopy(config)
        
#         # 更新特征提取器配置
#         if 'feature_extractor' in scenario_config:
#             result.setdefault('feature_extractor', {})
#             result['feature_extractor'].update(scenario_config['feature_extractor'])
        
#         # 更新阈值配置
#         if 'thresholds' in scenario_config:
#             result.setdefault('plan_generator', {})
#             result['plan_generator'].setdefault('thresholds', {})
#             result['plan_generator']['thresholds'].update(scenario_config['thresholds'])
        
#         # 更新权重配置
#         if 'weights' in scenario_config:
#             result.setdefault('plan_generator', {})
#             result['plan_generator'].setdefault('weights', {})
#             result['plan_generator']['weights'].update(scenario_config['weights'])
        
#         return result

# # # utils/config.py

# # import yaml
# # from pathlib import Path
# # from typing import Dict, Any, Optional
# # import logging
# # import copy

# # logger = logging.getLogger(__name__)

# # class ConfigManager:
# #     def __init__(self, default_config_path: str = "config/default_config.yaml"):
# #         self.default_config_path = Path(default_config_path)
# #         self.config = self._load_default_config()
    
# #     def _load_default_config(self) -> Dict[str, Any]:
# #         """Load default configuration"""
# #         try:
# #             with open(self.default_config_path) as f:
# #                 return yaml.safe_load(f)
# #         except FileNotFoundError:
# #             logger.warning(f"Default config not found at {self.default_config_path}")
# #             return {}
            
# #     def load_experiment_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
# #         """Load and merge experiment configuration"""
# #         if not config_path:
# #             return self.config.copy()
            
# #         try:
# #             with open(config_path) as f:
# #                 exp_config = yaml.safe_load(f)
# #                 return self._deep_merge(self.config.copy(), exp_config)
# #         except FileNotFoundError:
# #             logger.warning(f"Experiment config not found at {config_path}")
# #             return self.config.copy()
    
# #     def _deep_merge(self, base: Dict, update: Dict) -> Dict:
# #         """Deep merge two dictionaries"""
# #         for key, value in update.items():
# #             if key in base and isinstance(base[key], dict) and isinstance(value, dict):
# #                 base[key] = self._deep_merge(base[key], value)
# #             else:
# #                 base[key] = value
# #         return base

