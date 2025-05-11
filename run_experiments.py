import argparse
import logging
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

from utils.config import ConfigManager
from utils.logging import LoggingManager
from experiments.experiment_runner import ExperimentRunner, dict_to_experiment_config


logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', 
                       choices=['metadata', 'few_instances', 'rich_instances'])
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--scenario-comparison', action='store_true') 
    parser.add_argument('--config', type=str, default="config/experiment_config.yaml",
                       help='Experiment configuration file path')
    parser.add_argument('--default-config', type=str, default="config/default_config.yaml",
                       help='Default configuration file path')
    return parser.parse_args()


def load_config(config_path: str = "config/experiment_config.yaml", 
              default_config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """加载配置文件,合并默认配置和实验配置"""
    # 确保文件扩展名正确
    config_path = str(Path(config_path).with_suffix('.yaml'))
    default_config_path = str(Path(default_config_path).with_suffix('.yaml'))
    
    # 加载默认配置
    try:
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"默认配置文件未找到: {default_config_path}")
        config = {}
        
    # 加载实验配置
    try:
        with open(config_path, 'r') as f:
            exp_config = yaml.safe_load(f)
            config = deep_update(config, exp_config)
    except FileNotFoundError:
        logger.warning(f"未找到配置文件: {config_path}")
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}")
        raise
        
    return config


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """递归更新嵌套字典"""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def setup_logging(config_path: str = "config/experiment_config.yaml"):
    """设置日志配置"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            log_config = config.get('logging', {})
    except Exception:
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'experiments/logs/experiment.log'
        }

    # 创建日志目录
    log_dir = Path('experiments/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', 'experiments/logs/experiment.log')),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 设置各个模块的日志级别
    loggers = {
        'features': logging.INFO,
        'planning': logging.INFO,
        'models': logging.INFO,
        'experiments': logging.INFO,
        'validation': logging.INFO
    }
    
    for logger_name, level in loggers.items():
        logging.getLogger(logger_name).setLevel(level)
        
    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成")    

def main():
    """Main entry point for running experiments"""
    args = parse_args()
    
    # Initialize configuration
    config_manager = ConfigManager(args.default_config)
    config = config_manager.load_experiment_config(args.config)
    
    # Validate configuration
    if not config_manager.validate_config(config):
        logger.error("Invalid configuration")
        return
    
    # Update config based on command line arguments
    if args.scenario:
        if 'experiment' not in config:
            config['experiment'] = {}
        config['experiment']['scenario'] = args.scenario
        
        # Apply scenario-specific config
        config = config_manager._apply_scenario_config(config, args.scenario)
    
    # Setup logging
    LoggingManager().setup_logging(config.get('logging', {}))
    
    try:
        # Create experiment runner
        runner = ExperimentRunner(config)
        
        # Run appropriate experiment type
        if args.ablation:
            logger.info("Running feature ablation experiments")
            results = runner.run_feature_ablation()
        elif args.scenario_comparison:
            logger.info("Running scenario comparison experiments")
            results = runner.run_scenario_comparison()
        else:
            logger.info(f"Running single scenario: {config['experiment']['scenario']}")
            results = runner.run_single_experiment()
            
        logger.info("Experiments completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()