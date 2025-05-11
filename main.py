# schema_matching/main.py

import argparse
import logging
from experiments.experiment_runner import BatchExperimentRunner
import yaml
from pathlib import Path
import pandas as pd
import json
from typing import Dict, Any, Optional

from experiments.runners import ExperimentRunner, ExperimentConfig
from utils.data import DataLoader

logger = logging.getLogger(__name__)

def setup_logging(config: Dict[str, Any]):
    log_config = config['logging']
    log_path = Path(log_config['file'])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_config['format'],
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Schema Matching System')
    
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--scenario', type=str, 
                       choices=['metadata', 'few_instances', 'rich_instances'],
                       help='Override scenario in config')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch experiments')
    parser.add_argument('--data-dir', type=str,
                       help='Directory containing all Musicians datasets')
    
    return parser.parse_args()

def load_config(args) -> Dict[str, Any]:
    """加载并合并配置
    
    Args:
        args: 命令行参数
    """
    # 首先加载实验配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # 更新数据目录
        if args.data_dir:
            config['data']['base_dir'] = args.data_dir
            
        # 更新场景
        if args.scenario:
            config['experiment']['scenario'] = args.scenario
            
        return config
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}")
        raise


def main():
    # 解析参数
    args = parse_args()
    
    try:
        # 加载配置 - 修改这里，只传入args参数
        config = load_config(args)
        
        # 设置日志
        logger = setup_logging(config)
        logger.info("Starting schema matching experiment")
        
        if args.batch:
            # 运行批量实验
            batch_runner = BatchExperimentRunner(
                base_dir=args.data_dir,
                config=config
            )
            batch_runner.run_batch()
            logger.info("Batch experiments completed")
        else:
            # 运行单个实验
            experiment_config = ExperimentConfig(**config)
            runner = ExperimentRunner(experiment_config)
            runner.run_single_experiment()
            logger.info("Single experiment completed")
            
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise
    
if __name__ == '__main__':
    main()
    
