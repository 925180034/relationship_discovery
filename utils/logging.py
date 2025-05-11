# utils/logging.py

import logging
from pathlib import Path
from typing import Dict, Any

class LoggingManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not LoggingManager._initialized:
            self.root_logger = logging.getLogger()
            LoggingManager._initialized = True
    
    def setup_logging(self, config: Dict[str, Any]) -> None:
        """Setup logging configuration"""
        if not config:
            config = {
                'level': 'INFO',
                'file': 'logs/experiment.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
            
        # Clear existing handlers
        self.root_logger.handlers.clear()
        
        # Create log directory if needed
        log_path = Path(config['file'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set log level
        self.root_logger.setLevel(getattr(logging, config['level']))
        
        # Add file handler
        file_handler = logging.FileHandler(config['file'])
        file_handler.setFormatter(logging.Formatter(config['format']))
        self.root_logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(config['format']))
        self.root_logger.addHandler(console_handler)
        
        # Set third-party loggers to WARNING
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('filelock').setLevel(logging.WARNING)
# # utils/logging.py

# import logging
# from pathlib import Path
# from typing import Dict, Any
# import yaml

# def setup_logging(config: Dict[str, Any] = None):
#     """设置日志配置"""
#     if config is None:
#         config = {
#             'level': 'INFO',
#             'file': 'logs/experiment.log',
#             'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#         }
    
#     # 创建日志目录
#     log_dir = Path(config['file']).parent
#     log_dir.mkdir(parents=True, exist_ok=True)
    
#     # 配置根日志记录器
#     root_logger = logging.getLogger()
#     root_logger.setLevel(getattr(logging, config['level']))
    
#     # 清除现有处理器
#     root_logger.handlers = []
    
#     # 添加文件处理器
#     file_handler = logging.FileHandler(config['file'])
#     file_handler.setFormatter(logging.Formatter(config['format']))
#     root_logger.addHandler(file_handler)
    
#     # 添加控制台处理器
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(logging.Formatter(config['format']))
#     root_logger.addHandler(console_handler)
    
#     # 关闭一些特定模块的DEBUG日志
#     logging.getLogger('urllib3').setLevel(logging.WARNING)
#     logging.getLogger('filelock').setLevel(logging.WARNING)
    
#     logger = logging.getLogger(__name__)
#     logger.info("日志系统初始化完成")