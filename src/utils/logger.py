"""日志系统模块。

提供统一的日志配置，支持文件和控制台输出。
日志文件按天轮转，最多保留7天的备份。

使用示例：
    >>> from logger import setup_logger
    >>> logger = setup_logger('my_module', 'logs', logging.DEBUG)
    >>> logger.info('Processing started')
    >>> logger.error('An error occurred')
"""

import logging
from logging.handlers import TimedRotatingFileHandler
import os
from typing import Optional


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> logging.Logger:
    """配置并返回一个Logger实例。
    
    创建同时输出到文件和控制台的Logger。
    文件日志每天轮转，最多保留7天的备份。
    
    Args:
        name: Logger的名称，通常为模块名称
        log_dir: 日志文件保存目录（默认'logs'），不存在则自动创建
        level: 日志级别（默认logging.INFO）
            - logging.DEBUG: 调试信息
            - logging.INFO: 一般信息
            - logging.WARNING: 警告信息
            - logging.ERROR: 错误信息
            - logging.CRITICAL: 严重错误
            
    Returns:
        配置好的Logger对象，可直接使用debug/info/warning/error等方法
        
    Examples:
        >>> from logger import setup_logger
        >>> import logging
        >>> logger = setup_logger('my_analysis', 'logs', logging.INFO)
        >>> logger.info('Starting data analysis')
        >>> logger.error('Failed to load data')
        
    Notes:
        - 若Logger已存在handler，不会重复添加
        - 文件名为 {name}.log
        - 文件日志格式: %(asctime)s - %(name)s - %(levelname)s - %(message)s
        - 控制台日志格式: %(asctime)s - %(levelname)s - %(message)s
        - 日志按天轮转（午夜0点），保留最近7天的备份
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复日志输出

    if not logger.handlers:  # 避免重复添加 handler
        # 文件日志
        file_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            when="D",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# 默认项目日志
# 项目范围内使用此全局logger实例
logger = setup_logger("project")
