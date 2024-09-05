"""工具函数."""

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """
    从环境变量中获取值，如果不存在则使用默认值.
    
    Args:
        key: 环境变量名
        default: 默认值，可选
        
    Returns:
        环境变量值或默认值
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"环境变量 {key} 未设置且没有提供默认值")
    return value