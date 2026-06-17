import logging,os
from datetime import datetime
from utils.path_tool import get_abs_path

#日志保存的路径
LOG_ROOT = get_abs_path("logs")
#确保日志存在的目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

#定义默认日志输出格式
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s: %(lineno)d - %(message)s'
)

#获取日志信息
def get_logger(
        name:str  = "agent",
        console_level:int = logging.DEBUG,
        file_level:int = logging.INFO,
        log_file = None
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # uvicorn --reload 场景下清除残留的旧 handlers
    logger.handlers.clear()

    #配置控制台的handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    #文件handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger

logger = get_logger()
