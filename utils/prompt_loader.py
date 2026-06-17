import os
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.config_handler import prompts_config


#加载系统提示词
def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_config.get("main_prompt_path"))
    except KeyError as e:
        logger.error(f"在 yaml 中没有找到 main_prompt_path 配置项，系统无法启动！")
        raise e

    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"解析系统主提示词出现问题，文件路径: {system_prompt_path}")
        raise e



#加载rag提示词
def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_config.get("rag_summarize_prompt_path"))
    except KeyError as e:
        logger.error(f"在yaml中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析rag提示词出现问题，文件路径: {rag_prompt_path}")
        raise e



#加载报告提示词
def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_config.get("report_prompt_path"))
    except KeyError as e:
        logger.error(f"在yaml中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"解析报告提示词出现问题，文件路径: {report_prompt_path}")
        raise e
