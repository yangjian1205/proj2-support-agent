# agent/llm.py
# 模型工厂：全项目唯一的 LLM 创建入口
# 为什么抽一层：D27 要给模型绑定工具、D28 要用它做判定，只在这一个地方改
# 全项目只在这一个地方创建模型实例，其他文件都从这里拿——以后要换模型、加参数，只改这一个文件。


import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 用绝对路径读 .env :不管你在哪个目录跑，都能找到项目根目录

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

def get_llm(temperature:float = 0.3) -> ChatOpenAI:  # float类型标注
    """返回一个指向 DeepSeek 的模型对象。

    DeepSeek 提供的是 OpenAI 兼容接口，所以用 ChatOpenAI 这个壳子、
    把 base_url 指过去就能用，不需要专门的 DeepSeek SDK。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
             "没读到 DEEPSEEK_API_KEY。检查两件事：① .env 在项目根目录（和 agent/ 同级）"
            "② .env 里那行没有被 # 注释掉"
        )

    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL","deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com"),
        temperature=temperature,
        timeout=60,
    )
























