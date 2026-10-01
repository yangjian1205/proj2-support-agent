# agent/state.py
# 客服 Agent 的「共享状态」：整张图里所有节点都能读、能写的数据结构
# 为什么要单独一个文件：状态是全局契约，谁改了它所有节点都受影响，必须显眼

# LangGraph 是"跑流程的发动机" LangChain 是"调模型的工具带"
from typing import Annotated,TypedDict    # 两个标注工具
from langgraph.graph import add_messages  

class SupportState(TypedDict):
    """客服会话的共享状态。每个节点返回 {"某字段": 新值} 就会合并进来。"""

    # 对话历史。注意 Annotated[list, add_messages] 不是普通 list：
    #   普通 list  ← 每次赋值是「整个替换」，第二轮对话会把第一轮冲掉
    #   add_messages ← 每次是「追加」，历史自然累积，这就是多轮记忆的地基

    messages: Annotated[list,add_messages]   # 这里就是规约器 累加  ，下边的几个属于直接覆盖了

    # 这个会话属于哪个用户(D27 查订单要用)
    user_id: str

    # triage 节点判出来的意图：chat / query / high_risk
    # 条件边靠它决定走哪条路
    intent: str

    # ★ D28 新增：准备执行的写操作（prepare_action 写，approval_gate 读）
    # 没写 Annotated，所以是「整体替换」—— 正是我们要的：一次只审批一件事
    pending_action: dict

    # ★ D28 新增：审批结果（approval_gate 写，execute_action / reject_reply 读）
    approval: dict











