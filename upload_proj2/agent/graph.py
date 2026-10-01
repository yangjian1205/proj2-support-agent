# agent/graph.py
# 把「判断意图 → 按意图分流 → 干活」三个动作拼成一张图，并规定每条边怎么走。
# D27 改动：① query 独立成节点并挂工具  ② 加 tools 节点 + 循环  ③ triage 换结构化输出
# D28 改动：高危这一路从「回一句话」升级成「提案 → 人工审批 → 执行 → 落审计」


# agent/graph.py
# 把节点和边拼成一张图——这就是agent的1骨架
import uuid
from datetime import datetime
from typing import Literal  # 常是为了让 LLM 只能从几个固定选项里选参数值。

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig  # Runnable 不是「某几个类」，而是一份契约。谁签了这份契约（有 invoke/batch/stream），谁就能被塞进管道、被 LangGraph 当节点用。
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition # prebuilt 就是「预先造好的成品零件」
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent import audit
from agent.llm import get_llm
from agent.state import SupportState
from agent.tools import ORDERS, TOOLS, do_change_address, do_refund

# ---------- ① 结构化输出：给模型的答案套一个模具 ----------
class TriageDecision(BaseModel):
    """意图分流的结果。模型只能从这三个标签里选一个。"""
    intent: Literal["chat", "query", "high_risk"] = Field(description="意图标签")
    reason: str = Field(description="一句话说明判断依据")

# ---------- ② ★ D28 新增：高危操作的「提案」模具 ----------
class ActionPlan(BaseModel):
    """模型对高危请求的理解结果。注意它只能『描述』，不能执行。"""

    action: Literal["refund_order", "change_address", "unknown"] = Field(
        description="要执行的动作；用户意图不明确就填 unknown"
    )
    order_id: str = Field(default="",description="订单号，例如 A1001；没有就填空字符串")
    amount: float = Field(default=0.0, description="退款金额，用户没提就填 0")
    new_address: str = Field(default="", description="新的收货地址，不是改地址就填空")
    reason:str = Field(description="一句话说明用户想要什么")

# ---------- 提示词 ----------

TRIAGE_SYSTEM = """你在给客服会话分流。只输出一个单词，不要解释：

chat      —— 普通咨询：问政策、问产品、闲聊、问怎么操作
query     —— 需要查数据才能回答：问订单、物流、库存、账户余额
high_risk —— 要求改数据或动钱：退款、退货、改地址、改金额、投诉升级

判断不了就输出 chat。"""

CHAT_SYSTEM = """你是「云栖商城」的在线客服小云，说话简短、口语化、一次只问一件事。
现在还没接业务系统，涉及具体订单/物流时，请用户提供订单号，或告诉他会帮他转查询。
不要编造任何订单号、金额、时间。"""

QUERY_SYSTEM = """你是「云栖商城」的在线客服小云。规则：
1. 用户问订单、物流、库存时，必须调用工具查真实数据，不许自己编。
2. 工具返回什么就说什么，不要添油加醋；工具说查不到，就照实说查不到。
3. 说话简短口语化，一次只解决一件事。"""

PREPARE_SYSTEM = """你在把一个「高危请求」整理成一张审批单。只做信息提取，不要回答用户。
规则：
1. 退款/退货 → action 填 refund_order，amount 填用户说的金额（没提就填 0）。
2. 改收货地址 → action 填 change_address，new_address 填新地址全文。
3. 用户没说订单号、或意图不属于上面两类 → action 填 unknown。
4. 绝不编造订单号。"""

# 高危但缺信息时的追问话术
NEED_INFO_REPLY = (
    "这个操作需要人工专员审核后才能执行。麻烦你把订单号、以及具体想怎么处理告诉我，"
    "我这边帮你提交审批。"
)

# 模块级创建一次，全图共用（避免每个节点各建一个）
_llm = get_llm()

# 分流用的模型：输出被强制成 TriageDecision 对象，不会是乱七八糟的文字
# ★ method="function_calling" 是必须的！默认值 json_schema 在 DeepSeek 上直接报 400
_llm_triage = _llm.with_structured_output(TriageDecision, method="function_calling") # TriageDecision 就是"模型输出"和"代码判断"之间的那道接口。这里不是给人看的是给代码看的所以必须转
_llm_plan = _llm.with_structured_output(ActionPlan, method="function_calling")

# 查询用的模型：把三个工具的说明书塞给它，它就学会「什么时候该按哪个按钮」
_llm_tools = _llm.bind_tools(TOOLS)


# ---------- 节点函数 ----------
# 约定：输入是当前 state，返回「要更新的字段」字典（不用返回整个 state）


def triage_node(state:SupportState) -> dict:  # SupertState全局共享
    """意图分流节点：读用户最新一句话，判出 chat / query / high_risk。"""
    # 注意取消息的方式：state["messages"] 里是 Message 对象（不是 dict），
    # 所以要用 .content 点号访问，不能写 ["content"]
    last = state["messages"][-1].content # 从消息列表里拿最新的一条，读他的1正文
    # 方括号拿字典的键，点号拿对象的属性。 一个表达式里两种语法混着用，看着别扭但完全正确。

    decision = _llm_triage.invoke([
        SystemMessage(content=TRIAGE_SYSTEM),
        {"role": "user", "content": last},
    ])
    # decision 现在是一个 TriageDecision 对象，不是字符串。
    # decision.intent 一定是三个标签之一（Pydantic 帮你保证了），不用再写关键词兜底。
    
    return {"intent":decision.intent}  # 认不出来就按普通咨询走最安全

def route_intent(state: SupportState) -> str:
    """条件边的判断题：返回一个字符串，告诉图下一步去哪。

    纯代码判断，不调 LLM —— 安全分支必须由代码决定，不能交给模型自己选。
    """
    intent = state.get("intent", "chat")
    if intent == "high_risk":
        return "escalate"
    if intent == "query":        # ★ 今天新增的这一路
        return "query"
    return "chat"

def chat_node(state:SupportState) -> dict:
    """普通聊天节点：不带工具。"""
    # 这里直接传整段 message--add_message 保证历史都在里面，这就是多轮记忆
    msgs = [SystemMessage(content=CHAT_SYSTEM)] + list(state["messages"]) # 系统提示词+全部历史消息
    resp = _llm.invoke(msgs)
    # 返回 {"messages": [新消息]}：因为 messages 有 add_messages 归约器，
    # 这条会被「追加」进历史，而不是覆盖
    return {"messages":[resp]}

def query_node(state: SupportState) -> dict:
    """查询节点：挂着三个只读工具。"""
    msgs = [SystemMessage(content=QUERY_SYSTEM)] + list(state["messages"])
    resp = _llm_tools.invoke(msgs)
    return {"messages": [resp]}


def prepare_action_node(state:SupportState) -> dict:
    """★ D28 新增：把用户的话整理成一张「审批单」。

    这一节点绝不改数据 —— 它只写 state["pending_action"]。
    副作用（真退款、真改地址）全部留给审批之后的 execute_action。
    """
    last = state["messages"][-1].content
    plan = _llm_plan.invoke(
        [SystemMessage(content=PREPARE_SYSTEM),{"role":"user","content":last}]
    )

    # 信息不全 → 不进审批流程，回一句追问就结束
    if plan.action == "unknown" or not plan.order_id:
        return{
            "pending_action": {},   # 待审批清单
            "approval":{"status":"skipped"}, # 跳过本次审批
            "messages":[{"role":"assistant","content":NEED_INFO_REPLY}]  # 往历史对话里追加信息
        }

    # 代码补事实：用户没报金额就按订单实付全额；顺手查商品名，让审批单更好读
    order_id = plan.order_id.strip().upper()
    o = ORDERS.get(order_id)
    amount = plan.amount
    if plan.action == "refund_order" and amount <= 0:  # 这里是退款操作且用户没说金额就默认0
        amount = o["amount"] if o else 0.0  # 三元表达式 是当前订单就显示金额 否则就0

    # 越权/不存在 → 早退，别浪费人工去审一张非法单子
    uid = state.get("user_id")
    if not o or o["user_id"] != uid:
        return {
            "pending_action": {},
            "approval": {"status": "skipped"},
            "messages": [
                {"role": "assistant",
                 "content": f"订单 {order_id} 查不到，或者不属于当前账号，没法提交。"}
            ],
        }


    # 审批单里的文字由代码拼，不用模型的话 —— 审批人看的是数字，不能是「模型说」
    if plan.action =="refund_order":
        summary = f"订单 {order_id} ({o['items']}) 申请退款 ￥{amount}"
    else:
        summary = f"订单 {order_id}（{o['items']}）改收货地址：{o['address']} → {plan.new_address}"

    return {
        "pending_action": {
            "ticket_id": f"tk-{uuid.uuid4().hex[:6]}",
            "action": plan.action,
            "order_id": order_id,
            "amount": amount,
            "new_address": plan.new_address,
            "summary": summary,
            "reason": plan.reason,
            "requested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "approval": {},
    }


def has_action(state:SupportState) -> str:
    """prepare_action 之后：有单子就送审，没有就直接结束。"""
    return "gate" if state.get("pending_action") else "end"
    # approval_gate 节点 对应的是"gate" 也就是送审


def approval_gate_node(state:SupportState,config:RunnableConfig) ->dict:
    """★ D28 核心：整个项目唯一一处挂起点。

    这个节点里只做两件事：① interrupt 挂起等人工；② 审批回来后落审计。
    ✋ 绝不能在这里改业务数据 —— 因为 interrupt 恢复时整个节点会从头重跑，
       写在 interrupt 之前的副作用会执行两次（实测验证过）。
    """

    p = state.get("pending_action") or {} # 节点准备好的审批，否则就返回空
    # ★ D29 新增：把这张单子登记进待办表，HTTP 层才能拿 ticket_id 找回 thread_id。
    # 位置在 interrupt 之前 —— 恢复时节点会重跑，靠主键 + INSERT OR IGNORE 保证幂等。
    audit.register_pending(
        {**p, "thread_id": (config.get("configurable") or {}).get("thread_id")}
    )

   
    # ===== 挂起：把审批单交给图外面的人 =====
    # interrupt() 一被执行，图就在这里冻结并存档，invoke() 立刻返回。
    # 下面那行 decision = ... 只有在「人工审批完再 invoke 一次」之后才会跑到。
    decision = interrupt(
        {
            "ticket_id": p.get("ticket_id"),
            "thread_id": (config.get("configurable") or {}).get("thread_id"),
            "action": p.get("action"),
            "order_id": p.get("order_id"),
            "amount": p.get("amount"),
            "new_address": p.get("new_address"),
            "summary": p.get("summary"),
            "reason": p.get("reason"),
            "requested_at": p.get("requested_at"),
        }
    )
    # 所以这里是挂起 等待人工审批之后再执行 这里是第一次invoke

    # ===== 恢复：decision 就是主程序 resume 时传进来的东西 =====  # 这里是第二次invoke 然后才有下边的分支
    if isinstance(decision,str):        # 容错：只传了字符串
        decision = {"status":decision}
    approval = { 
        "status":decision.get("status","rejected"),  # 取审批结论 取不到就返回rejected
        "operator":decision.get("operator","staff-01"),  # 取审批人
        "note": decision.get("note","")  # 取备注
    }
    # 这里才是执行上边挂起的但是只取三个
    

    # 审批一落地就写审计（此时还没执行，result留空）
    audit.log_decision(   # 写一条审计记录
        {**p, "thread_id": (config.get("configurable") or {}).get("thread_id")}, # 也就是说这里主要是取thread_id 且这里是n+1个键值对
        approval,
    )
    return {"approval": approval}

#     config = {
#     "configurable": {                      # ← 第一层：这次运行的「身份」
#         "thread_id": "user-1001",
#         "checkpoint_ns": "",
#         "checkpoint_id": "1ef7a2b3-...",
#     },
#     "recursion_limit": 8,                  # ← 外层：框架「怎么跑」
#     "callbacks": [...],                    #    回调函数列表（做监控/日志用）
#     "metadata": {...},                     #    自定义元数据
#     "tags": [...],                         #    标签
# }

def route_after_approval(state: SupportState) -> str:
    """审批完的分叉：批了去执行，拒了去道个歉。"""
    status = (state.get("approval") or {}).get("status")
    return "execute" if status == "approved" else "reject"  # {execute:execute_action}

def execute_action_node(state: SupportState) -> dict:
    """★ D28 新增：真正改数据的唯一地方。跑到这里说明已经有人批过了。"""
    p = state["pending_action"]
    uid = state.get("user_id", "")

    if p["action"] == "refund_order":
        result = do_refund(p["order_id"], p["amount"], uid)
    else:
        result = do_change_address(p["order_id"], p["new_address"], uid)

    audit.log_execution(p["ticket_id"], result)       # 执行结果补回同一条审计记录 	audit.log_decision这是他前面的
    return {"messages": [{"role": "assistant", "content": result}]} # 返回助手消息以及刚才的结果


def reject_reply_node(state: SupportState) -> dict:
    """审批被拒：如实告诉用户，并把拒绝理由带上。"""
    p = state["pending_action"]
    a = state.get("approval") or {}
    note = a.get("note") or "不符合平台退款/改址规则"
    return {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"你提交的「{p['summary']}」人工审核没有通过（单号 {p['ticket_id']}）。"
                    f"原因：{note}。如果还有疑问，可以把具体情况告诉我，我帮你再登记一次。"
                ),
            }
        ]
    }













# ---------- 拼图 ----------


def build_graph(checkpointer=None):
    """创建并编译图。checkpointer 传进来 —— 传了才有记忆，也才挂得住。"""
    audit.init_db()

    builder = StateGraph(SupportState)

    builder.add_node("triage", triage_node)
    builder.add_node("chat", chat_node)
    builder.add_node("query", query_node)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    builder.add_node("prepare_action", prepare_action_node)      # ★ D28
    builder.add_node("approval_gate", approval_gate_node)        # ★ D28
    builder.add_node("execute_action", execute_action_node)      # ★ D28
    builder.add_node("reject_reply", reject_reply_node)          # ★ D28

    builder.add_edge(START, "triage")

    # 条件边：键 = route_intent 的返回值（逻辑名），值 = 真实节点名
    builder.add_conditional_edges(
        "triage",
        route_intent,
        {"chat": "chat", "query": "query", "escalate": "prepare_action"},
    )

    builder.add_conditional_edges(
        "query",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "query")

    # ★ D28：高危这一路 —— 提案 → 分流 → 挂起审批 → 执行/拒绝
    builder.add_conditional_edges(
        "prepare_action", has_action, {"gate": "approval_gate", "end": END}
    )
    builder.add_conditional_edges(
        "approval_gate", route_after_approval,
        {"execute": "execute_action", "reject": "reject_reply"},
    )
    builder.add_edge("execute_action", END)
    builder.add_edge("reject_reply", END)

    builder.add_edge("chat", END)

    return builder.compile(checkpointer=checkpointer)





