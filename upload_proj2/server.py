# server.py
# D29：把命令行里的客服 Agent 包成一个 HTTP 服务 —— 别人（前端 / 另一个系统）能调的那种。
# 启动：uvicorn server:app --reload --port 8000
# 接口文档：http://127.0.0.1:8000/docs   （FastAPI 自动生成的，不用自己写前端也能点着测）
#
# 三条设计铁律（D29 真正要带走的东西）：
#   1. 会话身份由 thread_id 决定，服务端必须无状态 —— 客户端只报 ID，不报状态
#   2. 挂起中的会话不许塞新消息，否则审批单静默消失（D28 实测过），接口层必须拦
#   3. 审批接口必须可重复调用而不出错：先查待办、再查是否真挂起、最后才 resume



import sqlite3
import sys
import uuid
from contextlib import asynccontextmanager  # 异步函数管理器
from pathlib import Path
from typing import Literal

from fastapi import FastAPI,HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel,Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from agent import audit
from agent.graph import build_graph

DB_PATH = ROOT / "checkpoint.db"
RECURSION_LIMIT  = 12    # 保险丝，和main 保持一致

graph = None   # 全局唯一一张图，在lifesqan里创建

@asynccontextmanager
async def lifesqan(app:FastAPI):
    """服务启动时建一次连接，关服务时释放。

    为什么 check_same_thread=False：FastAPI 用线程池跑同步路由，同一个连接会被
    不同线程用到。SqliteSaver 内部自带 threading.Lock（源码 95 行 self.lock），
    所以「一个连接 + 多线程」是官方推荐用法，不会串数据。
    """

    global graph
    conn = sqlite3.connect(str(DB_PATH),check_same_thread=False)  # 允许跨线程
    graph = build_graph(SqliteSaver(conn))  # 存档器
    yield
    conn.close()


app = FastAPI(title="云栖商城 · 客服 Agent", version="1.0.0", lifespan=lifesqan)

# ---------- 请求 / 响应模型：FastAPI 靠它自动校验参数、自动生成文档 ----------

class ChatIn(BaseModel):
    message: str = Field(min_length=1,max_length=500,description="用户这轮说的话")
    thread_id: str | None = Field(default=None,description="不传 = 服务端新建一个会话")  # 也就是对话编号 可以不传
    user_id: str = Field(
        default="u-1001",
        description="下单账号。⚠️ 真实系统里必须从登录态/token 里取，绝不能让前端传 —— 这里是为了演示越权拦截才留的口子",
    )

class ChatOut(BaseModel):
    thread_id: str = Field(description="客户端要记下来，下一轮带上他才能继续上一轮的同一会话")
    status: Literal["done", "pending_approval"] = Field(description="这轮是跑完了，还是卡在等人工审批")
    reply: str = Field(default="", description="给用户看的话")
    intent: str | None = Field(default=None,description="triage 判出来的意图")
    ticket: dict | None = Field(default=None, description="挂起时返回的审批单，前端拿 ticket_id 去提交审批")

# 这里chatout 跟 chatin 是一对 In 管"客户端进来的东西合不合法"，Out 管"你出去的东西合不合法"。

class ApprovalIn(BaseModel):
    status: Literal["approved", "rejected"] = Field(description="批 or 拒")
    operator: str = Field(default="staff-01", description="审批人账号")
    note: str = Field(default="", description="备注 / 拒绝理由")

# ---------- 三个小工具函数 ----------

def _config(thread_id: str) -> dict:
    """每个请求现拼 config。图是无状态的，会话身份全靠这个 thread_id。"""
    return{
        "configurable" : {"thread_id" : thread_id},
        "recursion_limit" : RECURSION_LIMIT,   # 这里就是那个保险丝
    }


def _pending_of(thread_id: str) -> dict | None:   # 判断是否挂起
    """这个会话现在挂没挂起？挂了就把审批单拿回来。

    为什么必须先查：挂起状态下直接传新消息进去，LangGraph 会当新一轮重新跑，
    那张审批单就静默丢了 —— 不报错、审计表也不留痕。这是 D28 实测出来的坑。
    """
    snapshot = graph.get_state(_config(thread_id))
    if not snapshot.next:   # next为空 = 没有待执行的节点 = 没挂起
        return None
    for t in snapshot.tasks:
        if t.interrupts:
            return t.interrupts[0].value  # 这里之前用过即审批表
    return None

# ---------- 接口 1：发消息 ----------

@app.post("/chat",response_model=ChatOut)  # @app.post路由装饰器
def chat(body:ChatIn):  # 这里不能用async这是因为下面invoke是同步的
    """发一句话，拿到回复；如果是高危操作，返回一张待审批单。"""
    thread_id = body.thread_id or f"t-{uuid.uuid4().hex[:8]}" # 32个16进制字符取前八个

    stuck = _pending_of(thread_id) # 查他是否还有没挂起的单子
    if stuck:
        raise HTTPException(  # 即这里就是有审批单 此时发消息进来 直接返回409 不动 等待审批完成
            status_code=409,
            detail={
                "code": "thread_suspended", # 给程序看的错误标识
                "message": "这个会话还挂着一张没批的审批单，先把审批做完再继续聊。",
                "ticket_id": stuck.get("ticket_id"),
                "summary": stuck.get("summary"),
            },
        )

    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": body.message}], # 身份跟历史记录 这里是追加不是覆盖
            "user_id": body.user_id,  # 身份进state
        },
        _config(thread_id),
    )

    # 挂着 = 这轮没跑完，返回审批单让前端去弹窗，而不是干等
    if result.get("__interrupt__"):
        ticket = result["__interrupt__"][0].value
        return ChatOut(
            thread_id=thread_id,
            status="pending_approval",
            reply="这个操作需要人工审核后才能执行，我已经把申请提交上去了。",
            intent=result.get("intent"),
            ticket=ticket,
        )

    return ChatOut(
        thread_id=thread_id,
        status="done",
        reply=result["messages"][-1].content,
        intent=result.get("intent"),
    )

# ---------- 接口 2：提交审批决定 ----------
# 客服agent停在了等人审批上，同意或者拒绝就调这个接口

@app.post("/approvals/{ticket_id}")
def decide(ticket_id:str,body:ApprovalIn):  # ApprovalIn这是请求体
    """人工审批。注意：这个接口会被前端重复点，必须自己防重。

    三道关卡，一道都不能少：
      ① pending 表里有没有这张单（没有 = 批过了 / 单号错）→ 404
      ② 这个会话现在还挂不挂着 → 不挂 → 409（不拦的话 Command(resume) 会从 START 重跑整张图）
      ③ 挂着的这张单，是不是路径里这张单 → 不是 → 409（防张冠李戴）
    """


    row = audit.get_pending(ticket_id)  # 去待办里查审批
    if row is None:
        raise HTTPException(
            status_code=404,  # 抛出错误类型 以及错误原因
            detail={"code":"ticket_not_found","message":f"没有待办的工单{ticket_id}(可能已经处理过了)"},
        )

    thread_id = row["thread_id"]  # 从这行会话里取出ID
    cfg = _config(thread_id)   # 调 _config 函数把 thread_id 包成 LangGraph 要的配置字典 一般是 {"configurable": {"thread_id": ...}}

    snaqshot = graph.get_state(cfg)  # cfg是langgraph的固定参数规格 thread_id只是其中一个字段 直接传字符串或裸字典会报错
    if not snaqshot.next:  # 下一步要执行哪些节点
        audit.drop_pending(ticket_id)   # 顺手清理脏数据 这里是没挂起但是他有待办记录 必须删
        raise HTTPException(
            status_code=409, # 同上
            detail={"code":"no_pending","message":"这个会话当前可能没有挂起审批，可能已经批过了"}
        )

    live = None
    for t in snaqshot.tasks: # 待执行任务列表 找到就停
        if t.interrupts:
            live = t.interrupts[0].value
            break
    if not live or live.get("ticket_id") != ticket_id:  # 找不到挂起切两个ID不对等
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ticket_mismatch",
                "message": "这个会话当前挂起的不是这单。",
                "current_ticket": (live or {}).get("ticket_id"), # 实际挂的是哪个
            },
        )

    result = graph.invoke(  # 恢复执行
        Command(resume={"status": body.status,"operator":body.operator,"note":body.note}),
        cfg,
    )

    audit.drop_pending(ticket_id)  # 审批完成 从待办表里删掉

    return {
        "ticket_id": ticket_id,
        "thread_id": thread_id,
        "status": body.status,
        "reply": result["messages"][-1].content,
    }

# ---------- 接口 3：看审批列表 ----------
@app.get("/approvals")
def approvals(limit:int = 20) -> dict:
    """待办 + 历史一次看全。前端的「审批工作台」就靠这一个接口。"""
    return {
        "pending": [dict(r) for r in audit.list_pending(limit)],
        "history": [dict(r) for r in audit.list_approvals(limit)],
    }
# 查看历史跟待办

# ---------- 辅助接口 ----------

@app.get("/sessions/{thread_id}")
def session_state(thread_id:str) -> dict:
    """调试用：看一个会话现在的状态（挂没挂起、聊了几轮）。"""
    snapshot = graph.get_state(_config(thread_id))
    msgs = snapshot.values.get("messages",[])
    return {
        "thread_id": thread_id,
        "suspended": bool(snapshot.next),  # 下一个待执行的节点
        "next_nodes": list(snapshot.next),  # 挂起时卡在哪个节点
        "pending_ticket": (_pending_of(thread_id) or {}).get("ticket_id"), # 查看有无待审批的
        "rounds": len(msgs),
        "recent": [{"role": m.type, "content": str(m.content)[:80]} for m in msgs[-6:]],  # langgraph的message对象
    }

@app.get("/health")
def health() -> dict:
    return {"ok": True, "graph_ready": graph is not None}














