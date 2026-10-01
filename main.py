# 这个文件负责「接用户输入、把图跑起来、把答案打印出来」，并且用 thread_id 决定「这是哪个人的会话」。

# main.py
# 命令行入口：接输入 → 跑图 → 打印。记忆能力由 SQLite 检查点提供
# 启动：python main.py
# D28：命令行入口。新增「人工审批」环节 —— 图挂起后由这里收人工决定，再 resume。


import sqlite3
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command  #  Command(resume=...) 也就是挂起

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from agent.audit import list_approvals
from agent.graph import build_graph 

DB_PATH = ROOT / "checkpoint.db"


def ask_huaman(payload:dict) -> dict | None: # 问人 有决定就返回 没有就空
    """把审批单打给人看，收回决定。返回要 resume 进图里的字典。

    返回 None 的两种情况：人工还没决定就按了 Ctrl+C / 关掉了输入。
    这时审批单会原样留在 checkpoint 里，不算「拒绝」——下次进来还能接着批。
    """

    print("\n" + "!" * 52)
    print("  ⚠️  需要人工审批（Agent 已挂起，等你决定）")
    print("!" * 52)
    print(f"  单号    : {payload.get('ticket_id')}")
    print(f"  会话    : {payload.get('thread_id')}")
    print(f"  动作    : {payload.get('action')}")
    print(f"  订单    : {payload.get('order_id')}")
    print(f"  内容    : {payload.get('summary')}")
    print(f"  用户原话: {payload.get('reason')}")
    print("!" * 52)

    try:
        ans = input("审批 [y=批准 / n=拒绝 / n=理由=拒绝带理由] ：").strip()
        # input(...) 把提示语打在屏幕上，然后程序停住，等你敲字并按回车
    except (EOFError,KeyboardInterrupt):
        print("\n（没有做出决定，审批单保留）")
        return None
    low = ans.lower()
    if low in ("y", "yes", "批", "批准", "同意"):
        return {"status": "approved", "operator": "staff-01", "note": ""} # 结论 审批人 备注
    # 允许「n 理由」这种写法：既拒绝了又顺手写了原因
    note = ans[2:].strip() if low.startswith("n ") else ("" if low == "n" else ans)
    # 也就是从理由开始 
    return {"status": "rejected", "operator": "staff-01", "note": note or "人工拒绝"}

def run_with_approval(graph,config,first_input) -> dict | None:
    """跑一轮，遇到挂起就处理，直到跑完为止。返回 None = 这轮还等着人批，没跑完。"""
    result = graph.invoke(first_input,config)
    while result.get("__interrupt__"):   # 只要还在挂起状态
        payload = result["__interrupt__"][0].value  # 拿走那张审批单
        decision = ask_huaman(payload)
        if decision is None:
            print(f"  审批单 {payload['ticket_id']} 已存档。下次用同一会话 ID 进来，会自动提示你处理它。")
            return None
        print(f"  → 已提交审批结果：{decision['status']}")
        result = graph.invoke(Command(resume=decision), config)   # ★ 从这里接着跑 人做了指令之后
    return result

def find_pending(graph,config) -> dict | None:  # 找有没有挂起的
    """查一下这个会话是不是还挂着一笔没批的审批单。

    为什么必须查：挂起状态下如果直接传一条新消息进去，
    LangGraph 会「当作新一轮对话」重新跑，那张审批单就静默丢了、审计表里也不会留痕。
    """

    snapshot = graph.get_state(config)  # 这是读档接口 检查点
    if not snapshot.next:          # next 为空 = 没有待执行的节点 = 没挂起
        return None
    for t in snapshot.tasks:  # 这里不是空的 
        if t.interrupts:    # 被卡住
            return t.interrupts[0].value # 取第一个中断对象，跟上边的类似一张表
    return None




def main() -> None:
    # check_same_thread=False：允许连接跨线程使用（D29 接 FastAPI 时会用到）
    conn = sqlite3.connect(str(DB_PATH),check_same_thread=False)
    checkpointer = SqliteSaver(conn)   # 建表是全自动的，不用手动 setup()
    graph = build_graph(checkpointer)  # 位置参数传入，对应 def build_graph(checkpointer=None)。这也是上一轮说的那件事——记忆开关在这里被真正打开。

    # 会话 ID：直接回车 = 新建一个；填上次那个 = 续上之前的对话

    tid = input("会话 ID (回车 = 新建一个) ：").strip()
    thread_id = tid or f"t-{uuid.uuid4().hex[:8]}"  # 左边不是空的就取左边否则取右边   # uuid.uuid4()随机唯一字符串 hex[:8]32位16进制取前8个
    config = {
    "configurable": {"thread_id": thread_id},
    "recursion_limit": 12,   # ★ 保险丝：最多走 8 步，防止工具循环停不下来
}  # langraph 硬性规定

    print(f"\n当前会话 thread_id = {thread_id}")
    print("（记下它。下次启动填同一个 ID，就能接着聊。输入 /history 看历史，exit 退出）\n")

    # ★ 开工第一件事：上一轮是不是还挂着一笔没批的审批单？
    carried = find_pending(graph,config)
    if carried:
        print("发现上一个进程遗留、还没处理的审批单，先把它批完：")
        decision = ask_huaman(carried)
        if decision is not None:
            done = graph.invoke(Command(resume=decision),config)
            print(f"客服:{done['messages'][-1].content}\n")


    while True:
        try:
            q = input("你：").strip() # 读你这一轮的提问去掉首尾空白
        except (EOFError,KeyboardInterrupt): # 捕获两种特定异常
            break

        if q.lower() in ("exit","quit","q"): 
            break 
        if not q:
            continue

        if q == "/history":
            snaqshot = graph.get_state(config)  # 从检查点读会当前状态 最新状态
            msgs = snaqshot.values.get("messages",[]) # 不存在就返回空列表 values才是状态字典
            print(f"---历史共{len(msgs)}条---")
            for m in msgs:
                role = {"human": "你", "ai": "客服", "tool": "工具"}.get(m.type, m.type)  # 是用户收的话就取你否则就取客服
                print(f" [{role}] {m.content[:60]}")
            print("-" * 30)
            continue    
        # 只传「这一轮新增的内容」：messages 有 add_messages 归约器，
        # 它会自动追加到历史后面，不用你自己拼历史


        if q =="/approvals":
            rows = list_approvals(10)  # 最多十条
            print(f"---最近 {len(rows)} 条审批记录---")
            for r in rows:
                print(
                    f" {r['ticket_id']} | {r['action']} | {r['order_id']} | {r['decision']}"
                    f" | 审批人={r['operator']} | {r['decided_at']}"
                )
                print(f"    内容: {r['summary']}")
                print(f"    执行: {r['result'] or '（未执行）'}")
            print("-" * 30)
            continue

        result = run_with_approval(
            graph,
            config,
            {"messages": [{"role": "user", "content": q}], "user_id": "u-1001"}, # 这里是最后一次的高风险路线 approal没挂规约器 所以是最后一轮，他只关心执行还是道歉不用挂，留着历史是负担
        )
        if result is None:      # 这轮还没批完，什么都不用打印
            continue

        # 只打印「本轮」的痕迹：从最后一条用户消息往后看，
        # 否则每一轮都会把前面几轮的工具调用重复打一遍


        msgs = result["messages"]  # 所有的历史记录
        start = max(i for i, m in enumerate(msgs) if m.type == "human")  # enumerate 拆成下标跟元素(0, 第一条消息) 只保留用户信息
        for m in result["messages"]:
            if m.type == "ai" and getattr(m, "tool_calls", None):   # 模型申请调工具 取值对象是模型并且调用了工具
                for tc in m.tool_calls:
                    print(f"    [调工具] {tc['name']}({tc['args']})") # 工具名和调用的参数
            elif m.type == "tool":                              # 工具的真实返回
                print(f"    [工具返回] {str(m.content)[:80]}")  # 上述条件至少有一个不成立

        print(f"客服:{result['messages'][-1].content}")
        print(f"     [意图={result.get('intent')} | 历史 {len(result['messages'])}条]")

    conn.close()
    print("\n已退出。记忆已存进 checkpoint.db。")

if __name__=="__main__":
    main()    
 












