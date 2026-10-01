# agent/audit.py
# 审批审计表：谁、什么时候、批了哪一笔、执行结果是什么
# 独立成一张表的理由：这是「事后追责」的唯一凭据，不能只存在聊天记录里
# D29 改动：① 新增 pending 表 —— HTTP 层只有 ticket_id，得靠它反查 thread_id
#          ② 所有连接显式 close（with closing），不再每次泄漏一个句柄
#          ③ 开 WAL：服务化后多个请求同时读写，默认模式容易撞锁



import sqlite3
from contextlib import closing  # sqlite3.Connection 他没法自动关闭
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DB = ROOT / "audit.db"

def _conn() -> sqlite3.Connection: # 定义一个函数返回迷你数据库连接
    """每次用库都新开一个连接。

    为什么服务化后反而要「每次新开」：
    SQLite 连接不贵，而一个共享连接要自己处理多线程 + 事务边界，很容易出诡异 bug。
    开新连接 = 每个请求各用各的，天然隔离；timeout=10 让并发写时最多等 10 秒再报锁。
    """
    c = sqlite3.connect(str(AUDIT_DB),timeout=10.0) # 连接数据库文件，不存在就直接建
    c.row_factory = sqlite3.Row   # 让查询结果能按列名取值：row["ticket_id"]
    return c

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    """建两张表。程序启动时调一次，重复调也不会出错。

    ★ 注意 with closing(...) as c 不会替你 commit —— 它只管关连接。
      而 with _conn() as c 会 commit（sqlite3 连接的上下文协议），但不 close。
      想「既 commit 又 close」，就 closing + 手动 c.commit()，一个都不能少。
    """
    with closing(_conn()) as c:
        c.execute("PRAGMA journal_mode=WAL;")  # 写不读档，服务端标配
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS approvals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,  
                ticket_id    TEXT,  
                thread_id    TEXT,  
                action       TEXT,  
                order_id     TEXT,  
                summary      TEXT,  
                requested_at TEXT,  
                decided_at   TEXT,  
                decision     TEXT,  
                operator     TEXT,  
                note         TEXT,  
                result       TEXT
            )
            """
        )
# 自增主键1、2、3 # 工单号 # langgraph的线程ID # 要执行的动作 # 涉及的单号  # 给审批人看的摘要  # Agent 发起审批的时间 # 审批人做决定的时间 # 审批人做决定的时间 # 决定结果：approve（批）/ reject（拒）# 谁点的按钮（审批人账号名）

        # ★ D29 新增：待办登记表。ticket_id 当主键，一行就是一张没批的单子
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS pending (
                ticket_id    TEXT PRIMARY KEY,
                thread_id    TEXT,
                action       TEXT,
                order_id     TEXT,
                summary      TEXT,
                requested_at TEXT
            )
            """
        )
        c.commit()

# 这两个表不一样是因为 上边的是档案 这里的数便签 且没有的五个值是是审批之后才有的值 待办还没到审批

# ---------- 待办登记（D29 新增） ----------

def register_pending(payload:dict) -> None:  # 登记表
    """Agent 一挂起就登记一张待办单，HTTP 层才有办法「拿 ticket_id 找回会话」。

    ★ 为什么是 INSERT OR IGNORE 而不是 INSERT：
      approval_gate 节点在恢复时会从头重跑一遍，这个函数会被调用两次。
      ticket_id 是从 state 里读的、重跑时值不变，第二次插入直接被主键挡掉 —— 天然幂等。
    """
    
    with closing(_conn()) as c:  # 这里是新建一个连接
        c.execute(
            "INSERT OR IGNORE INTO pending"
            " (ticket_id, thread_id, action, order_id, summary, requested_at)"
            " VALUES (?,?,?,?,?,?)",
            (
                payload.get("ticket_id"),
                payload.get("thread_id"),
                payload.get("action"),
                payload.get("order_id"),
                payload.get("summary"),
                payload.get("requested_at"),
            ),
        )
        c.commit()


def get_pending(ticket_id:str):
    """按单号查待办。查不到返回 None（说明批过了或者单号错）。"""
    with closing(_conn()) as c:
        return c.execute(
            "SELECT * FROM pending WHERE ticket_id = ?", (ticket_id,)
        ).fetchone() # 这里只要单号



def list_pending(limit: int = 20) -> list:
    with closing(_conn()) as c:
        return c.execute(
            "SELECT * FROM pending ORDER BY requested_at DESC LIMIT ?", (limit,)
        ).fetchall()  # 降序排列


def drop_pending(ticket_id:str) -> None:
    """审批处理完，把待办划掉。"""
    with closing(_conn()) as c:
        c.execute("DELETE FROM pending WHERE ticket_id = ?", (ticket_id,))
        c.commit()



def log_decision(payload: dict, approval: dict) -> None:  # 审批后接受两个字典
    """审批人点完「批/拒」之后写一条。此时还不知道执行结果，result 留空。"""
    with closing(_conn()) as c:
        c.execute(
            "INSERT INTO approvals (ticket_id, thread_id, action, order_id, summary,"
            " requested_at, decided_at, decision, operator, note, result)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",   # 占位符
            (
                payload.get("ticket_id"),
                payload.get("thread_id"),
                payload.get("action"),
                payload.get("order_id"),
                payload.get("summary"),
                payload.get("requested_at"),
                _now(),
                approval.get("status"),
                approval.get("operator"),
                approval.get("note"),
                "",
            ),
        )
        c.commit()
# 这里对应11个占位符 前六个是agent发起的申请  后四个是人工决定

def log_execution(ticket_id: str, result: str) -> None:  # 动作真的执行完了 在调他
    """真执行完了，把结果补回同一条记录上。"""
    with closing(_conn()) as c:
        c.execute(
            "UPDATE approvals SET result = ? WHERE ticket_id = ?", (result, ticket_id)
        )
        c.commit()
# UPDATE 语句：找到 ticket_id 等于传入值的那行，把它的 result 列改成新值。两个 ? 对应 (result, ticket_id)，顺序不能反。
# 为什么更新记录 上边的退款操作在这步之前是空白的 只有到这步才知道成功了 因此要覆盖上边的记录

def list_approvals(limit: int = 10) -> list[sqlite3.Row]:
    with closing(_conn()) as c:
        return c.execute(
            "SELECT * FROM approvals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

# 查记录

























