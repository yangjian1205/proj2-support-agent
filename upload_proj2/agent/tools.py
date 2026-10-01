# agent/tools.py
# 三件事：① 一份假业务数据库（省掉真系统，P2 不碰外部依赖）
#          ② 三个工具函数：查订单 / 查物流 / 查库存
#          ③ 越权校验：别人的订单不给看（安全边界的最内层）


from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# ========== ① 假数据库：真实系统里这层是订单服务 / 物流服务 / 库存服务 ==========


ORDERS = {
    "A1001": {
        "user_id": "u-1001", "status": "已发货", "amount": 299.0,
        "items": "硅胶烤垫 x2", "created_at": "2026-09-20 14:32",
        "address": "浙江省杭州市西湖区文三路 100 号 3 幢 501",
    },
    "A1002": {
        "user_id": "u-1001", "status": "待发货", "amount": 158.0,
        "items": "竹砧板 x1", "created_at": "2026-09-24 20:05",
        "address": "浙江省杭州市余杭区良睦路 999 号 2 单元 1203",
    },
    "A2001": {
        "user_id": "u-1002", "status": "待付款", "amount": 88.0,
        "items": "LED 梳妆镜灯 x1", "created_at": "2026-09-22 09:10",
        "address": "广东省广州市天河区天河路 385 号 1806",
    },
}


LOGISTICS = {
    "A1001": (
        "2026-09-21 10:20 已揽收（广州仓）"
        " → 2026-09-22 18:40 到达杭州转运中心"
        " → 2026-09-24 08:15 派送中（杭州西湖区）"
    ),
    "A1002": "2026-09-24 20:05 已下单，等待仓库发货",
}

INVENTORY = {
    "硅胶烤垫": {"sku": "SKU-1001", "stock": 128, "warehouse": "广州仓"},
    "竹砧板":   {"sku": "SKU-1002", "stock": 0,   "warehouse": "广州仓"},
    "LED 梳妆镜灯": {"sku": "SKU-1003", "stock": 42, "warehouse": "杭州仓"},
}

# ========== ② 三个工具 ==========
# 注意每个函数的 docstring：它不是写给人看的注释，
# 它是「菜单上的菜名和介绍」—— 模型就是根据这段话决定要不要调这个工具。
# 所以 docstring 必须写清楚：这个工具能查什么、参数长什么样。


@tool
def get_order(order_id:str,state:Annotated[dict,InjectedState])-> str:   
    """查询订单的状态、金额、商品和下单时间。order_id 是订单号，例如 A1001。"""
    # 这个参数类型是 dict，但它带有 InjectedState 标记，不要由 LLM 生成，而是由 LangGraph 在调用工具时自动从当前图状态 state 注入进来。
    uid = state.get("user_id")      # 从共享状态里拿当前用户
    o = ORDERS.get(order_id.strip().upper())  # 容错：用户可能打小写带空格
    if not o:
        return f"没有找到订单 {order_id}。"
    if o["user_id"] != uid:    # ★ 越权拦截：这一层必须在代码里做
        return f"订单 {order_id}不属于当前用户，无法查看。"
    return(
        f"订单 {order_id}:状态={o['status']},金额=￥{o['amount']}"
        f"商品={o['items']}，下单时间={o['created_at']}"
    )



@tool
def track_logistics(order_id:str,state:Annotated[dict,InjectedState])-> str: 
    """查询订单的物流轨迹。order_id 是订单号。用户问「到哪了」「什么时候到」时用这个。"""
    uid = state.get("user_id")  # 用户名
    real_id = order_id.strip().upper()  # 真正查表的订单号
    o = ORDERS.get(real_id) # 也就是用户要查的订单全部大写在这里查
    if not o or o["user_id"] != uid: # 订单不存在或者订单不属于你
        return f"订单 {order_id} 不存在或不属于当前用户。"
    return LOGISTICS.get(real_id, f"订单 {order_id} 还没有物流信息。")
    # 物流表里有就出，没有就返回后面的字符串


@tool
def get_inventory(product_name:str) -> str:
    """查询商品库存和发货仓。product_name 是商品名称，例如「硅胶烤垫」。库存是公开信息，不需要登录。"""
    for name,info in INVENTORY.items():  # name商品名 info商品值 键值对
        if name in product_name or product_name in name:  # 名字相等就算成
            if info["stock"] == 0:
                return f"{name} ({info['sku']})当前缺货，{info['warehouse']}暂无库存。"
            return f"{name}（{info['sku']}）还剩 {info['stock']} 件，发货仓：{info['warehouse']}。"
    return f"没有找到商品「{product_name}」，可以换个说法再问我。"

# ③ 打包成一个列表，后面 graph.py 要拿它去 bind_tools 和喂给 ToolNode
TOOLS = [get_order, track_logistics, get_inventory]

# 不需要用户身份，所以参数里没有 InjectedState。

# ========== ③ ★ D28 新增：两个写操作（普通函数，不是 @tool） ==========
WRITE_LOG: list[str] = []      # 内存里留一份执行痕迹，方便验收时打印


def do_refund(order_id: str, amount: float, uid: str) -> str:
    """真退款。调用前必须已经拿到人工审批 —— 这里再做一次越权兜底。"""
    o = ORDERS.get(order_id.strip().upper())
    if not o:
        return f"退款失败：订单 {order_id} 不存在。"
    if o["user_id"] != uid:
        return f"退款失败：订单 {order_id} 不属于当前用户。"
    if amount <= 0 or amount > o["amount"]:
        return f"退款失败：金额￥{amount} 不合法(实付￥{o['amount']})。"
    o["status"] = "已退款"
    msg = f"已为订单 {order_id} 退款 ￥{amount}，预计 1-3 个工作日到账。"
    WRITE_LOG.append(msg)
    return msg

def do_change_address(order_id: str, new_address: str, uid: str) -> str:
    """真改地址。同样必须已经审批过。"""
    o = ORDERS.get(order_id.strip().upper())
    if not o:
        return f"修改失败：订单 {order_id} 不存在。"
    if o["user_id"] != uid:
        return f"修改失败：订单 {order_id} 不属于当前用户。"
    if o["status"] in ("已发货", "已退款"):
        return f"修改失败：订单 {order_id} 当前状态为「{o['status']}」，不能再改地址。"
    if not new_address.strip():
        return "修改失败：新地址为空。"
    old = o["address"]
    o["address"] = new_address.strip()
    msg = f"订单 {order_id} 的收货地址已从「{old}」改为「{new_address}」。"
    WRITE_LOG.append(msg)
    return msg
                       
#  这里不加工具是因为 模型有概率不听话 审批不能靠模型自觉




