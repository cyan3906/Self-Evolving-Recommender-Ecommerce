from typing import List, Dict
import sys
from pathlib import Path
from langchain.tools import tool, ToolRuntime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# print(PROJECT_ROOT)
from tools.product_data import products # 获取模拟的产品数据

from typing import Any
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from agents.evolution_agent.agent_init import LayerAgentState

from server.is_execute_tool import update_state_machine, is_tool_ok

@tool
def get_top_hot_products(runtime: ToolRuntime[None, LayerAgentState], display_total: int = 20) -> Command:
    """
    根据hot热度，返回前25%高热度商品。
    :param display_total: 计划总展示商品数量
    :return: 筛选后的高热度商品列表（hot降序）
    """
    
    ok , v = is_tool_ok(
        runtime.state["state_machine"],
        "get_top_hot_products",
    )
    
    if not ok:
        return v
    
    # 1.按hot降序
    sorted_products = sorted(products, key=lambda x: x["hot"], reverse=True)
    # print(sorted_products)
    # 计算取多少条，向上取整
    take_count = int(display_total * 0.25)
    # 向上取整处理，不足1至少返回1条
    if display_total * 0.25 > take_count:
        take_count += 1
    take_count = max(1, take_count)

    # 不能超过实际商品总数
    take_count = min(take_count, len(sorted_products))

    ans = sorted_products[:take_count]

    return Command(
        update={

            # 更新状态机
            "state_machine": update_state_machine(
                runtime.state["state_machine"],
                "get_top_hot_products",
                ans,
            ),
        
            # ToolMessage 给 LLM 看
            "messages": [
                ToolMessage(
                    content=(
                        "get_top_hot_products 工具已经执行完成。"
                        "工具状态已更新为 state=2。"
                        "请继续执行状态机中的下一个工具，"
                        "不要重复调用 get_top_hot_products。"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
    

# 测试示例
if __name__ == "__main__":
    res = get_top_hot_products(display_total=20)
    print(f"取商品数量：{len(res)}")
    for item in res:
        print(f"id:{item['product_id']}, hot:{item['hot']}")