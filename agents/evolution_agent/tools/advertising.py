from __future__ import annotations

from collections import defaultdict
from typing import Any
import sys
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from typing import Any
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from agents.evolution_agent.agent_init import LayerAgentState
from server.is_execute_tool import update_state_machine, is_tool_ok, StateMachineScheduler

STATE_PENDING = 0       # 未开始
STATE_RUNNING = 1       # 执行中
STATE_COMPLETED = 2     # 已完成
STATE_ERROR = 3         # 异常

@tool
def new_user_recommend(
    runtime: ToolRuntime[None, LayerAgentState],
) -> Command:
    """
    为新用户推荐商品。
    """

    ok , v = is_tool_ok(
        runtime.state["state_machine"],
        "new_user_recommend",
    )
    
    if not ok:
        return v
    
    
    tool_result = "为新用户推荐商品。"
    
    return Command(
        update={

            # 更新状态机
            "state_machine": update_state_machine(
                runtime.state["state_machine"],
                "new_user_recommend",
                tool_result,
            ),

            # ToolMessage 给 LLM 看
            "messages": [
                ToolMessage(
                    content=(
                        "new_user_recommend 工具已经执行完成。"
                        "工具状态已更新为 state=2。"
                        "请继续执行状态机中的下一个工具，"
                        "不要重复调用 new_user_recommend。"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
    
@tool
def promotion_recommend(runtime: ToolRuntime[None, LayerAgentState]) -> Command:
    """
    根据促销活动为用户推荐商品。
    """
    scheduler = StateMachineScheduler(runtime.state["state_machine"])
    res = scheduler.resolve("promotion_recommend")
    
    """
    res 返回类型
    {'requested_tool': 'generate_response', 'requested_layer': 'layer_3', 'current_layer': 'layer_3', 'layer_state': 0, 'executable_tools': ['generate_response']}
    """
    
    executable_tool = res.get("executable_tools")[0]
    
    if res.get("requested_tool") == executable_tool:
        # 需要修改 state状态
        
        
        
        
        return "根据促销活动为用户推荐商品。"


    return f"工具调用顺序错误，请先调用 {executable_tool} 工具。"
