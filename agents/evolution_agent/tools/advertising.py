from __future__ import annotations

from collections import defaultdict
from typing import Any
import sys
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from typing import Any
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from evolution_agent.agent_init import LayerAgentState
from evolution_agent.tools.is_execute_tool import StateMachineScheduler

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

    input("进入了 new_user_recommend，点我继续")

    # ==========================================
    # 1. 获取当前状态机
    # ==========================================

    scheduler = StateMachineScheduler(
        runtime.state["state_machine"]
    )

    res = scheduler.resolve("new_user_recommend")

    """
        res 示例：

        {
            "requested_tool": "new_user_recommend",
            "requested_layer": "layer_2",
            "current_layer": "layer_2",
            "layer_state": 0,
            "executable_tools": [
                "new_user_recommend"
            ]
        }
    """

    executable_tools = res.get(
        "executable_tools",
        []
    )

    # ==========================================
    # 2. 当前没有可执行工具
    # ==========================================

    if not executable_tools:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="当前没有可执行工具。",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    executable_tool = executable_tools[0]

    # ==========================================
    # 3. 检查执行顺序
    # ==========================================

    if res.get("requested_tool") != executable_tool:

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "工具调用顺序错误，"
                            f"请先调用 {executable_tool} 工具。"
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    # ==========================================
    # 4. 执行业务逻辑
    # ==========================================

    input(
        "恭喜你成功调用了 "
        "new_user_recommend 工具。"
    )

    recommend_result = {
        "message": "为新用户推荐商品。",
        "status": "success",
    }

    # ==========================================
    # 5. 复制 state_machine
    # ==========================================

    new_state_machine = deepcopy(
        runtime.state["state_machine"]
    )

    # ==========================================
    # 6. 更新工具状态
    # ==========================================

    tool_mapping = new_state_machine.get(
        "tool_mapping",
        {}
    )

    tool_meta = tool_mapping.get(
        "new_user_recommend"
    )

    if tool_meta is not None:

        # 标记工具已经完成
        tool_meta["state"] = 2

        # 保存工具执行结果
        tool_meta["result"] = recommend_result

    # ==========================================
    # 7. 更新 Layer 状态
    # ==========================================

    layer_name = res.get("requested_layer")

    layer_mapping = new_state_machine.get(
        "layer",
        {}
    )

    if layer_name:

        layer_meta = layer_mapping.get(
            layer_name
        )

        if layer_meta is not None:

            # 如果这一层只有当前一个工具，
            # 可以直接认为 layer 完成
            layer_meta["state"] = 2

    # ==========================================
    # 8. Command 写回 AgentState
    # ==========================================

    return Command(
        update={

            # 更新状态机
            "state_machine": new_state_machine,

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
def promotion_recommend(runtime: ToolRuntime[None, LayerAgentState]) -> str:
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
