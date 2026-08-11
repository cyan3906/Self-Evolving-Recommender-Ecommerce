from __future__ import annotations
import os,sys
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from langchain.tools import tool, ToolRuntime
from agent_init import LayerAgentState
from typing import Any


# ============================================================
# 状态定义
# ============================================================

STATE_PENDING = 0       # 未开始
STATE_RUNNING = 1       # 执行中
STATE_COMPLETED = 2     # 已完成
STATE_ERROR = 3         # 异常


# ============================================================
# State Machine 示例
# ============================================================

state_machine: dict[str, Any] = {

    # ========================================================
    # Layer 状态
    # ========================================================

    "layer_1": {
        "state": STATE_PENDING,

        "tools": [
            "hybrid_retrieval",
            "new_user_recommend",
        ],

        # layer_1 没有上层
        "parent": "root",
    },

    "layer_2": {
        "state": STATE_PENDING,

        "tools": [
            "rerank",
            "filter_product",
        ],

        # layer_2 必须等 layer_1 完成
        "parent": "layer_1",
    },

    "layer_3": {
        "state": STATE_PENDING,

        "tools": [
            "generate_response",
        ],

        # layer_3 必须等 layer_2 完成
        "parent": "layer_2",
    },

    # ========================================================
    # Tool 状态
    # ========================================================

    "tool_mapping": {

        # --------------------------
        # layer_1
        # --------------------------

        "hybrid_retrieval": {
            "layer": "layer_1",
            "state": STATE_PENDING,

            # layer 内的第一个工具
            "parent": "root",
        },

        "new_user_recommend": {
            "layer": "layer_1",
            "state": STATE_PENDING,

            # 必须等 hybrid_retrieval 完成
            "parent": "hybrid_retrieval",
        },

        # --------------------------
        # layer_2
        # --------------------------

        "rerank": {
            "layer": "layer_2",
            "state": STATE_PENDING,
            "parent": "root",
        },

        "filter_product": {
            "layer": "layer_2",
            "state": STATE_PENDING,
            "parent": "rerank",
        },

        # --------------------------
        # layer_3
        # --------------------------

        "generate_response": {
            "layer": "layer_3",
            "state": STATE_PENDING,
            "parent": "root",
        },
    },
}


# ============================================================
# State Machine Scheduler
# ============================================================


class StateMachineScheduler:
    """
    Agent 状态机调度器。

    调度规则：

    1. 模型想调用一个 Tool
    2. 获取 Tool 所属 Layer
    3. 检查 Layer.parent
    4. 如果父 Layer 没完成：
         递归向上寻找
    5. 找到当前真正需要处理的 Layer
    6. 在该 Layer 内检查 Tool.parent
    7. 找到当前真正可以执行的 Tool

    Layer 和 Tool 是两套独立的依赖关系：

        Layer:
            layer_1
                ↓
            layer_2
                ↓
            layer_3

        Tool:
            hybrid_retrieval
                    ↓
            new_user_recommend
    """

    def __init__(
        self,
        runtime_state: dict[str, Any],
    ):
        self.runtime_state = runtime_state

    # ========================================================
    # 对外主入口
    # ========================================================

    def resolve(
        self,
        target_tool: str,
    ) -> dict[str, Any]:
        """
        根据模型想调用的 target_tool，
        找到当前真正应该执行的 Layer 和 Tool。

        例如：

            模型想调用：
                generate_response

            但是：
                layer_1 = 0
                layer_2 = 0
                layer_3 = 0

            那么实际返回：

                current_layer = layer_1
                executable_tools = ["hybrid_retrieval"]
        """

        # ----------------------------------------------------
        # 1. 获取目标 Tool
        # ----------------------------------------------------

        tool_meta = self._get_tool_meta(target_tool)

        requested_layer = tool_meta["layer"]

        # ----------------------------------------------------
        # 2. 找当前真正应该处理的 Layer
        # ----------------------------------------------------

        current_layer = self.resolve_layer(
            requested_layer
        )

        # ----------------------------------------------------
        # 3. 找当前 Layer 真正可以执行的 Tool
        # ----------------------------------------------------

        if current_layer == requested_layer:

            #
            # 模型请求的 Tool 就在当前 Layer
            #
            # 优先沿 target_tool 的 parent 链进行解析
            #

            resolved_tool = self.resolve_tool(
                target_tool
            )

            executable_tools = []

            if resolved_tool is not None:
                executable_tools.append(
                    resolved_tool
                )

            #
            # 如果已经没有目标链路需要执行，
            # 再检查这个 Layer 是否还有别的可执行 Tool
            #
            if not executable_tools:
                executable_tools = (
                    self.get_executable_tools(
                        current_layer
                    )
                )

        else:

            #
            # 当前应该处理的是更上层 Layer
            #
            # target_tool 还没有资格执行
            #
            executable_tools = (
                self.get_executable_tools(
                    current_layer
                )
            )

        return {
            "requested_tool": target_tool,
            "requested_layer": requested_layer,

            "current_layer": current_layer,

            "layer_state": self.runtime_state[
                current_layer
            ].get(
                "state",
                STATE_PENDING,
            ),

            "executable_tools": executable_tools,
        }

    # ========================================================
    # Layer 递归解析
    # ========================================================

    def resolve_layer(
        self,
        layer_name: str,
        visited: set[str] | None = None,
    ) -> str:
        """
        递归寻找当前真正应该处理的 Layer。

        举例：

            layer_1 state=2
            layer_2 state=0
            layer_3 state=0

        模型准备调用 layer_3 的工具。

        检查：

            layer_3.parent = layer_2

        layer_2 没完成：

            向上递归 layer_2

        检查：

            layer_2.parent = layer_1

        layer_1 已完成：

            所以当前应该执行 layer_2

        最终：

            return "layer_2"
        """

        if visited is None:
            visited = set()

        # ----------------------------------------------------
        # 防止 Layer 环形依赖
        # ----------------------------------------------------

        if layer_name in visited:
            raise RuntimeError(
                f"检测到 Layer 循环依赖: {layer_name}"
            )

        visited.add(layer_name)

        # ----------------------------------------------------
        # 获取 Layer
        # ----------------------------------------------------

        layer = self._get_layer(layer_name)

        parent_layer = layer.get(
            "parent",
            "root",
        )

        # ----------------------------------------------------
        # 已经到最顶层
        # ----------------------------------------------------

        if parent_layer in (
            None,
            "",
            "root",
        ):
            return layer_name

        # ----------------------------------------------------
        # 获取父 Layer
        # ----------------------------------------------------

        parent = self._get_layer(
            parent_layer
        )

        parent_state = parent.get(
            "state",
            STATE_PENDING,
        )

        # ----------------------------------------------------
        # 父 Layer 已完成
        #
        # 那当前 Layer 就可以开始了
        # ----------------------------------------------------

        if parent_state == STATE_COMPLETED:
            return layer_name

        # ----------------------------------------------------
        # 父 Layer 没完成
        #
        # 当前 Layer 不允许执行
        #
        # 递归检查父 Layer
        # ----------------------------------------------------

        return self.resolve_layer(
            layer_name=parent_layer,
            visited=visited,
        )

    # ========================================================
    # Tool 递归解析
    # ========================================================

    def resolve_tool(
        self,
        tool_name: str,
        visited: set[str] | None = None,
    ) -> str | None:
        """
        递归寻找当前 Tool 链真正应该执行的节点。

        例如：

            tool_a
              ↓
            tool_b
              ↓
            tool_c

        模型想执行：

            tool_c

        当前：

            tool_a = 2
            tool_b = 0
            tool_c = 0

        那么：

            return tool_b

        --------------------------------

        如果：

            tool_a = 2
            tool_b = 2
            tool_c = 0

        那么：

            return tool_c
        """

        if visited is None:
            visited = set()

        # ----------------------------------------------------
        # 防止 Tool 环形依赖
        # ----------------------------------------------------

        if tool_name in visited:
            raise RuntimeError(
                f"检测到 Tool 循环依赖: {tool_name}"
            )

        visited.add(tool_name)

        tool_meta = self._get_tool_meta(
            tool_name
        )

        tool_state = tool_meta.get(
            "state",
            STATE_PENDING,
        )

        parent_tool = tool_meta.get(
            "parent",
            "root",
        )

        # ----------------------------------------------------
        # 当前 Tool 已经完成
        # ----------------------------------------------------

        if tool_state == STATE_COMPLETED:
            return None

        # ----------------------------------------------------
        # 当前 Tool 正在执行
        #
        # 不应该重复执行
        # ----------------------------------------------------

        if tool_state == STATE_RUNNING:
            return None

        # ----------------------------------------------------
        # 当前 Tool 没有父节点
        #
        # 当前 Tool 可以直接执行
        # ----------------------------------------------------

        if parent_tool in (
            None,
            "",
            "root",
        ):
            return tool_name

        # ----------------------------------------------------
        # 获取 Parent Tool
        # ----------------------------------------------------

        parent_meta = self._get_tool_meta(
            parent_tool
        )

        # ----------------------------------------------------
        # 校验父子 Tool 是否属于同一个 Layer
        # ----------------------------------------------------

        current_layer = tool_meta.get(
            "layer"
        )

        parent_layer = parent_meta.get(
            "layer"
        )

        if current_layer != parent_layer:
            raise ValueError(
                f"Tool parent 必须位于同一个 Layer："
                f"{tool_name}({current_layer}) -> "
                f"{parent_tool}({parent_layer})"
            )

        parent_state = parent_meta.get(
            "state",
            STATE_PENDING,
        )

        # ----------------------------------------------------
        # Parent 已经完成
        #
        # 当前 Tool 可以执行
        # ----------------------------------------------------

        if parent_state == STATE_COMPLETED:
            return tool_name

        # ----------------------------------------------------
        # Parent 未完成
        #
        # 递归向前寻找
        # ----------------------------------------------------

        return self.resolve_tool(
            tool_name=parent_tool,
            visited=visited,
        )

    # ========================================================
    # 获取 Layer 中所有当前可以执行的 Tool
    # ========================================================

    def get_executable_tools(
        self,
        layer_name: str,
    ) -> list[str]:
        """
        获取当前 Layer 所有可执行 Tool。

        支持并行 Tool：

                    tool_a
                    /    \\
               tool_b   tool_c

        tool_a 完成后：

            tool_b
            tool_c

        会同时返回：

            ["tool_b", "tool_c"]
        """

        layer = self._get_layer(
            layer_name
        )

        layer_tools = layer.get(
            "tools",
            [],
        )

        executable_tools: list[str] = []

        for tool_name in layer_tools:

            tool_meta = self._get_tool_meta(
                tool_name
            )

            state = tool_meta.get(
                "state",
                STATE_PENDING,
            )

            # ------------------------------------------------
            # 已经完成
            # ------------------------------------------------

            if state == STATE_COMPLETED:
                continue

            # ------------------------------------------------
            # 正在执行
            #
            # 防止重复调用
            # ------------------------------------------------

            if state == STATE_RUNNING:
                continue

            parent_tool = tool_meta.get(
                "parent",
                "root",
            )

            # ------------------------------------------------
            # root Tool
            # ------------------------------------------------

            if parent_tool in (
                None,
                "",
                "root",
            ):
                executable_tools.append(
                    tool_name
                )
                continue

            parent_meta = self._get_tool_meta(
                parent_tool
            )

            parent_state = parent_meta.get(
                "state",
                STATE_PENDING,
            )

            # ------------------------------------------------
            # Parent Tool 已经完成
            # ------------------------------------------------

            if parent_state == STATE_COMPLETED:
                executable_tools.append(
                    tool_name
                )

        return executable_tools

    # ========================================================
    # 修改 Tool 状态
    # ========================================================

    def mark_tool_running(
        self,
        tool_name: str,
    ) -> None:
        """
        Tool 开始执行。
        """

        tool_meta = self._get_tool_meta(
            tool_name
        )

        tool_meta["state"] = STATE_RUNNING

        layer_name = tool_meta["layer"]

        self.refresh_layer_state(
            layer_name
        )

    def mark_tool_completed(
        self,
        tool_name: str,
    ) -> None:
        """
        Tool 执行完成。
        """

        tool_meta = self._get_tool_meta(
            tool_name
        )

        tool_meta["state"] = STATE_COMPLETED

        layer_name = tool_meta["layer"]

        self.refresh_layer_state(
            layer_name
        )

    def mark_tool_error(
        self,
        tool_name: str,
    ) -> None:
        """
        Tool 执行异常。
        """

        tool_meta = self._get_tool_meta(
            tool_name
        )

        tool_meta["state"] = STATE_ERROR

        layer_name = tool_meta["layer"]

        self.refresh_layer_state(
            layer_name
        )

    def reset_tool(
        self,
        tool_name: str,
    ) -> None:
        """
        Tool 重置为未开始。

        主要用于异常重试。
        """

        tool_meta = self._get_tool_meta(
            tool_name
        )

        tool_meta["state"] = STATE_PENDING

        layer_name = tool_meta["layer"]

        self.refresh_layer_state(
            layer_name
        )

    # ========================================================
    # 自动刷新 Layer 状态
    # ========================================================

    def refresh_layer_state(
        self,
        layer_name: str,
    ) -> int:
        """
        根据当前 Layer 下所有 Tool 状态，
        自动计算 Layer.state。

        规则：

        全部 Tool = 2
            Layer = 2

        任意 Tool = 3
            Layer = 3

        任意 Tool = 1
            Layer = 1

        其他
            Layer = 0
        """

        layer = self._get_layer(
            layer_name
        )

        tools = layer.get(
            "tools",
            [],
        )

        # ----------------------------------------------------
        # 空 Layer
        # ----------------------------------------------------

        if not tools:
            layer["state"] = STATE_COMPLETED
            return STATE_COMPLETED

        states = []

        for tool_name in tools:

            tool_meta = self._get_tool_meta(
                tool_name
            )

            state = tool_meta.get(
                "state",
                STATE_PENDING,
            )

            states.append(state)

        # ----------------------------------------------------
        # 全部完成
        # ----------------------------------------------------

        if all(
            state == STATE_COMPLETED
            for state in states
        ):
            layer["state"] = STATE_COMPLETED

        # ----------------------------------------------------
        # 存在异常
        # ----------------------------------------------------

        elif any(
            state == STATE_ERROR
            for state in states
        ):
            layer["state"] = STATE_ERROR

        # ----------------------------------------------------
        # 存在执行中
        # ----------------------------------------------------

        elif any(
            state == STATE_RUNNING
            for state in states
        ):
            layer["state"] = STATE_RUNNING

        # ----------------------------------------------------
        # 其他情况
        # ----------------------------------------------------

        else:
            layer["state"] = STATE_PENDING

        return layer["state"]

    # ========================================================
    # 刷新所有 Layer
    # ========================================================

    def refresh_all_layers(
        self,
    ) -> None:
        """
        刷新 runtime_state 中所有 Layer。
        """

        for key, value in self.runtime_state.items():

            if key == "tool_mapping":
                continue

            if not isinstance(
                value,
                dict,
            ):
                continue

            if "tools" not in value:
                continue

            self.refresh_layer_state(
                key
            )

    # ========================================================
    # 判断整个 State Machine 是否完成
    # ========================================================

    def is_all_completed(
        self,
    ) -> bool:
        """
        判断所有 Layer 是否全部完成。
        """

        self.refresh_all_layers()

        for key, value in self.runtime_state.items():

            if key == "tool_mapping":
                continue

            if not isinstance(
                value,
                dict,
            ):
                continue

            if "tools" not in value:
                continue

            if (
                value.get("state")
                != STATE_COMPLETED
            ):
                return False

        return True

    # ========================================================
    # 获取当前应该工作的 Layer
    # ========================================================

    def get_current_layer(
        self,
    ) -> str | None:
        """
        从 Layer 依赖关系中，
        找到当前第一个没有完成的 Layer。

        例如：

            layer_1 = 2
            layer_2 = 0
            layer_3 = 0

        返回：

            layer_2
        """

        self.refresh_all_layers()

        layers = self._get_layers()

        for layer_name in layers:

            layer = self.runtime_state[
                layer_name
            ]

            state = layer.get(
                "state",
                STATE_PENDING,
            )

            if state == STATE_COMPLETED:
                continue

            current_layer = self.resolve_layer(
                layer_name
            )

            return current_layer

        return None

    # ========================================================
    # 内部工具函数
    # ========================================================

    def _get_tool_meta(
        self,
        tool_name: str,
    ) -> dict[str, Any]:

        tool_mapping = self.runtime_state.get(
            "tool_mapping",
            {},
        )

        tool_meta = tool_mapping.get(
            tool_name
        )

        if tool_meta is None:
            raise ValueError(
                f"State Machine 中找不到 Tool："
                f"{tool_name}"
            )

        return tool_meta

    def _get_layer(
        self,
        layer_name: str,
    ) -> dict[str, Any]:

        layer = self.runtime_state.get(
            layer_name
        )

        if layer is None:
            raise ValueError(
                f"State Machine 中找不到 Layer："
                f"{layer_name}"
            )

        if not isinstance(
            layer,
            dict,
        ):
            raise ValueError(
                f"{layer_name} 不是有效 Layer"
            )

        return layer

    def _get_layers(
        self,
    ) -> list[str]:

        layers = []

        for key, value in self.runtime_state.items():

            if key == "tool_mapping":
                continue

            if not isinstance(
                value,
                dict,
            ):
                continue

            if "tools" not in value:
                continue

            layers.append(key)

        return layers


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":

    scheduler = StateMachineScheduler(
        state_machine
    )

    # ========================================================
    # 场景 1
    #
    # Agent 一开始就想调用 layer_3 的 generate_response
    # ========================================================

    print("=" * 70)
    print("第一次：Agent 想调用 generate_response")
    print("=" * 70)

    result = scheduler.resolve(
        "generate_response"
    )

    print(result)

    """
    输出应该类似：

    {
        'requested_tool': 'generate_response',
        'requested_layer': 'layer_3',

        'current_layer': 'layer_1',

        'layer_state': 0,

        'executable_tools': [
            'hybrid_retrieval'
        ]
    }
    """

    # ========================================================
    # 执行 hybrid_retrieval
    # ========================================================

    print()
    print("=" * 70)
    print("开始执行 hybrid_retrieval")
    print("=" * 70)

    scheduler.mark_tool_running(
        "hybrid_retrieval"
    )

    print(
        state_machine["tool_mapping"][
            "hybrid_retrieval"
        ]
    )

    # 模拟执行成功

    scheduler.mark_tool_completed(
        "hybrid_retrieval"
    )

    # ========================================================
    # 第二次
    # ========================================================

    print()
    print("=" * 70)
    print("第二次：Agent 还是想调用 generate_response")
    print("=" * 70)

    result = scheduler.resolve(
        "generate_response"
    )

    print(result)

    """
    现在应该返回：

    {
        current_layer: layer_1,

        executable_tools: [
            new_user_recommend
        ]
    }
    """

    # ========================================================
    # 完成 new_user_recommend
    # ========================================================

    scheduler.mark_tool_running(
        "new_user_recommend"
    )

    scheduler.mark_tool_completed(
        "new_user_recommend"
    )

    print()
    print(
        "layer_1 state:",
        state_machine["layer_1"]["state"]
    )

    # 此时 layer_1 自动变成 2

    # ========================================================
    # 第三次
    # ========================================================

    print()
    print("=" * 70)
    print("第三次：Agent 想调用 generate_response")
    print("=" * 70)

    result = scheduler.resolve(
        "generate_response"
    )

    print(result)

    """
    现在：

        layer_1 = 2
        layer_2 = 0
        layer_3 = 0

    所以应该进入：

        layer_2

    返回：

        executable_tools = [
            rerank
        ]
    """

    # ========================================================
    # 完成 layer_2
    # ========================================================

    scheduler.mark_tool_running(
        "rerank"
    )

    scheduler.mark_tool_completed(
        "rerank"
    )

    scheduler.mark_tool_running(
        "filter_product"
    )

    scheduler.mark_tool_completed(
        "filter_product"
    )

    # ========================================================
    # 第四次
    # ========================================================

    print()
    print("=" * 70)
    print("第四次：Agent 想调用 generate_response")
    print("=" * 70)

    result = scheduler.resolve(
        "generate_response"
    )

    print(result)

    """
    现在：

        layer_1 = 2
        layer_2 = 2
        layer_3 = 0

    所以：

        current_layer = layer_3

        executable_tools = [
            generate_response
        ]
    """

    # ========================================================
    # 完成最后 Tool
    # ========================================================

    scheduler.mark_tool_running(
        "generate_response"
    )

    scheduler.mark_tool_completed(
        "generate_response"
    )

    # ========================================================
    # 查看最终状态
    # ========================================================

    print()
    print("=" * 70)
    print("最终状态")
    print("=" * 70)

    print(
        "layer_1:",
        state_machine["layer_1"]["state"]
    )

    print(
        "layer_2:",
        state_machine["layer_2"]["state"]
    )

    print(
        "layer_3:",
        state_machine["layer_3"]["state"]
    )

    print(
        "是否全部完成:",
        scheduler.is_all_completed()
    )