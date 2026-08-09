from __future__ import annotations

from typing import Any
from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from langchain_core.globals import set_debug

set_debug(True)
# ============================================================
# 1. 定义三层 AgentState
# ============================================================

class ThreeLayerAgentState(AgentState):
    """
    LangChain 默认 AgentState 本身已经包含 messages。

    我们额外增加三层结构：
        layer_1 -> Tool1 的结果
        layer_2 -> Tool2 的结果
        layer_3 -> Tool3 的结果
    """

    layer_1: NotRequired[dict[str, Any]]
    layer_2: NotRequired[dict[str, Any]]
    layer_3: NotRequired[dict[str, Any]]


# ============================================================
# 2. Tool 1
# ============================================================

@tool
def analyze_user_input(
    runtime: ToolRuntime[None, ThreeLayerAgentState],
) -> Command:
    """
    第一个工具。

    必须最先执行。
    负责解析用户输入，并把结果写入 layer_1。
    """

    # ----------------------------------------
    # 从 AgentState 获取用户输入
    # ----------------------------------------

    messages = runtime.state["messages"]

    user_input = ""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            user_input = str(message.content)
            break

    # ----------------------------------------
    # 模拟 Tool 1 的处理
    # ----------------------------------------

    result = {
        "tool": "analyze_user_input",
        "original_input": user_input,
        "intent": "查询商品信息",
        "keywords": [
            "手机",
            "价格",
            "推荐",
        ],
    }
    
    print("\n========== TOOL 1 ==========")
    print(result)

    # ----------------------------------------
    # 写入 AgentState.layer_1
    # ----------------------------------------

    return Command(
        update={
            "layer_1": result,

            # ToolMessage 是给模型看的
            "messages": [
                ToolMessage(
                    content=(
                        "Tool1 已完成用户意图解析。"
                        "结果已经写入 AgentState.layer_1。"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# ============================================================
# 3. Tool 2
# ============================================================

@tool
def retrieve_products(
    runtime: ToolRuntime[None, ThreeLayerAgentState],
) -> Command:
    """
    第二个工具。

    必须在 analyze_user_input 之后执行。

    从 layer_1 获取 Tool1 的结果，
    执行商品检索，
    然后把结果写入 layer_2。
    """

    # ----------------------------------------
    # 获取第一层状态
    # ----------------------------------------

    layer_1 = runtime.state.get("layer_1")

    if not layer_1:
        raise RuntimeError(
            "layer_1 不存在，必须先执行 analyze_user_input"
        )

    keywords = layer_1.get("keywords", [])

    # ----------------------------------------
    # 模拟 ES / Milvus 检索
    # ----------------------------------------

    result = {
        "tool": "retrieve_products",
        "query_keywords": keywords,
        "products": [
            {
                "id": 1001,
                "name": "小米手机",
                "price": 2999,
                "score": 0.92,
            },
            {
                "id": 1002,
                "name": "荣耀手机",
                "price": 3299,
                "score": 0.89,
            },
            {
                "id": 1003,
                "name": "OPPO 手机",
                "price": 2799,
                "score": 0.85,
            },
        ],
    }

    print("\n========== TOOL 2 ==========")
    print("读取 layer_1：")
    print(layer_1)

    print("\n生成 layer_2：")
    print(result)

    # ----------------------------------------
    # 只写 layer_2
    # ----------------------------------------

    return Command(
        update={
            "layer_2": result,
            "messages": [
                ToolMessage(
                    content=(
                        "Tool2 已完成商品检索。"
                        "结果已经写入 AgentState.layer_2。"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# ============================================================
# 4. Tool 3
# ============================================================

@tool
def rank_products(
    runtime: ToolRuntime[None, ThreeLayerAgentState],
) -> Command:
    """
    第三个工具。

    必须在 retrieve_products 之后执行。

    读取 layer_1 + layer_2，
    做最终排序，
    然后写入 layer_3。
    """

    layer_1 = runtime.state.get("layer_1")
    layer_2 = runtime.state.get("layer_2")

    if not layer_1:
        raise RuntimeError(
            "layer_1 不存在"
        )

    if not layer_2:
        raise RuntimeError(
            "layer_2 不存在"
        )

    products = layer_2.get("products", [])

    # ----------------------------------------
    # 模拟推荐排序
    # ----------------------------------------

    ranked_products = sorted(
        products,
        key=lambda item: item["score"],
        reverse=True,
    )

    result = {
        "tool": "rank_products",
        "user_intent": layer_1["intent"],
        "ranking_strategy": "按照推荐得分降序",
        "ranked_products": ranked_products,
        "best_product": (
            ranked_products[0]
            if ranked_products
            else None
        ),
    }

    print("\n========== TOOL 3 ==========")

    print("读取 layer_1：")
    print(layer_1)

    print("\n读取 layer_2：")
    print(layer_2)

    print("\n生成 layer_3：")
    print(result)

    # ----------------------------------------
    # 写入第三层
    # ----------------------------------------

    return Command(
        update={
            "layer_3": result,
            "messages": [
                ToolMessage(
                    content=(
                        "Tool3 已完成商品排序。"
                        "结果已经写入 AgentState.layer_3。"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# ============================================================
# 5. 创建模型
# ============================================================
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key="sk-EvGLrMBijuYrWTRfHSYEr5qGATNAfPBJ1q1l9WtwXHyzQ3ee",  # 替换成你的真实API Key
    base_url="https://api.uiuihao.com/v1"      # openai官方接口地址；如果是中转代理就填你的代理url
)

# ============================================================
# 6. 创建 Agent
# ============================================================

agent = create_agent(
    model=model,

    tools=[
        analyze_user_input,
        retrieve_products,
        rank_products,
    ],

    # 重点：
    # 告诉 LangChain 使用我们自己定义的 AgentState
    state_schema=ThreeLayerAgentState,

    system_prompt="""
你是一个商品推荐 Agent。

对于用户的商品推荐请求，必须严格按照下面顺序执行：

第一步：
调用 analyze_user_input
解析用户需求。

第二步：
等待 analyze_user_input 完成之后，
调用 retrieve_products。

第三步：
等待 retrieve_products 完成之后，
调用 rank_products。

禁止跳过任何步骤。
禁止同时调用多个工具。

执行顺序必须严格为：

analyze_user_input
    ↓
retrieve_products
    ↓
rank_products

三个工具全部完成后，
根据最终结果回答用户。
""",
)


# ============================================================
# 7. 执行
# ============================================================

if __name__ == "__main__":

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "给我推荐一款3000元左右的手机",
                }
            ]
        }
    )

    # ========================================================
    # 查看 AgentState
    # ========================================================

    print("\n\n================================")
    print("最终 AgentState")
    print("================================")

    print("\n【第一层 layer_1】")
    print(result.get("layer_1"))

    print("\n【第二层 layer_2】")
    print(result.get("layer_2"))

    print("\n【第三层 layer_3】")
    print(result.get("layer_3"))

    print("\n【最终模型回答】")

    
    final_message = result["messages"][-1]

    print(final_message.content)
    print(111)
    
    print(result)