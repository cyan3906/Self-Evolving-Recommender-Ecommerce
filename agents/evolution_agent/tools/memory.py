from typing import List, Dict
import sys
import math
from pathlib import Path
from langchain.tools import tool, ToolRuntime


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print(PROJECT_ROOT)
# from tools.product_data import products # 获取模拟的产品数据
from langchain.messages import ToolMessage
from langgraph.types import Command
from agents.evolution_agent.agent_init import LayerAgentState
from server.is_execute_tool import update_state_machine, is_tool_ok, StateMachineScheduler
from agents.evolution_agent.tools.rag import es_search


def softmax_normalize(memory_data: Dict) -> Dict:
    memories = memory_data.get("memories", [])
    if not memories:
        return memory_data.copy()
    confs = [m["confidence"] for m in memories]
    exp_list = [math.exp(c) for c in confs]
    sum_exp = sum(exp_list)
    new_mem = memory_data.copy()
    new_mem["memories"] = []
    for idx, m in enumerate(memories):
        item = m.copy()
        item["confidence_softmax"] = exp_list[idx] / sum_exp
        new_mem["memories"].append(item)
    return new_mem

def deduplicate_by_key(item_list: list[dict], unique_key="product_id") -> list[dict]:
    temp = {}
    for item in item_list:
        k = item[unique_key]
        # 后面出现的覆盖前面，保留最新
        temp[k] = item
    return list(temp.values())

@tool
def short_term_memory(runtime: ToolRuntime[None, LayerAgentState]) -> Command:
    """
    返回短期记忆，基于用户短期固化偏好标签召回商品。
    """
    
    ok , v = is_tool_ok(
        runtime.state["state_machine"],
        "short_term_memory",
    )
    if not ok:
        return v
    
    
    search_num = 10
    
    memory_data = {
        "user_id": "user_001",
        "scope": "short_term",
        "memories": [
            {
                "key": "brand:huawei",
                "confidence": 0.85975,
                "event_nums": 3
            },
            {
                "key": "category:手机",
                "confidence": 0.85975,
                "event_nums": 3
            }
        ]
    }
    memory_data = softmax_normalize(memory_data) # 归一化 confidence
    items = memory_data["memories"]
    
    tool_result = []
    
    for item in items:
        who = item["key"].split(":")[1]
        _ = es_search(who, item["confidence_softmax"] * search_num)
        if len(_) > 0:
            tool_result.extend(_)
    
    tool_result = deduplicate_by_key(tool_result)
    
    return Command(
        update={
            # --------------------------
            # 保存更新后的状态机
            # --------------------------
            "state_machine": update_state_machine(
                runtime.state["state_machine"],
                "short_term_memory",
                tool_result,
            ),
            # --------------------------
            # 给 LLM 看的 ToolMessage
            # --------------------------
            "messages": [
                ToolMessage(
                    content=(
                        "short_term_memory 已执行完成。"
                        "请不要重复调用 short_term_memory。"
                        f"短期记忆共获得 {len(tool_result)} 条结果。"
                        "完整结果已保存到 "
                        "AgentState.short_term_memory_layer。"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

@tool
def long_term_memory(runtime: ToolRuntime[None, LayerAgentState]) -> Command:
    """
    返回长期记忆，基于用户长期固化偏好标签召回商品。
    """
    search_num = 10

    memory_data = {
        "user_id": "user_001",
        "scope": "long_term",
        "memories": [
            {
                "key": "brand:huawei",
                "confidence": 0.85975,
                "event_nums": 3
            },
            {
                "key": "category:手机",
                "confidence": 0.85975,
                "event_nums": 3
            }
        ]
    }

    memory_data = softmax_normalize(memory_data)
    items = memory_data["memories"]

    ans = []
    for item in items:
        who = item["key"].split(":")[1]
        _ = es_search(who, item["confidence_softmax"] * search_num)
        if len(_) > 0:
            ans.extend(_)

    return deduplicate_by_key(ans)



if __name__ == "__main__":
    print(short_term_memory())
    # print(long_term_memory())
