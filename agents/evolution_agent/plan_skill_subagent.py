
"""
这里其实可以封装成一个agent，
每个agent都有一个技能计划。


"""
def return_sequential_skill_plan() -> str:
    """
    返回串行技能计划。
    """
    
    plan = {
        "layer_1": ["hybrid-retrieval","new‑user‑welcome‑recommend"],
    }
    
    return plan


def return_parallel_skill_plan() -> str:
    """
    返回并行技能计划。
    """
    
    plan = {
        "layer_1": ["hybrid-retrieval"],
        "layer_2": ["new‑user‑welcome‑recommend"],
    }


def return_hybridbrid_skill_plan() -> str:
    """
    返回混合技能计划。
    """
    
    plan = {
        "layer_1": ["hybrid-retrieval","new‑user‑welcome‑recommend"],
        "layer_2": ["..."],
    }
    
    
    return plan