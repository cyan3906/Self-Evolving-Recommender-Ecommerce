
"""
这里其实可以封装成一个agent，
每个agent都有一个技能计划。


"""

def return_sequential_skill_plan() -> str:
    """
    返回串行技能计划。
    """
    
    state_machine = {
        "layer_1": {
            "state": 0,  # layer_1整体状态：0未开始 /1执行中 /2全部完成 /3异常
            "tools": ["hybrid_search","new_user_recommend"],
            "parent": "root",
        },
        "tool_mapping": {
            "hybrid_search": {
                "layer": "layer_1",
                "state": 0,
                "parent": "new_user_recommend",
                "result": None,
            },
            "new_user_recommend": {
                "layer": "layer_1",
                "state": 0,
                "parent": "root",
                "result": None,
            }
        }
    }
    
    
    return state_machine


def return_parallel_skill_plan() -> str:
    """
    返回并行技能计划。
    """
    
    state_machine = {
        "layer_1": {
            "state": 0,  # layer_1整体状态：0未开始 /1执行中 /2全部完成 /3异常
            "tools": ["hybrid_search","new_user_recommend"],
            "parent": "root",
        },
        "layer_2": {
            "state": 0,  # layer_2整体状态：0未开始 /1执行中 /2全部完成 /3异常
            "tools": ["get_top_hot_products"],
            "parent": "root",
        },
        "layer_3":{
            "state": 0,  # layer_3整体状态：0未开始 /1执行中 /2全部完成 /3异常
            "tools": ["short_term_memory"],
            "parent": "root",
        },
        
        "tool_mapping": {
            "hybrid_search": {
                "layer": "layer_1",
                "state": 0,
                "parent": "new_user_recommend",
                "result": None,
            },
            "new_user_recommend": {
                "layer": "layer_1",
                "state": 0,
                "parent": "root",
                "result": None,
            },
            "get_top_hot_products": {
                "layer": "layer_2",
                "state": 0,
                "parent": "root",
                "result": None,
            },
            "short_term_memory": {
                "layer": "layer_3",
                "state": 0,
                "parent": "root",
                "result": None,
            },
        }
    }
    
    
    return state_machine


def return_hybridbrid_skill_plan() -> str:
    """
    返回混合技能计划。
    """
    
    plan = {
        "layer_1": ["hybrid-retrieval","new_user_recommend"],
        "layer_2": ["..."],
    }
    
    
    return plan