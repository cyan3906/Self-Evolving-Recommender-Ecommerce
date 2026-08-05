from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from ab_func import return_strategy_handlers
from build_batch_data import return_batch_data
from server.abtest_engine import ABTestEngine

if __name__ == "__main__":
    
    engine = ABTestEngine(
        bucket_count=10_000,
    )

    strategy_handlers = return_strategy_handlers()

    batch_data = return_batch_data()
    
    print(strategy_handlers)
    print(batch_data)
    
    