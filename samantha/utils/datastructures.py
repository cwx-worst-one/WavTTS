from typing import Any, Dict, List


def select_keys(sample: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: v for k, v in sample.items() if k in keys}
