"""Vendor 源码路径注入。

上游 MediSee 仓库的 model/ 与 utils/ 目录使用顶层绝对导入
(``from utils.utils import ...``, ``from model.MediSee import MediSeeForCausalLM``)。
为了 1:1 保留上游源码而不修改任何一行 vendor 代码，这里把
``medseg/inference/mllm/medisee/_vendor`` 目录加入 ``sys.path``，从而让 model/utils
两个 namespace package 能被 Python 正确发现。

注意: 项目内不存在顶级 ``model``/``utils`` 包冲突（medseg 自身使用
``medseg.utils`` 这样的相对名字），因此这次注入是安全的。
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_THIS_DIR, "_vendor")


def ensure_vendor_on_path() -> str:
    """Idempotently insert vendor dir into ``sys.path``.

    Returns the absolute vendor directory path.
    """
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    return _VENDOR_DIR


__all__ = ["ensure_vendor_on_path"]
