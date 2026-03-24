"""
In-repo stub: original `best_logger` was team-local (see `.gitignore` history), not a PyPI dep.

Provides the import surface used by AgentEvolver; verbose output uses loguru at DEBUG.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

__all__ = [
    "NestedJsonItem",
    "SeqItem",
    "print_dict",
    "print_listofdict",
    "print_nested",
    "register_logger",
]


def register_logger(
    mods=None,
    non_console_mods=None,
    auto_clean_mods=None,
    base_log_path: str = "",
    debug: bool = False,
    **kwargs: Any,
) -> None:
    return None


def print_dict(d: Dict[str, Any], **kwargs: Any) -> None:
    try:
        logger.debug("print_dict: {}", json.dumps(d, ensure_ascii=False, default=str)[:8000])
    except Exception:
        logger.debug("print_dict: {!r}", d)


def print_listofdict(
    items: List[Dict[str, Any]],
    mod: str = "",
    header: str = "",
    narrow: bool = False,
    **kwargs: Any,
) -> None:
    try:
        logger.debug(
            "print_listofdict mod={} header={}: {}",
            mod,
            header,
            json.dumps(items, ensure_ascii=False, default=str)[:8000],
        )
    except Exception:
        logger.debug("print_listofdict mod={} header={} ({} items)", mod, header, len(items))


@dataclass
class SeqItem:
    text: List[Any]
    title: List[Any]
    count: List[Any]
    color: List[Any]


@dataclass
class NestedJsonItem:
    item_id: str
    outcome: str
    len_prompt_ids: int
    len_response_ids: int
    len_input_ids: int
    reward: str
    content: SeqItem
    final_reward: Optional[Any] = None


def print_nested(
    nested_items_print_buffer: Dict[str, Any],
    main_content: str = "",
    header: str = "",
    mod: str = "",
    narrow: bool = False,
    attach: str = "",
    **kwargs: Any,
) -> None:
    logger.debug(
        "print_nested mod={} header={} keys={}",
        mod,
        header,
        list(nested_items_print_buffer.keys()),
    )
