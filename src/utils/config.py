from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_project_relative_paths(config: Dict[str, Any], project_root: str | Path) -> Dict[str, Any]:
    """把配置中的相对路径统一转换为项目根目录下的绝对路径。"""
    root = Path(project_root).resolve()
    for key, value in config.get("paths", {}).items():
        config["paths"][key] = str((root / value).resolve())
    return config
