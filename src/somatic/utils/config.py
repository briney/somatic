"""Configuration utilities for Hydra configs."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary to YAML.

    Parameters
    ----------
    config
        Configuration dictionary.
    path
        Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass to a dictionary recursively.

    Parameters
    ----------
    obj
        Dataclass instance.

    Returns
    -------
    dict
        Dictionary representation.
    """
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got {type(obj)}")

    result = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        if is_dataclass(value):
            result[field.name] = dataclass_to_dict(value)
        elif isinstance(value, (list, tuple)):
            result[field.name] = [
                dataclass_to_dict(v) if is_dataclass(v) else v for v in value
            ]
        elif isinstance(value, dict):
            result[field.name] = {
                k: dataclass_to_dict(v) if is_dataclass(v) else v
                for k, v in value.items()
            }
        else:
            result[field.name] = value

    return result


def dict_to_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    """Convert a dictionary to a dataclass instance.

    Parameters
    ----------
    cls
        Dataclass type.
    data
        Dictionary with field values.

    Returns
    -------
    T
        Dataclass instance.
    """
    if not is_dataclass(cls):
        raise ValueError(f"Expected dataclass type, got {cls}")

    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name not in data:
            continue

        value = data[field_name]

        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[field_name] = dict_to_dataclass(field_type, value)
        else:
            kwargs[field_name] = value

    return cls(**kwargs)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two configuration dictionaries recursively.

    Parameters
    ----------
    base
        Base configuration.
    override
        Override configuration (takes precedence).

    Returns
    -------
    dict
        Merged configuration.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def flatten_config(
    config: dict[str, Any], prefix: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten a nested configuration dictionary.

    Parameters
    ----------
    config
        Nested configuration dictionary.
    prefix
        Prefix for keys.
    sep
        Separator between nested keys.

    Returns
    -------
    dict
        Flattened configuration.
    """
    result = {}

    for key, value in config.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_config(value, new_key, sep))
        else:
            result[new_key] = value

    return result
