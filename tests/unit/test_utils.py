"""Tests for utility functions."""

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from somatic.utils import (
    dataclass_to_dict,
    dict_to_dataclass,
    flatten_config,
    get_generator,
    load_yaml,
    merge_configs,
    save_yaml,
    set_seed,
)


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        values1 = torch.rand(10).tolist()

        set_seed(42)
        values2 = torch.rand(10).tolist()

        assert values1 == values2

    def test_different_seeds(self):
        set_seed(42)
        values1 = torch.rand(10).tolist()

        set_seed(123)
        values2 = torch.rand(10).tolist()

        assert values1 != values2

    def test_deterministic_mode(self):
        # Should not raise
        set_seed(42, deterministic=True)
        _ = torch.rand(10)


class TestGetGenerator:
    def test_basic_generator(self):
        gen = get_generator(42)
        assert isinstance(gen, torch.Generator)

    def test_reproducibility(self):
        gen1 = get_generator(42)
        values1 = torch.rand(10, generator=gen1).tolist()

        gen2 = get_generator(42)
        values2 = torch.rand(10, generator=gen2).tolist()

        assert values1 == values2

    def test_different_seeds(self):
        gen1 = get_generator(42)
        values1 = torch.rand(10, generator=gen1).tolist()

        gen2 = get_generator(123)
        values2 = torch.rand(10, generator=gen2).tolist()

        assert values1 != values2


class TestLoadSaveYaml:
    def test_save_and_load(self, tmp_path):
        config = {
            "model": {"d_model": 256, "n_layers": 6},
            "training": {"lr": 1e-4, "batch_size": 32},
        }

        path = tmp_path / "config.yaml"
        save_yaml(config, path)
        loaded = load_yaml(path)

        assert loaded == config

    def test_creates_parent_dirs(self, tmp_path):
        config = {"key": "value"}
        path = tmp_path / "nested" / "dir" / "config.yaml"

        save_yaml(config, path)
        assert path.exists()


@dataclass
class InnerConfig:
    value: int
    name: str


@dataclass
class OuterConfig:
    inner: InnerConfig
    items: list[int]


class TestDataclassToDict:
    def test_simple_dataclass(self):
        config = InnerConfig(value=10, name="test")
        result = dataclass_to_dict(config)

        assert result == {"value": 10, "name": "test"}

    def test_nested_dataclass(self):
        config = OuterConfig(inner=InnerConfig(value=5, name="inner"), items=[1, 2, 3])
        result = dataclass_to_dict(config)

        assert result == {"inner": {"value": 5, "name": "inner"}, "items": [1, 2, 3]}

    def test_non_dataclass_raises(self):
        with pytest.raises(ValueError, match="Expected dataclass"):
            dataclass_to_dict({"key": "value"})


class TestDictToDataclass:
    def test_simple_dataclass(self):
        data = {"value": 10, "name": "test"}
        result = dict_to_dataclass(InnerConfig, data)

        assert result.value == 10
        assert result.name == "test"

    def test_missing_field_uses_default(self):
        @dataclass
        class ConfigWithDefault:
            required: int
            optional: str = "default"

        data = {"required": 5}
        result = dict_to_dataclass(ConfigWithDefault, data)

        assert result.required == 5
        assert result.optional == "default"

    def test_non_dataclass_raises(self):
        with pytest.raises(ValueError, match="Expected dataclass type"):
            dict_to_dataclass(dict, {"key": "value"})


class TestMergeConfigs:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"model": {"d_model": 256, "n_layers": 6}, "lr": 1e-4}
        override = {"model": {"n_layers": 12}, "batch_size": 32}

        result = merge_configs(base, override)

        assert result["model"]["d_model"] == 256
        assert result["model"]["n_layers"] == 12
        assert result["lr"] == 1e-4
        assert result["batch_size"] == 32

    def test_base_unchanged(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}

        merge_configs(base, override)
        assert base == {"a": 1, "b": 2}


class TestFlattenConfig:
    def test_flat_config(self):
        config = {"a": 1, "b": 2}
        result = flatten_config(config)
        assert result == {"a": 1, "b": 2}

    def test_nested_config(self):
        config = {"model": {"d_model": 256, "n_layers": 6}, "lr": 1e-4}
        result = flatten_config(config)

        assert result == {"model.d_model": 256, "model.n_layers": 6, "lr": 1e-4}

    def test_deeply_nested(self):
        config = {"level1": {"level2": {"level3": "value"}}}
        result = flatten_config(config)

        assert result == {"level1.level2.level3": "value"}

    def test_custom_separator(self):
        config = {"model": {"d_model": 256}}
        result = flatten_config(config, sep="/")

        assert result == {"model/d_model": 256}
