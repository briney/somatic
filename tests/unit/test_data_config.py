"""Tests for data configuration parsing."""

import pytest
from omegaconf import OmegaConf

from somatic.data.config import (
    DatasetConfig,
    is_single_eval_dataset,
    is_single_train_dataset,
    normalize_fractions,
    parse_eval_config,
    parse_train_config,
)


class TestNormalizeFractions:
    """Tests for normalize_fractions function."""

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        result = normalize_fractions({})
        assert result == {}

    def test_no_fractions_specified(self):
        """All datasets get equal weight when no fractions specified."""
        fractions = {"a": None, "b": None, "c": None}
        result = normalize_fractions(fractions)

        assert len(result) == 3
        assert abs(result["a"] - 1 / 3) < 1e-6
        assert abs(result["b"] - 1 / 3) < 1e-6
        assert abs(result["c"] - 1 / 3) < 1e-6

    def test_all_fractions_specified(self):
        """Normalize when all fractions specified."""
        fractions = {"a": 0.6, "b": 0.4}
        result = normalize_fractions(fractions)

        assert abs(result["a"] - 0.6) < 1e-6
        assert abs(result["b"] - 0.4) < 1e-6

    def test_all_fractions_specified_needs_normalization(self):
        """Normalize when all fractions specified but don't sum to 1."""
        fractions = {"a": 3.0, "b": 2.0}
        result = normalize_fractions(fractions)

        assert abs(result["a"] - 0.6) < 1e-6
        assert abs(result["b"] - 0.4) < 1e-6

    def test_some_fractions_specified(self):
        """Unspecified fractions share remaining mass."""
        fractions = {"a": 0.6, "b": None}
        result = normalize_fractions(fractions)

        assert abs(result["a"] - 0.6) < 1e-6
        assert abs(result["b"] - 0.4) < 1e-6

    def test_multiple_unspecified(self):
        """Multiple unspecified share remaining mass equally."""
        fractions = {"a": 0.4, "b": None, "c": None}
        result = normalize_fractions(fractions)

        assert abs(result["a"] - 0.4) < 1e-6
        assert abs(result["b"] - 0.3) < 1e-6
        assert abs(result["c"] - 0.3) < 1e-6

    def test_specified_exceeds_one(self):
        """When specified fractions exceed 1.0, normalize everything."""
        fractions = {"a": 0.8, "b": 0.4, "c": None}
        result = normalize_fractions(fractions)

        # c gets 0 because 0.8 + 0.4 > 1.0
        # Then normalized: 0.8 / 1.2 ≈ 0.667, 0.4 / 1.2 ≈ 0.333, 0 / 1.2 = 0
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6


class TestIsSingleTrainDataset:
    """Tests for is_single_train_dataset function."""

    def test_string_is_single(self):
        """String path is a single dataset."""
        assert is_single_train_dataset("/path/to/data.parquet") is True

    def test_dict_is_multi(self):
        """Dict config is multi-dataset."""
        cfg = OmegaConf.create({"a": {"path": "/a.parquet"}})
        assert is_single_train_dataset(cfg) is False

    def test_none_is_not_single(self):
        """None is not a single dataset."""
        assert is_single_train_dataset(None) is False


class TestParseTrainConfig:
    """Tests for parse_train_config function."""

    def test_none_raises(self):
        """None config raises ValueError."""
        with pytest.raises(ValueError, match="train config is required"):
            parse_train_config(None)

    def test_single_path(self):
        """Single path string returns single dataset."""
        paths, fractions = parse_train_config("/path/to/train.parquet")

        assert paths == {"train": "/path/to/train.parquet"}
        assert fractions == {"train": 1.0}

    def test_multi_dataset_with_fractions(self):
        """Multi-dataset config with fractions."""
        cfg = OmegaConf.create(
            {
                "dataset_a": {"path": "/a.parquet", "fraction": 0.6},
                "dataset_b": {"path": "/b.parquet", "fraction": 0.4},
            }
        )

        paths, fractions = parse_train_config(cfg)

        assert paths == {"dataset_a": "/a.parquet", "dataset_b": "/b.parquet"}
        assert abs(fractions["dataset_a"] - 0.6) < 1e-6
        assert abs(fractions["dataset_b"] - 0.4) < 1e-6

    def test_multi_dataset_shorthand(self):
        """Multi-dataset with path shorthand (just strings)."""
        cfg = OmegaConf.create(
            {
                "dataset_a": "/a.parquet",
                "dataset_b": "/b.parquet",
            }
        )

        paths, fractions = parse_train_config(cfg)

        assert paths == {"dataset_a": "/a.parquet", "dataset_b": "/b.parquet"}
        assert abs(fractions["dataset_a"] - 0.5) < 1e-6
        assert abs(fractions["dataset_b"] - 0.5) < 1e-6

    def test_multi_dataset_partial_fractions(self):
        """Multi-dataset with some fractions specified."""
        cfg = OmegaConf.create(
            {
                "dataset_a": {"path": "/a.parquet", "fraction": 0.7},
                "dataset_b": {"path": "/b.parquet"},  # No fraction
            }
        )

        paths, fractions = parse_train_config(cfg)

        assert paths == {"dataset_a": "/a.parquet", "dataset_b": "/b.parquet"}
        assert abs(fractions["dataset_a"] - 0.7) < 1e-6
        assert abs(fractions["dataset_b"] - 0.3) < 1e-6


class TestIsSingleEvalDataset:
    """Tests for is_single_eval_dataset function."""

    def test_string_is_single(self):
        """String path is a single dataset."""
        assert is_single_eval_dataset("/path/to/data.parquet") is True

    def test_dict_is_multi(self):
        """Dict config is multi-dataset."""
        cfg = OmegaConf.create({"validation": "/val.parquet"})
        assert is_single_eval_dataset(cfg) is False

    def test_none_is_not_single(self):
        """None is not a single dataset."""
        assert is_single_eval_dataset(None) is False


class TestParseEvalConfig:
    """Tests for parse_eval_config function."""

    def test_empty_config(self):
        """Empty eval config returns empty dict."""
        result = parse_eval_config({}, OmegaConf.create({}))
        assert result == {}

    def test_none_config(self):
        """None eval config returns empty dict."""
        result = parse_eval_config(None, OmegaConf.create({}))
        assert result == {}

    def test_single_path(self):
        """Single path string returns single dataset with key 'eval'."""
        result = parse_eval_config("/path/to/eval.parquet", OmegaConf.create({}))

        assert len(result) == 1
        assert "eval" in result
        assert result["eval"].path == "/path/to/eval.parquet"
        assert result["eval"].batch_size is None
        assert result["eval"].load_coords is None

    def test_named_dataset_shorthand(self):
        """Named dataset with path shorthand."""
        cfg = OmegaConf.create({"validation": "/path/to/val.parquet"})
        global_cfg = OmegaConf.create({})

        result = parse_eval_config(cfg, global_cfg)

        assert "validation" in result
        assert result["validation"].path == "/path/to/val.parquet"
        assert result["validation"].batch_size is None
        assert result["validation"].load_coords is None

    def test_full_config(self):
        """Full config with all options."""
        cfg = OmegaConf.create(
            {
                "test": {
                    "path": "/path/to/test.parquet",
                    "batch_size": 64,
                    "load_coords": True,
                }
            }
        )
        global_cfg = OmegaConf.create({})

        result = parse_eval_config(cfg, global_cfg)

        assert result["test"].path == "/path/to/test.parquet"
        assert result["test"].batch_size == 64
        assert result["test"].load_coords is True

    def test_multiple_datasets(self):
        """Multiple eval datasets."""
        cfg = OmegaConf.create(
            {
                "validation": "/val.parquet",
                "test": {"path": "/test.parquet", "batch_size": 32},
            }
        )
        global_cfg = OmegaConf.create({})

        result = parse_eval_config(cfg, global_cfg)

        assert len(result) == 2
        assert result["validation"].path == "/val.parquet"
        assert result["test"].path == "/test.parquet"
        assert result["test"].batch_size == 32

    def test_metrics_config(self):
        """Per-dataset metrics configuration."""
        cfg = OmegaConf.create(
            {
                "struct_val": {
                    "path": "/struct_val.parquet",
                    "load_coords": True,
                    "metrics": {"only": ["masked_accuracy", "p_at_l"]},
                }
            }
        )
        global_cfg = OmegaConf.create({})

        result = parse_eval_config(cfg, global_cfg)

        assert result["struct_val"].metrics is not None
        assert result["struct_val"].metrics["only"] == ["masked_accuracy", "p_at_l"]


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_minimal_config(self):
        """Create config with just path."""
        cfg = DatasetConfig(path="/path/to/data.parquet")
        assert cfg.path == "/path/to/data.parquet"
        assert cfg.fraction is None
        assert cfg.batch_size is None
        assert cfg.load_coords is None
        assert cfg.metrics is None

    def test_full_config(self):
        """Create config with all fields."""
        cfg = DatasetConfig(
            path="/path/to/data.parquet",
            fraction=0.5,
            batch_size=64,
            load_coords=True,
            metrics={"only": ["accuracy"]},
        )
        assert cfg.path == "/path/to/data.parquet"
        assert cfg.fraction == 0.5
        assert cfg.batch_size == 64
        assert cfg.load_coords is True
        assert cfg.metrics == {"only": ["accuracy"]}
