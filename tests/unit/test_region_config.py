"""Tests for RegionEvalConfig."""

from somatic.eval.region_config import RegionEvalConfig, build_region_eval_config


class TestRegionEvalConfig:
    """Tests for RegionEvalConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegionEvalConfig()

        assert config.enabled is False
        assert config.mode == "per-position"
        assert config.position_batch_size == 32

        # All individual regions should be disabled by default
        assert config.hcdr1 is False
        assert config.hcdr2 is False
        assert config.hcdr3 is False
        assert config.lcdr1 is False
        assert config.lcdr2 is False
        assert config.lcdr3 is False
        assert config.hfwr1 is False
        assert config.hfwr2 is False
        assert config.hfwr3 is False
        assert config.hfwr4 is False
        assert config.lfwr1 is False
        assert config.lfwr2 is False
        assert config.lfwr3 is False
        assert config.lfwr4 is False

        # All aggregates should be disabled by default
        assert config.all_cdr is False
        assert config.all_fwr is False
        assert config.heavy is False
        assert config.light is False
        assert config.overall is False
        assert config.germline is False
        assert config.nongermline is False

    def test_custom_config(self):
        """Test config with custom values."""
        config = RegionEvalConfig(
            enabled=True,
            mode="standard",
            position_batch_size=64,
            hcdr3=True,
            lcdr3=True,
            all_cdr=True,
        )

        assert config.enabled is True
        assert config.mode == "standard"
        assert config.position_batch_size == 64
        assert config.hcdr3 is True
        assert config.lcdr3 is True
        assert config.all_cdr is True
        # Others still default
        assert config.hcdr1 is False
        assert config.all_fwr is False

    def test_get_enabled_regions_empty(self):
        """Test get_enabled_regions with no regions enabled."""
        config = RegionEvalConfig()

        regions = config.get_enabled_regions()
        assert regions == set()

    def test_get_enabled_regions_some(self):
        """Test get_enabled_regions with some regions enabled."""
        config = RegionEvalConfig(
            hcdr1=True,
            hcdr3=True,
            lcdr3=True,
            hfwr2=True,
        )

        regions = config.get_enabled_regions()
        assert regions == {"hcdr1", "hcdr3", "lcdr3", "hfwr2"}

    def test_get_enabled_regions_all_cdrs(self):
        """Test get_enabled_regions with all CDRs enabled."""
        config = RegionEvalConfig(
            hcdr1=True,
            hcdr2=True,
            hcdr3=True,
            lcdr1=True,
            lcdr2=True,
            lcdr3=True,
        )

        regions = config.get_enabled_regions()
        assert regions == {"hcdr1", "hcdr2", "hcdr3", "lcdr1", "lcdr2", "lcdr3"}

    def test_get_enabled_aggregates_empty(self):
        """Test get_enabled_aggregates with no aggregates enabled."""
        config = RegionEvalConfig()

        aggregates = config.get_enabled_aggregates()
        assert aggregates == set()

    def test_get_enabled_aggregates_some(self):
        """Test get_enabled_aggregates with some aggregates enabled."""
        config = RegionEvalConfig(
            all_cdr=True,
            heavy=True,
        )

        aggregates = config.get_enabled_aggregates()
        assert aggregates == {"all_cdr", "heavy"}

    def test_get_enabled_aggregates_all(self):
        """Test get_enabled_aggregates with all aggregates enabled."""
        config = RegionEvalConfig(
            all_cdr=True,
            all_fwr=True,
            heavy=True,
            light=True,
            overall=True,
            germline=True,
            nongermline=True,
        )

        aggregates = config.get_enabled_aggregates()
        assert aggregates == {
            "all_cdr",
            "all_fwr",
            "heavy",
            "light",
            "overall",
            "germline",
            "nongermline",
        }

    def test_has_any_enabled_false(self):
        """Test has_any_enabled when nothing is enabled."""
        config = RegionEvalConfig()

        assert config.has_any_enabled() is False

    def test_has_any_enabled_with_region(self):
        """Test has_any_enabled when a region is enabled."""
        config = RegionEvalConfig(hcdr3=True)

        assert config.has_any_enabled() is True

    def test_has_any_enabled_with_aggregate(self):
        """Test has_any_enabled when an aggregate is enabled."""
        config = RegionEvalConfig(all_cdr=True)

        assert config.has_any_enabled() is True

    def test_has_any_enabled_with_both(self):
        """Test has_any_enabled when both region and aggregate are enabled."""
        config = RegionEvalConfig(hcdr3=True, all_cdr=True)

        assert config.has_any_enabled() is True

    def test_germline_aggregate(self):
        """Test germline aggregate configuration."""
        config = RegionEvalConfig(germline=True)

        assert config.germline is True
        assert config.has_any_enabled() is True
        aggregates = config.get_enabled_aggregates()
        assert "germline" in aggregates

    def test_nongermline_aggregate(self):
        """Test nongermline aggregate configuration."""
        config = RegionEvalConfig(nongermline=True)

        assert config.nongermline is True
        assert config.has_any_enabled() is True
        aggregates = config.get_enabled_aggregates()
        assert "nongermline" in aggregates

    def test_germline_nongermline_together(self):
        """Test germline and nongermline aggregates enabled together."""
        config = RegionEvalConfig(germline=True, nongermline=True)

        assert config.germline is True
        assert config.nongermline is True
        aggregates = config.get_enabled_aggregates()
        assert "germline" in aggregates
        assert "nongermline" in aggregates

    def test_germline_with_regions(self):
        """Test germline aggregate combined with region tracking."""
        config = RegionEvalConfig(
            hcdr3=True,
            lcdr3=True,
            germline=True,
            nongermline=True,
        )

        regions = config.get_enabled_regions()
        aggregates = config.get_enabled_aggregates()

        assert regions == {"hcdr3", "lcdr3"}
        assert "germline" in aggregates
        assert "nongermline" in aggregates


class TestBuildRegionEvalConfig:
    """Tests for build_region_eval_config function."""

    def test_empty_dict(self):
        """Test with empty dictionary."""
        config = build_region_eval_config({})

        assert config.enabled is False
        assert config.mode == "per-position"

    def test_none_input(self):
        """Test with None input."""
        config = build_region_eval_config(None)

        assert config.enabled is False
        assert config.mode == "per-position"

    def test_full_config(self):
        """Test with full configuration dictionary."""
        cfg_dict = {
            "enabled": True,
            "mode": "standard",
            "position_batch_size": 64,
            "hcdr1": False,
            "hcdr2": False,
            "hcdr3": True,
            "lcdr1": False,
            "lcdr2": False,
            "lcdr3": True,
            "hfwr1": False,
            "hfwr2": False,
            "hfwr3": False,
            "hfwr4": False,
            "lfwr1": False,
            "lfwr2": False,
            "lfwr3": False,
            "lfwr4": False,
            "all_cdr": True,
            "all_fwr": True,
            "heavy": True,
            "light": True,
            "overall": False,
            "germline": True,
            "nongermline": False,
        }

        config = build_region_eval_config(cfg_dict)

        assert config.enabled is True
        assert config.mode == "standard"
        assert config.position_batch_size == 64
        assert config.hcdr3 is True
        assert config.lcdr3 is True
        assert config.all_cdr is True
        assert config.all_fwr is True
        assert config.heavy is True
        assert config.light is True
        assert config.overall is False
        assert config.germline is True
        assert config.nongermline is False
        assert config.hcdr1 is False

    def test_partial_config(self):
        """Test with partial configuration dictionary."""
        cfg_dict = {
            "enabled": True,
            "hcdr3": True,
            "all_cdr": True,
        }

        config = build_region_eval_config(cfg_dict)

        assert config.enabled is True
        assert config.hcdr3 is True
        assert config.all_cdr is True
        # Defaults for unspecified
        assert config.mode == "per-position"
        assert config.position_batch_size == 32
        assert config.hcdr1 is False

    def test_unknown_keys_ignored(self):
        """Test that unknown keys are ignored."""
        cfg_dict = {
            "enabled": True,
            "unknown_key": "value",
            "another_unknown": 123,
        }

        config = build_region_eval_config(cfg_dict)

        assert config.enabled is True
        # No error for unknown keys, they're just ignored

    def test_mode_variants(self):
        """Test different mode values."""
        for mode in ["standard", "per-position", "region-level"]:
            config = build_region_eval_config({"mode": mode})
            assert config.mode == mode
