"""Built-in evaluation metrics."""

from .classification import LossMetric, MaskedAccuracyMetric, PerplexityMetric
from .contact import PrecisionAtLMetric
from .probes import CDRProbeMetric, ChainProbeMetric, PositionProbeMetric
from .region import RegionAccuracyMetric, RegionLossMetric, RegionPerplexityMetric

__all__ = [
    # Classification metrics
    "MaskedAccuracyMetric",
    "PerplexityMetric",
    "LossMetric",
    # Contact prediction
    "PrecisionAtLMetric",
    # Probe-based metrics
    "ChainProbeMetric",
    "PositionProbeMetric",
    "CDRProbeMetric",
    # Region-based metrics
    "RegionAccuracyMetric",
    "RegionPerplexityMetric",
    "RegionLossMetric",
]
