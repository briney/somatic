# somatic

**Antibody language models that learn the intrinsic patterns of somatic hypermutation.**

Somatic is designed to overcome the [germline bias problem](https://academic.oup.com/bioinformatics/article/40/11/btae618/7845256) in antibody language models by using an information-weighted masked language modeling objective that [preferentially masks](https://www.cell.com/patterns/fulltext/S2666-3899(25)00087-X) high-value sequence positions. The model features chain-aware attention to jointly reason over paired heavy and light chain sequences.

## Installation

```bash
pip install somatic-lm
```

For development:
```bash
git clone https://github.com/briney/somatic.git
cd somatic
pip install -e .
```

## Quick Start

### Training a Model

Train a model on your antibody sequence data:

```bash
# Basic training
somatic train data.train=/path/to/sequences.csv

# With custom configuration
somatic train -c my_config.yaml data.train=/path/to/sequences.csv

# Use a smaller model architecture
somatic train data.train=/path/to/sequences.csv model=small
```

Your training data should be a CSV or Parquet file. The default column names for heavy and light chain sequences (in alignment with [AIRR-formatting guidelines](https://docs.airr-community.org/en/stable/datarep/rearrangements.html)) are `sequence_aa:0` and `sequence_aa:1` for the heavy and light chains, respectively. The default column names can be overriden at the CLI using `data.heavy_col=<col_name>` and `data.light_col=<col_name>`.

For multi-GPU training:
```bash
accelerate launch -m somatic.train data.train=/path/to/sequences.csv
```

### Encoding Sequences

Extract embeddings from a trained model:

```bash
# Get mean-pooled embeddings
somatic encode -c checkpoints/best.pt -i sequences.csv -o embeddings.pt --pooling mean

# Get full sequence embeddings (no pooling)
somatic encode -c model.pt -i sequences.parquet -o embeddings.npy --pooling none

# Available pooling strategies: mean, cls, max, mean_max, none
```

### Model Configuration

Check the parameter count for different model configurations:

```bash
# Default (base) model
somatic model-size

# Small model variant
somatic model-size model=small

# Large model with custom layers
somatic model-size model=large model.n_layers=32
```

## Python API

```python
from somatic import SomaticEncoder

# Load a trained model
encoder = SomaticEncoder.from_pretrained("model.pt", pooling="mean")

# Encode a single antibody
embedding = encoder.encode(
    heavy_chain="EVQLVESGGGLVQPGGSLRLSCAASGFTFS...",
    light_chain="DIQMTQSPSSLSASVGDRVTITCRASQSIS..."
)

# Encode a batch
embeddings = encoder.encode_batch(heavy_chains, light_chains, batch_size=32)

# Predict masked positions
result = encoder.predict(
    heavy_chain="EVQLV<mask><mask>SGGG...",
    light_chain="DIQMT..."
)
print(result["heavy_chain"])  # Predicted sequence
```

## Model Architecture

Somatic uses a modern transformer architecture with:

- **Chain-aware attention**: Hybrid self/cross attention for paired heavy/light chains
- **Rotary position embeddings (RoPE)**: For relative position encoding
- **SwiGLU feed-forward networks**: Improved training dynamics
- **Pre-norm architecture**: Layer normalization before attention and FFN

## Configuration

Somatic uses [Hydra](https://hydra.cc/) for configuration management. Override any config value from the command line:

```bash
somatic train \
    data.train=/path/to/data.csv \
    model=large \
    train.batch_size=64 \
    train.learning_rate=1e-4 \
    train.max_steps=100000
```

## License

MIT
