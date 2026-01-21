"""PyTorch Dataset for antibody sequences and structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Supported structure file extensions
STRUCTURE_EXTENSIONS = {".pdb", ".ent", ".cif", ".mmcif"}

# Supported sequence file extensions
SEQUENCE_EXTENSIONS = {".csv", ".tsv", ".parquet"}

# Standard 3-letter to 1-letter amino acid mapping
AA3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    # Non-standard / modified residues
    "MSE": "M",  # Selenomethionine
    "SEC": "C",  # Selenocysteine (sometimes)
    "PYL": "K",  # Pyrrolysine
    "HYP": "P",  # Hydroxyproline
    "SEP": "S",  # Phosphoserine
    "TPO": "T",  # Phosphothreonine
    "PTR": "Y",  # Phosphotyrosine
    "CSO": "C",  # S-hydroxycysteine
    "CME": "C",  # S,S-(2-hydroxyethyl)thiocysteine
    "MLY": "K",  # N-dimethyl-lysine
    "UNK": "X",  # Unknown
}


class StructureData(NamedTuple):
    """Parsed structure data.

    Attributes:
        pid: Structure identifier (filename stem or structure ID).
        protein_sequence: One-letter amino acid sequence.
        coords: Backbone coordinates [L, 3, 3] for N, CA, C atoms.
        chain_id: Chain identifier used for extraction.
    """

    pid: str
    protein_sequence: str
    coords: np.ndarray
    chain_id: str | None


def _get_one_letter_code(res_name: str) -> str:
    """Convert 3-letter amino acid code to 1-letter code.

    Args:
        res_name: 3-letter residue name (uppercase).

    Returns:
        1-letter amino acid code, or 'X' for unknown residues.
    """
    # First check our mapping
    if res_name in AA3TO1:
        return AA3TO1[res_name]

    # Try Biopython's three_to_one for standard residues
    try:
        from Bio.PDB.Polypeptide import three_to_one

        return three_to_one(res_name)
    except (KeyError, ImportError):
        pass

    # Unknown residue
    return "X"


def parse_structure(
    path: str | Path,
    *,
    chain_id: str | None = None,
    strict: bool = False,
) -> StructureData:
    """Parse a PDB or mmCIF file and extract sequence and backbone coordinates.

    Args:
        path: Path to .pdb, .ent, .cif, or .mmcif file.
        chain_id: Specific chain to extract. If None, uses first polymer chain.
        strict: If True, raise on missing backbone atoms; else fill with NaN.

    Returns:
        StructureData with pid, sequence, and coords [L, 3, 3].

    Raises:
        ValueError: If structure cannot be parsed or has no valid residues.
        FileNotFoundError: If the file does not exist.
    """
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import is_aa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    suffix = path.suffix.lower()

    # Select parser based on file extension
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:  # .pdb, .ent, or unknown
        parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(path.stem, str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse structure file {path}: {e}") from e

    models = list(structure.get_models())
    if len(models) == 0:
        raise ValueError(f"No models found in {path}")

    model = models[0]

    # Find target chain
    chain = None
    if chain_id is not None:
        for ch in model:
            if ch.id == chain_id:
                chain = ch
                break
        if chain is None:
            raise ValueError(f"Chain '{chain_id}' not found in {path}")
    else:
        # Find first chain with amino acid residues
        for ch in model:
            residues = [r for r in ch if is_aa(r, standard=False)]
            if residues:
                chain = ch
                break
        if chain is None:
            raise ValueError(f"No protein chain found in {path}")

    used_chain_id = chain.id

    seq_chars: list[str] = []
    coords_list: list[list[list[float]]] = []

    for residue in chain:
        if not is_aa(residue, standard=False):
            continue

        # Get one-letter code
        res_name = residue.resname.upper().strip()
        aa = _get_one_letter_code(res_name)
        seq_chars.append(aa)

        # Extract N, CA, C coordinates
        try:
            n_coord = residue["N"].coord.tolist()
            ca_coord = residue["CA"].coord.tolist()
            c_coord = residue["C"].coord.tolist()
            coords_list.append([n_coord, ca_coord, c_coord])
        except KeyError as e:
            if strict:
                raise ValueError(
                    f"Missing backbone atom {e} in residue {residue.id} of {path}"
                ) from e
            # Fill with NaN for missing atoms
            coords_list.append([[np.nan] * 3, [np.nan] * 3, [np.nan] * 3])

    if len(seq_chars) == 0:
        raise ValueError(f"No amino acid residues extracted from {path}")

    return StructureData(
        pid=path.stem,
        protein_sequence="".join(seq_chars),
        coords=np.array(coords_list, dtype=np.float32),
        chain_id=used_chain_id,
    )


class AntibodyDataset(Dataset):
    """
    Dataset for paired antibody heavy/light chain sequences.

    Reads data from CSV or Parquet files with columns:
    - heavy_chain, light_chain (required)
    - heavy_cdr_mask, light_cdr_mask (optional)
    - heavy_non_templated_mask, light_non_templated_mask (optional)
    - heavy_coords, light_coords (optional) - CA atom coordinates
    """

    def __init__(
        self,
        data_path: str | Path,
        max_length: int = 320,
        heavy_col: str = "heavy_chain",
        light_col: str = "light_chain",
        heavy_cdr_col: str = "heavy_cdr_mask",
        light_cdr_col: str = "light_cdr_mask",
        heavy_nongermline_col: str = "heavy_non_templated_mask",
        light_nongermline_col: str = "light_non_templated_mask",
        load_coords: bool = False,
        heavy_coords_col: str = "heavy_coords",
        light_coords_col: str = "light_coords",
    ) -> None:
        self.data_path = Path(data_path)
        self.max_length = max_length

        self.heavy_col = heavy_col
        self.light_col = light_col
        self.heavy_cdr_col = heavy_cdr_col
        self.light_cdr_col = light_cdr_col
        self.heavy_nongermline_col = heavy_nongermline_col
        self.light_nongermline_col = light_nongermline_col

        self.load_coords = load_coords
        self.heavy_coords_col = heavy_coords_col
        self.light_coords_col = light_coords_col

        self.df = self._load_data()

        if heavy_col not in self.df.columns or light_col not in self.df.columns:
            raise ValueError(f"Missing required columns: {heavy_col}, {light_col}")

        self.has_cdr_mask = (
            heavy_cdr_col in self.df.columns and light_cdr_col in self.df.columns
        )
        self.has_nt_mask = (
            heavy_nongermline_col in self.df.columns and light_nongermline_col in self.df.columns
        )
        self.has_coords = (
            load_coords
            and heavy_coords_col in self.df.columns
            and light_coords_col in self.df.columns
        )

    def _load_data(self) -> pd.DataFrame:
        """Load data from file, ensuring mask columns are read as strings.

        For CSV/TSV files, mask columns are explicitly read as strings to prevent
        pandas from interpreting pure-digit strings as integers (e.g., "00011" -> 11).
        """
        if self.data_path.suffix == ".parquet":
            return pd.read_parquet(self.data_path)
        elif self.data_path.suffix in [".csv", ".tsv"]:
            sep = "\t" if self.data_path.suffix == ".tsv" else ","
            # Force string dtype for mask columns to preserve leading zeros
            dtype_overrides = {
                self.heavy_cdr_col: str,
                self.light_cdr_col: str,
                self.heavy_nongermline_col: str,
                self.light_nongermline_col: str,
            }
            return pd.read_csv(self.data_path, sep=sep, dtype=dtype_overrides)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _parse_mask(self, mask_str: str) -> list[int] | None:
        """Parse mask string into list of integers.

        Expects a string of contiguous digits without delimiters.
        Examples:
            - CDR mask: "000011110000022200003333" (0=FW, 1=CDR1, 2=CDR2, 3=CDR3)
            - NT mask: "0000100110000001" (0=germline, 1=non-germline)

        Args:
            mask_str: String of digit characters (no delimiters).

        Returns:
            List of integers, one per position, or None if input is NaN/empty.
        """
        if pd.isna(mask_str):
            return None
        if isinstance(mask_str, str):
            if not mask_str:  # Empty string
                return None
            return [int(c) for c in mask_str]
        return list(mask_str)

    def _parse_coords(self, coords_data: Any) -> np.ndarray | None:
        """Parse coordinate data from various formats.

        Supports:
        - numpy array (N, 3)
        - JSON string of list of lists
        - Comma-separated string of flattened coordinates

        Args:
            coords_data: Raw coordinate data from dataframe.

        Returns:
            Numpy array of shape (N, 3) or None if invalid.
        """
        if coords_data is None or (isinstance(coords_data, float) and pd.isna(coords_data)):
            return None

        if isinstance(coords_data, np.ndarray):
            return coords_data.astype(np.float32)

        if isinstance(coords_data, str):
            # Try JSON format first
            try:
                parsed = json.loads(coords_data)
                return np.array(parsed, dtype=np.float32)
            except json.JSONDecodeError:
                pass

            # Try comma-separated format (flattened)
            try:
                values = [float(x) for x in coords_data.split(",")]
                n_coords = len(values) // 3
                return np.array(values, dtype=np.float32).reshape(n_coords, 3)
            except (ValueError, TypeError):
                return None

        if isinstance(coords_data, (list, tuple)):
            return np.array(coords_data, dtype=np.float32)

        return None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        result = {
            "heavy_chain": row[self.heavy_col],
            "light_chain": row[self.light_col],
        }

        if self.has_cdr_mask:
            result["heavy_cdr_mask"] = self._parse_mask(row[self.heavy_cdr_col])
            result["light_cdr_mask"] = self._parse_mask(row[self.light_cdr_col])
        else:
            result["heavy_cdr_mask"] = None
            result["light_cdr_mask"] = None

        if self.has_nt_mask:
            result["heavy_non_templated_mask"] = self._parse_mask(row[self.heavy_nongermline_col])
            result["light_non_templated_mask"] = self._parse_mask(row[self.light_nongermline_col])
        else:
            result["heavy_non_templated_mask"] = None
            result["light_non_templated_mask"] = None

        if self.has_coords:
            result["heavy_coords"] = self._parse_coords(row[self.heavy_coords_col])
            result["light_coords"] = self._parse_coords(row[self.light_coords_col])
        else:
            result["heavy_coords"] = None
            result["light_coords"] = None

        return result


class MultiDataset(Dataset):
    """Combines multiple datasets with weighted sampling."""

    def __init__(
        self,
        datasets: dict[str, Dataset],
        weights: dict[str, float] | None = None,
    ) -> None:
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())

        self.lengths = {name: len(ds) for name, ds in datasets.items()}
        self.total_length = sum(self.lengths.values())

        self._build_index_map()

        if weights is None:
            weights = {name: 1.0 for name in self.dataset_names}

        total_weight = sum(weights.values())
        self.weights = {name: w / total_weight for name, w in weights.items()}
        self._build_sampling_probs()

    def _build_index_map(self) -> None:
        self.index_map = []
        for name in self.dataset_names:
            for local_idx in range(self.lengths[name]):
                self.index_map.append((name, local_idx))

    def _build_sampling_probs(self) -> None:
        probs = []
        for name, local_idx in self.index_map:
            prob = self.weights[name] / self.lengths[name]
            probs.append(prob)

        total = sum(probs)
        self.sampling_probs = [p / total for p in probs]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dataset_name, local_idx = self.index_map[idx]
        item = self.datasets[dataset_name][local_idx]
        item["_dataset"] = dataset_name
        return item

    def get_sampler_weights(self) -> torch.Tensor:
        return torch.tensor(self.sampling_probs)


class StructureDataset(Dataset):
    """Dataset for a folder of PDB/mmCIF structure files.

    This dataset is designed for evaluation with structure-based metrics.
    Each structure file produces one sample with:
      - pid: structure identifier (filename stem)
      - seq: amino acid sequence extracted from structure
      - coords: backbone coordinates [max_length, 3, 3] for N, CA, C atoms
      - masks: boolean mask [max_length] for valid positions
      - nan_masks: same as masks (all backbone atoms present or NaN)

    Note: No VQ indices are provided since these are raw structures without
    pre-computed structure tokens.

    Args:
        folder_path: Path to folder containing PDB/mmCIF files.
        max_length: Maximum sequence length (for padding/truncation).
        chain_id: Specific chain to extract from each file. If None, uses
            first polymer chain found in each structure.
        strict: If True, raise errors on missing backbone atoms.
            If False (default), fill missing atoms with NaN.
        recursive: If True, search subdirectories recursively.

    Raises:
        ValueError: If folder_path is not a directory or contains no
            structure files.

    Example config:
        .. code-block:: yaml

            eval:
              pdb_benchmark:
                path: /path/to/pdb_folder
                format: structure
                chain_id: A
                metrics:
                  only: [lddt, tm_score, rmsd]
    """

    def __init__(
        self,
        folder_path: str | Path,
        max_length: int,
        *,
        chain_id: str | None = None,
        strict: bool = False,
        recursive: bool = False,
    ):
        self.folder_path = Path(folder_path)
        if not self.folder_path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        self.max_length = int(max_length)
        self.chain_id = chain_id
        self.strict = bool(strict)
        self.recursive = bool(recursive)

        # Discover structure files
        if recursive:
            self._files = sorted(
                [
                    p
                    for p in self.folder_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in STRUCTURE_EXTENSIONS
                ]
            )
        else:
            self._files = sorted(
                [
                    p
                    for p in self.folder_path.iterdir()
                    if p.is_file() and p.suffix.lower() in STRUCTURE_EXTENSIONS
                ]
            )

        if len(self._files) == 0:
            raise ValueError(
                f"No structure files found in {folder_path}. "
                f"Supported extensions: {STRUCTURE_EXTENSIONS}"
            )

        # Flags for compatibility with existing dataset code
        self.has_coords = True
        self._is_parquet = False

    def __len__(self) -> int:
        """Return number of structure files in the dataset."""
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Load and parse a structure file.

        Args:
            idx: Index of the structure file to load.

        Returns:
            Dict with keys:
                - pid (str): Structure identifier
                - seq (str): Amino acid sequence
                - coords (Tensor): [max_length, 3, 3] backbone coordinates
                - masks (Tensor): [max_length] boolean mask for valid positions
                - nan_masks (Tensor): [max_length] same as masks

            Note: 'indices' key is NOT included (not available for raw structures).
        """
        path = self._files[idx]

        # Parse structure
        data = parse_structure(
            path,
            chain_id=self.chain_id,
            strict=self.strict,
        )

        seq_len = len(data.protein_sequence)
        coords = data.coords  # [L, 3, 3]

        # Truncate if needed
        if seq_len > self.max_length:
            seq = data.protein_sequence[: self.max_length]
            coords = coords[: self.max_length]
            seq_len = self.max_length
        else:
            seq = data.protein_sequence

        # Pad coordinates to max_length with NaN
        pad_len = self.max_length - seq_len
        coords_padded = np.full((self.max_length, 3, 3), np.nan, dtype=np.float32)
        coords_padded[:seq_len] = coords

        # Build mask (True for valid positions)
        mask = [True] * seq_len + [False] * pad_len

        out: dict[str, torch.Tensor | str] = {
            "pid": data.pid,
            "seq": seq,
            "coords": torch.tensor(coords_padded, dtype=torch.float32),
            "masks": torch.tensor(mask, dtype=torch.bool),
            "nan_masks": torch.tensor(mask, dtype=torch.bool),
        }

        return out

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"StructureDataset("
            f"folder={self.folder_path}, "
            f"num_files={len(self._files)}, "
            f"max_length={self.max_length})"
        )


def detect_dataset_format(path: str | Path) -> str:
    """Detect the dataset format based on file/directory contents.

    Auto-detection rules:
    - CSV, TSV, Parquet files → "sequence"
    - Directory containing parquet files (shards) → "sequence"
    - Directory containing PDB/mmCIF files → "structure"

    Args:
        path: Path to a file or directory.

    Returns:
        Either "sequence" or "structure".

    Raises:
        ValueError: If the path doesn't exist or format cannot be determined.
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in SEQUENCE_EXTENSIONS:
            return "sequence"
        if suffix in STRUCTURE_EXTENSIONS:
            # Single structure file - treat as structure
            return "structure"
        raise ValueError(
            f"Unknown file extension '{suffix}'. "
            f"Supported sequence formats: {SEQUENCE_EXTENSIONS}. "
            f"Supported structure formats: {STRUCTURE_EXTENSIONS}."
        )

    # Path is a directory
    if path.is_dir():
        # Check for parquet files (sharded sequence dataset)
        parquet_files = list(path.glob("*.parquet"))
        if parquet_files:
            return "sequence"

        # Check for structure files
        structure_files = [
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in STRUCTURE_EXTENSIONS
        ]
        if structure_files:
            return "structure"

        # Check subdirectories recursively for structure files
        structure_files_recursive = [
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in STRUCTURE_EXTENSIONS
        ]
        if structure_files_recursive:
            return "structure"

        raise ValueError(
            f"Cannot determine format for directory {path}. "
            f"No parquet or structure files found."
        )

    raise ValueError(f"Path is neither a file nor a directory: {path}")
