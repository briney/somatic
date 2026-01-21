"""Tests for dataset classes."""

import pytest
import pandas as pd
import torch

from somatic.data.dataset import (
    AntibodyDataset,
    MultiDataset,
    StructureDataset,
    detect_dataset_format,
    parse_structure,
    StructureData,
    STRUCTURE_EXTENSIONS,
    SEQUENCE_EXTENSIONS,
)


# Minimal PDB file content for testing
MINIMAL_PDB_CONTENT = """\
HEADER    TEST STRUCTURE
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.320   1.540   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.970   2.840   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.480   2.720   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.040   1.620   0.000  1.00  0.00           O
ATOM      9  N   SER A   3       6.120   3.860   0.000  1.00  0.00           N
ATOM     10  CA  SER A   3       7.570   3.960   0.000  1.00  0.00           C
ATOM     11  C   SER A   3       8.190   2.570   0.000  1.00  0.00           C
ATOM     12  O   SER A   3       7.440   1.590   0.000  1.00  0.00           O
END
"""

# PDB with two chains
TWO_CHAIN_PDB_CONTENT = """\
HEADER    TWO CHAIN TEST
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   GLY B   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      5  CA  GLY B   1      11.458  10.000  10.000  1.00  0.00           C
ATOM      6  C   GLY B   1      12.009  11.420  10.000  1.00  0.00           C
END
"""

# Minimal mmCIF content with all required fields for BioPython
MINIMAL_CIF_CONTENT = """\
data_TEST
#
_entry.id TEST
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1 ? 0.000 0.000 0.000 1.00 0.00 ? 1 ALA A N 1
ATOM 2 C CA . ALA A 1 1 ? 1.458 0.000 0.000 1.00 0.00 ? 1 ALA A CA 1
ATOM 3 C C . ALA A 1 1 ? 2.009 1.420 0.000 1.00 0.00 ? 1 ALA A C 1
ATOM 4 N N . GLY A 1 2 ? 3.320 1.540 0.000 1.00 0.00 ? 2 GLY A N 1
ATOM 5 C CA . GLY A 1 2 ? 3.970 2.840 0.000 1.00 0.00 ? 2 GLY A CA 1
ATOM 6 C C . GLY A 1 2 ? 5.480 2.720 0.000 1.00 0.00 ? 2 GLY A C 1
#
"""


class TestAntibodyDataset:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        data = {
            "heavy_chain": ["EVQLVESGGGLVQPGGSLRL", "QVQLQQSGAELARPGASVKM"],
            "light_chain": ["DIQMTQSPSSLSASVGDRVT", "DIVMTQSPDSLAVSLGERAT"],
        }
        df = pd.DataFrame(data)
        path = tmp_path / "test_data.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.fixture
    def sample_csv_with_masks(self, tmp_path):
        """Sample CSV with detailed CDR masks (0=FW, 1=CDR1, 2=CDR2, 3=CDR3)."""
        data = {
            "heavy_chain": ["EVQLVESGGGLVQPGGSLRL", "QVQLQQSGAELARPGASVKM"],
            "light_chain": ["DIQMTQSPSSLSASVGDRVT", "DIVMTQSPDSLAVSLGERAT"],
            # Detailed CDR mask format: 0=FW, 1=CDR1, 2=CDR2, 3=CDR3
            "heavy_cdr_mask": ["00011100002220003330", "00001110002220003330"],
            "light_cdr_mask": ["00000111002220003330", "00000011100222000333"],
        }
        df = pd.DataFrame(data)
        path = tmp_path / "test_data_masks.csv"
        df.to_csv(path, index=False)
        return path

    def test_load_csv(self, sample_csv):
        dataset = AntibodyDataset(sample_csv)
        assert len(dataset) == 2

    def test_getitem(self, sample_csv):
        dataset = AntibodyDataset(sample_csv)
        item = dataset[0]

        assert "heavy_chain" in item
        assert "light_chain" in item
        assert item["heavy_chain"] == "EVQLVESGGGLVQPGGSLRL"
        assert item["light_chain"] == "DIQMTQSPSSLSASVGDRVT"

    def test_getitem_with_masks(self, sample_csv_with_masks):
        dataset = AntibodyDataset(sample_csv_with_masks)
        item = dataset[0]

        assert "heavy_cdr_mask" in item
        assert "light_cdr_mask" in item
        assert item["heavy_cdr_mask"] is not None
        assert len(item["heavy_cdr_mask"]) == 20

    def test_detailed_cdr_mask_values(self, sample_csv_with_masks):
        """Test that detailed CDR mask values (0,1,2,3) are preserved."""
        dataset = AntibodyDataset(sample_csv_with_masks)
        item = dataset[0]

        mask = item["heavy_cdr_mask"]
        # Should contain values 0, 1, 2, 3 for FW, CDR1, CDR2, CDR3
        assert 0 in mask  # Framework
        assert 1 in mask  # CDR1
        assert 2 in mask  # CDR2
        assert 3 in mask  # CDR3
        assert max(mask) == 3
        assert min(mask) == 0

    def test_csv_mask_dtype_preserved(self, tmp_path):
        """Test that mask columns with leading zeros are read correctly."""
        data = {
            "heavy_chain": ["EVQLVE"],
            "light_chain": ["DIQMTQ"],
            # Leading zeros should be preserved (not read as int)
            "heavy_cdr_mask": ["001111"],
            "light_cdr_mask": ["001111"],
        }
        df = pd.DataFrame(data)
        path = tmp_path / "test_dtype.csv"
        df.to_csv(path, index=False)

        dataset = AntibodyDataset(path)
        item = dataset[0]

        # Should have 6 values, not 4 (if it was misread as int 1111)
        assert len(item["heavy_cdr_mask"]) == 6
        assert item["heavy_cdr_mask"] == [0, 0, 1, 1, 1, 1]

    def test_missing_columns(self, tmp_path):
        data = {"wrong_col": ["EVQL"]}
        df = pd.DataFrame(data)
        path = tmp_path / "bad_data.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError):
            AntibodyDataset(path)

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("dummy")

        with pytest.raises(ValueError):
            AntibodyDataset(path)


class TestMultiDataset:
    @pytest.fixture
    def two_datasets(self, tmp_path):
        # Dataset 1
        data1 = {
            "heavy_chain": ["EVQLVESGGGLVQPGGSLRL"],
            "light_chain": ["DIQMTQSPSSLSASVGDRVT"],
        }
        df1 = pd.DataFrame(data1)
        path1 = tmp_path / "data1.csv"
        df1.to_csv(path1, index=False)

        # Dataset 2
        data2 = {
            "heavy_chain": ["QVQLQQSGAELARPGASVKM", "EVQLLESGGGLVQPGGSLRL"],
            "light_chain": ["DIVMTQSPDSLAVSLGERAT", "EIVMTQSPATLSVSPGERAT"],
        }
        df2 = pd.DataFrame(data2)
        path2 = tmp_path / "data2.csv"
        df2.to_csv(path2, index=False)

        ds1 = AntibodyDataset(path1)
        ds2 = AntibodyDataset(path2)

        return {"ds1": ds1, "ds2": ds2}

    def test_combined_length(self, two_datasets):
        multi = MultiDataset(two_datasets)
        assert len(multi) == 3  # 1 + 2

    def test_getitem(self, two_datasets):
        multi = MultiDataset(two_datasets)
        item = multi[0]

        assert "heavy_chain" in item
        assert "_dataset" in item

    def test_sampling_weights(self, two_datasets):
        multi = MultiDataset(two_datasets)
        weights = multi.get_sampler_weights()

        assert len(weights) == 3
        assert abs(weights.sum().item() - 1.0) < 1e-6

    def test_custom_weights(self, two_datasets):
        multi = MultiDataset(two_datasets, weights={"ds1": 2.0, "ds2": 1.0})

        # ds1 has 1 sample with weight 2/3, ds2 has 2 samples with weight 1/3 total
        weights = multi.get_sampler_weights()
        assert len(weights) == 3


class TestStructureDataset:
    """Tests for StructureDataset class."""

    @pytest.fixture
    def pdb_folder(self, tmp_path):
        """Create a folder with PDB files."""
        folder = tmp_path / "structures"
        folder.mkdir()

        # Create PDB files
        (folder / "struct1.pdb").write_text(MINIMAL_PDB_CONTENT)
        (folder / "struct2.pdb").write_text(MINIMAL_PDB_CONTENT)

        return folder

    @pytest.fixture
    def mixed_structure_folder(self, tmp_path):
        """Create a folder with mixed PDB and mmCIF files."""
        folder = tmp_path / "mixed_structures"
        folder.mkdir()

        (folder / "struct1.pdb").write_text(MINIMAL_PDB_CONTENT)
        (folder / "struct2.cif").write_text(MINIMAL_CIF_CONTENT)

        return folder

    @pytest.fixture
    def nested_structure_folder(self, tmp_path):
        """Create a folder with nested subdirectories containing structures."""
        folder = tmp_path / "nested"
        folder.mkdir()

        subdir1 = folder / "subdir1"
        subdir1.mkdir()
        (subdir1 / "struct1.pdb").write_text(MINIMAL_PDB_CONTENT)

        subdir2 = folder / "subdir2"
        subdir2.mkdir()
        (subdir2 / "struct2.pdb").write_text(MINIMAL_PDB_CONTENT)

        return folder

    def test_load_pdb_folder(self, pdb_folder):
        dataset = StructureDataset(pdb_folder, max_length=100)
        assert len(dataset) == 2

    def test_getitem_returns_expected_keys(self, pdb_folder):
        dataset = StructureDataset(pdb_folder, max_length=100)
        item = dataset[0]

        assert "pid" in item
        assert "seq" in item
        assert "coords" in item
        assert "masks" in item
        assert "nan_masks" in item

    def test_getitem_types(self, pdb_folder):
        dataset = StructureDataset(pdb_folder, max_length=100)
        item = dataset[0]

        assert isinstance(item["pid"], str)
        assert isinstance(item["seq"], str)
        assert isinstance(item["coords"], torch.Tensor)
        assert isinstance(item["masks"], torch.Tensor)
        assert isinstance(item["nan_masks"], torch.Tensor)

    def test_getitem_shapes(self, pdb_folder):
        max_length = 100
        dataset = StructureDataset(pdb_folder, max_length=max_length)
        item = dataset[0]

        assert item["coords"].shape == (max_length, 3, 3)
        assert item["masks"].shape == (max_length,)
        assert item["nan_masks"].shape == (max_length,)

    def test_sequence_extracted_correctly(self, pdb_folder):
        dataset = StructureDataset(pdb_folder, max_length=100)
        item = dataset[0]

        # The minimal PDB has ALA, GLY, SER = "AGS"
        assert item["seq"] == "AGS"

    def test_masks_correct(self, pdb_folder):
        max_length = 100
        dataset = StructureDataset(pdb_folder, max_length=max_length)
        item = dataset[0]

        # First 3 positions should be True (valid), rest False (padding)
        assert item["masks"][:3].all()
        assert not item["masks"][3:].any()

    def test_coords_padded_with_nan(self, pdb_folder):
        max_length = 100
        dataset = StructureDataset(pdb_folder, max_length=max_length)
        item = dataset[0]

        # First 3 residues should have real coords
        assert not torch.isnan(item["coords"][:3]).any()
        # Rest should be NaN
        assert torch.isnan(item["coords"][3:]).all()

    def test_truncation(self, pdb_folder):
        # Set max_length shorter than sequence
        max_length = 2
        dataset = StructureDataset(pdb_folder, max_length=max_length)
        item = dataset[0]

        assert len(item["seq"]) == 2
        assert item["seq"] == "AG"  # First two residues
        assert item["coords"].shape == (2, 3, 3)

    def test_mixed_formats(self, mixed_structure_folder):
        dataset = StructureDataset(mixed_structure_folder, max_length=100)
        assert len(dataset) == 2

    def test_recursive_search(self, nested_structure_folder):
        # Without recursive, should find nothing
        with pytest.raises(ValueError, match="No structure files found"):
            StructureDataset(nested_structure_folder, max_length=100, recursive=False)

        # With recursive, should find 2 files
        dataset = StructureDataset(
            nested_structure_folder, max_length=100, recursive=True
        )
        assert len(dataset) == 2

    def test_not_a_directory_raises(self, tmp_path):
        file_path = tmp_path / "not_a_dir.pdb"
        file_path.write_text(MINIMAL_PDB_CONTENT)

        with pytest.raises(ValueError, match="Not a directory"):
            StructureDataset(file_path, max_length=100)

    def test_empty_folder_raises(self, tmp_path):
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        with pytest.raises(ValueError, match="No structure files found"):
            StructureDataset(empty_folder, max_length=100)

    def test_has_coords_flag(self, pdb_folder):
        dataset = StructureDataset(pdb_folder, max_length=100)
        assert dataset.has_coords is True

    def test_repr(self, pdb_folder):
        dataset = StructureDataset(pdb_folder, max_length=100)
        repr_str = repr(dataset)

        assert "StructureDataset" in repr_str
        assert "num_files=2" in repr_str
        assert "max_length=100" in repr_str


class TestParseStructure:
    """Tests for parse_structure function."""

    @pytest.fixture
    def pdb_file(self, tmp_path):
        path = tmp_path / "test.pdb"
        path.write_text(MINIMAL_PDB_CONTENT)
        return path

    @pytest.fixture
    def two_chain_pdb(self, tmp_path):
        path = tmp_path / "two_chain.pdb"
        path.write_text(TWO_CHAIN_PDB_CONTENT)
        return path

    @pytest.fixture
    def cif_file(self, tmp_path):
        path = tmp_path / "test.cif"
        path.write_text(MINIMAL_CIF_CONTENT)
        return path

    def test_parse_pdb(self, pdb_file):
        result = parse_structure(pdb_file)

        assert isinstance(result, StructureData)
        assert result.pid == "test"
        assert result.protein_sequence == "AGS"
        assert result.coords.shape == (3, 3, 3)  # 3 residues, 3 atoms, 3 coords
        assert result.chain_id == "A"

    def test_parse_cif(self, cif_file):
        result = parse_structure(cif_file)

        assert isinstance(result, StructureData)
        assert result.protein_sequence == "AG"
        assert result.coords.shape == (2, 3, 3)

    def test_specific_chain_id(self, two_chain_pdb):
        # Get chain A (1 residue)
        result_a = parse_structure(two_chain_pdb, chain_id="A")
        assert result_a.protein_sequence == "A"
        assert result_a.chain_id == "A"

        # Get chain B (1 residue)
        result_b = parse_structure(two_chain_pdb, chain_id="B")
        assert result_b.protein_sequence == "G"
        assert result_b.chain_id == "B"

    def test_chain_not_found_raises(self, pdb_file):
        with pytest.raises(ValueError, match="Chain 'Z' not found"):
            parse_structure(pdb_file, chain_id="Z")

    def test_file_not_found_raises(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.pdb"
        with pytest.raises(FileNotFoundError):
            parse_structure(nonexistent)

    def test_coords_shape(self, pdb_file):
        result = parse_structure(pdb_file)

        # Each residue should have N, CA, C coords (3 atoms, 3 xyz)
        assert result.coords.shape[1] == 3  # 3 backbone atoms
        assert result.coords.shape[2] == 3  # x, y, z


class TestDetectDatasetFormat:
    """Tests for detect_dataset_format function."""

    def test_csv_file(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.touch()
        assert detect_dataset_format(csv_file) == "sequence"

    def test_tsv_file(self, tmp_path):
        tsv_file = tmp_path / "data.tsv"
        tsv_file.touch()
        assert detect_dataset_format(tsv_file) == "sequence"

    def test_parquet_file(self, tmp_path):
        parquet_file = tmp_path / "data.parquet"
        parquet_file.touch()
        assert detect_dataset_format(parquet_file) == "sequence"

    def test_pdb_file(self, tmp_path):
        pdb_file = tmp_path / "structure.pdb"
        pdb_file.touch()
        assert detect_dataset_format(pdb_file) == "structure"

    def test_cif_file(self, tmp_path):
        cif_file = tmp_path / "structure.cif"
        cif_file.touch()
        assert detect_dataset_format(cif_file) == "structure"

    def test_mmcif_file(self, tmp_path):
        mmcif_file = tmp_path / "structure.mmcif"
        mmcif_file.touch()
        assert detect_dataset_format(mmcif_file) == "structure"

    def test_ent_file(self, tmp_path):
        ent_file = tmp_path / "structure.ent"
        ent_file.touch()
        assert detect_dataset_format(ent_file) == "structure"

    def test_parquet_directory(self, tmp_path):
        parquet_dir = tmp_path / "parquet_shards"
        parquet_dir.mkdir()
        (parquet_dir / "shard1.parquet").touch()
        (parquet_dir / "shard2.parquet").touch()

        assert detect_dataset_format(parquet_dir) == "sequence"

    def test_pdb_directory(self, tmp_path):
        pdb_dir = tmp_path / "structures"
        pdb_dir.mkdir()
        (pdb_dir / "struct1.pdb").touch()
        (pdb_dir / "struct2.pdb").touch()

        assert detect_dataset_format(pdb_dir) == "structure"

    def test_mixed_structure_directory(self, tmp_path):
        mixed_dir = tmp_path / "mixed"
        mixed_dir.mkdir()
        (mixed_dir / "struct1.pdb").touch()
        (mixed_dir / "struct2.cif").touch()

        assert detect_dataset_format(mixed_dir) == "structure"

    def test_nested_structure_directory(self, tmp_path):
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        subdir = nested_dir / "subdir"
        subdir.mkdir()
        (subdir / "struct.pdb").touch()

        # Should find structure files recursively
        assert detect_dataset_format(nested_dir) == "structure"

    def test_unknown_extension_raises(self, tmp_path):
        unknown_file = tmp_path / "data.xyz"
        unknown_file.touch()

        with pytest.raises(ValueError, match="Unknown file extension"):
            detect_dataset_format(unknown_file)

    def test_nonexistent_path_raises(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(ValueError, match="Path does not exist"):
            detect_dataset_format(nonexistent)

    def test_empty_directory_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="Cannot determine format"):
            detect_dataset_format(empty_dir)

    def test_case_insensitive_extensions(self, tmp_path):
        upper_pdb = tmp_path / "struct.PDB"
        upper_pdb.touch()
        assert detect_dataset_format(upper_pdb) == "structure"

        upper_csv = tmp_path / "data.CSV"
        upper_csv.touch()
        assert detect_dataset_format(upper_csv) == "sequence"


class TestExtensionConstants:
    """Tests for extension constant sets."""

    def test_structure_extensions(self):
        assert ".pdb" in STRUCTURE_EXTENSIONS
        assert ".ent" in STRUCTURE_EXTENSIONS
        assert ".cif" in STRUCTURE_EXTENSIONS
        assert ".mmcif" in STRUCTURE_EXTENSIONS

    def test_sequence_extensions(self):
        assert ".csv" in SEQUENCE_EXTENSIONS
        assert ".tsv" in SEQUENCE_EXTENSIONS
        assert ".parquet" in SEQUENCE_EXTENSIONS

    def test_no_overlap(self):
        # Ensure no extension is in both sets
        assert STRUCTURE_EXTENSIONS.isdisjoint(SEQUENCE_EXTENSIONS)
