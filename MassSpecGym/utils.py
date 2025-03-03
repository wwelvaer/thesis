import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import pandas as pd
import typing as T
import pulp
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import groupby
from pathlib import Path
from myopic_mces.myopic_mces import MCES
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs, Draw
from rdkit.Chem.Descriptors import ExactMolWt
from huggingface_hub import hf_hub_download
from torchmetrics.wrappers import BootStrapper
from torchmetrics.metric import Metric


def load_massspecgym(fold: T.Optional[str] = None) -> pd.DataFrame:
    """
    Load the MassSpecGym dataset.

    Args:
        fold (str, optional): Fold name to load. If None, the entire dataset is loaded.
    """
    df = pd.read_csv(hugging_face_download("MassSpecGym.tsv"), sep="\t")
    df = df.set_index("identifier")
    df['mzs'] = df['mzs'].apply(parse_spec_array)
    df['intensities'] = df['intensities'].apply(parse_spec_array)
    if fold is not None:
        df = df[df['fold'] == fold]
    return df


def load_unlabeled_mols(col_name: str = "smiles", size: int = 4, cache_dir=None) -> pd.Series:
    """
    Load a list of unlabeled molecules.

    Args:
        col_name (str, optional): Name of the column to return. Should be one of ["smiles", "selfies"].
        size (int, optional): Size of dataset in Millions of entries. Should be one of [1, 4, 118].
    """
    datasets = {
        1: "molecules/candidate_pools/MassSpecGym_retrieval_molecules_1M.tsv",
        4: "molecules/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv",
        118: "molecules/candidate_pools/MassSpecGym_retrieval_molecules_pubchem_118M.tsv"
    }
    
    assert size in datasets.keys(), f"Unknown dataset size {size}, valid sizes: {datasets.keys()}"
    print(f"Fetching: {datasets[size]}")
    return pd.read_csv(
        hugging_face_download(datasets[size], cache_dir=cache_dir),
        sep="\t"
    )[col_name]


def load_massspecgym_mols(fold: T.Optional[str] = None, unique: bool = True) -> pd.Series:
    """
    Load a list of molecules from the MassSpecGym dataset.

    Args:
        fold (str, optional): Fold name to load. If None, the entire dataset is loaded.
        unique (bool, optional): Whether to return only unique molecules.
    """
    df = load_massspecgym(fold)
    mols = df["smiles"]
    if unique:
        mols = pd.Series(mols.unique())
    return mols


def load_train_mols(unique: bool = True) -> pd.Series:
    """
    Load a list of training molecules.

    Args:
        unique (bool, optional): Whether to return only unique molecules.
    """
    return load_massspecgym_mols("train", unique=unique)


def load_val_mols(unique: bool = True) -> pd.Series:
    """
    Load a list of validation molecules.

    Args:
        unique (bool, optional): Whether to return only unique molecules.
    """
    return load_massspecgym_mols("val", unique=unique)


def pad_spectrum(
    spec: np.ndarray, max_n_peaks: int, pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad a spectrum to a fixed number of peaks by appending zeros to the end of the spectrum.
    
    Args:
        spec (np.ndarray): Spectrum to pad represented as numpy array of shape (n_peaks, 2).
        max_n_peaks (int): Maximum number of peaks in the padded spectrum.
        pad_value (float, optional): Value to use for padding.
    """
    n_peaks = spec.shape[0]
    if n_peaks > max_n_peaks:
        raise ValueError(
            f"Number of peaks in the spectrum ({n_peaks}) is greater than the maximum number of peaks."
        )
    else:
        return np.pad(
            spec,
            ((0, max_n_peaks - n_peaks), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )


def morgan_fp(mol: Chem.Mol, fp_size=2048, radius=2, to_np=True):
    """
    Compute Morgan fingerprint for a molecule.
    
    Args:
        mol (Chem.Mol): _description_
        fp_size (int, optional): Size of the fingerprint.
        radius (int, optional): Radius of the fingerprint.
        to_np (bool, optional): Convert the fingerprint to numpy array.
    """

    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    if to_np:
        fp_np = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        fp = fp_np
    return fp


def tanimoto_morgan_similarity(mol1: T.Union[Chem.Mol, str], mol2: T.Union[Chem.Mol, str]) -> float:
    """
    Compute Tanimoto similarity between two molecules using Morgan fingerprints.

    Args:
        mol1 (T.Union[Chem.Mol, str]): First molecule as RDKit molecule or SMILES string.
        mol2 (T.Union[Chem.Mol, str]): Second molecule as RDKit molecule or SMILES string.
    """
    if isinstance(mol1, str):
        mol1 = Chem.MolFromSmiles(mol1)
    if isinstance(mol2, str):
        mol2 = Chem.MolFromSmiles(mol2)
    return DataStructs.TanimotoSimilarity(morgan_fp(mol1, to_np=False), morgan_fp(mol2, to_np=False))


def standardize_smiles(smiles: T.Union[str, T.List[str]]) -> T.Union[str, T.List[str]]:
    """
    Standardize SMILES representation of a molecule using PubChem standardization.
    """
    try:
        from standardizeUtils.standardizeUtils import (
            standardize_structure_with_pubchem,
            standardize_structure_list_with_pubchem,
        )
    except ImportError:
        raise ImportError(
            "The standardizeUtils package is required for SMILES standardization. "
            "Please install it using: pip install "
            "git+https://github.com/boecker-lab/standardizeUtils@b415f1c51b49f6c5cd0e9c6ab89224c8ad657a35#egg=standardizeUtils"
        )

    if isinstance(smiles, str):
        return standardize_structure_with_pubchem(smiles, 'smiles')
    elif isinstance(smiles, list):
        return standardize_structure_list_with_pubchem(smiles, 'smiles')
    else:
        raise ValueError("Input should be a SMILES tring or a list of SMILES strings.")


def mol_to_inchi_key(mol: Chem.Mol, twod: bool = True) -> str:
    """
    Convert a molecule to InChI Key representation.
    
    Args:
        mol (Chem.Mol): RDKit molecule object.
        twod (bool, optional): Return 2D InChI Key (first 14 characers of InChI Key).
    """
    inchi_key = Chem.MolToInchiKey(mol)
    if twod:
        inchi_key = inchi_key.split("-")[0]
    return inchi_key


def smiles_to_inchi_key(mol: str, twod: bool = True) -> str:
    """
    Convert a SMILES molecule to InChI Key representation.
    
    Args:
        mol (str): SMILES string.
        twod (bool, optional): Return 2D InChI Key (first 14 characers of InChI Key).
    """
    mol = Chem.MolFromSmiles(mol)
    return mol_to_inchi_key(mol, twod)


def hugging_face_download(file_name: str, cache_dir=None) -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.
    
    Args:
        file_name (str): Name of the file to download.
    """
    return hf_hub_download(
        repo_id="roman-bushuiev/MassSpecGym",
        filename="data/" + file_name,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def init_plotting(figsize=(6, 2), font_scale=1.0, style="whitegrid"):
    # Set default figure size
    plt.show()  # Does not work without this line for some reason
    sns.set_theme(rc={"figure.figsize": figsize})
    mpl.rcParams['svg.fonttype'] = 'none'
    # Set default style and font scale
    sns.set_style(style)
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette(["#009473", "#D94F70", "#5A5B9F", "#F0C05A", "#7BC4C4", "#FF6F61"])


def parse_spec_array(arr: str) -> np.ndarray:
    return np.array(list(map(float, arr.split(","))))


def spec_array_to_str(arr: np.ndarray) -> str:
    return ",".join(map(str, arr))


def compute_mass(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    return ExactMolWt(mol)


def plot_spectrum(spec, hue=None, xlim=None, ylim=None, mirror_spec=None, highl_idx=None,
                  figsize=(6, 2), colors=None, save_pth=None):

    if colors is not None:
        assert len(colors) >= 3
    else:
        colors = ['blue', 'green', 'red']

    # Normalize input spectrum
    def norm_spec(spec):
        assert len(spec.shape) == 2
        if spec.shape[0] != 2:
            spec = spec.T
        mzs, ins = spec[0], spec[1]
        return mzs, ins / max(ins) * 100
    mzs, ins = norm_spec(spec)

    # Initialize plotting
    init_plotting(figsize=figsize)
    fig, ax = plt.subplots(1, 1)

    # Setup color palette
    if hue is not None:
        norm = matplotlib.colors.Normalize(vmin=min(hue), vmax=max(hue), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.cool)
        plt.colorbar(mapper, ax=ax)

    # Plot spectrum
    for i in range(len(mzs)):
        if hue is not None:
            color = mcolors.to_hex(mapper.to_rgba(hue[i]))
        else:
            color = colors[0]
        plt.plot([mzs[i], mzs[i]], [0, ins[i]], color=color, marker='o', markevery=(1, 2), mfc='white', zorder=2)

    # Plot mirror spectrum
    if mirror_spec is not None:
        mzs_m, ins_m = norm_spec(mirror_spec)

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            label = str(round(-x)) if x < 0 else str(round(x))
            return label

        for i in range(len(mzs_m)):
            plt.plot([mzs_m[i], mzs_m[i]], [0, -ins_m[i]], color=colors[2], marker='o', markevery=(1, 2), mfc='white',
                     zorder=1)
        ax.yaxis.set_major_formatter(major_formatter)

    # Setup axes
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(0, max(mzs) + 10)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel('m/z')
    plt.ylabel('Intensity [%]')

    if save_pth is not None:
        raise NotImplementedError()


def show_mols(mols, legends='new_indices', smiles_in=False, svg=False, sort_by_legend=False, max_mols=500,
              legend_float_decimals=4, mols_per_row=6, save_pth: T.Optional[Path] = None):
    """
    Returns svg image representing a grid of skeletal structures of the given molecules. Copy-pasted
     from https://github.com/pluskal-lab/DreaMS/blob/main/dreams/utils/mols.py

    :param mols: list of rdkit molecules
    :param smiles_in: True - SMILES inputs, False - RDKit mols
    :param legends: list of labels for each molecule, length must be equal to the length of mols
    :param svg: True - return svg image, False - return png image
    :param sort_by_legend: True - sort molecules by legend values
    :param max_mols: maximum number of molecules to show
    :param legend_float_decimals: number of decimal places to show for float legends
    :param mols_per_row: number of molecules per row to show
    :param save_pth: path to save the .svg image to
    """
    if smiles_in:
        mols = [Chem.MolFromSmiles(e) for e in mols]

    if legends == 'new_indices':
        legends = list(range(len(mols)))
    elif legends == 'masses':
        legends = [ExactMolWt(m) for m in mols]
    elif callable(legends):
        legends = [legends(e) for e in mols]

    if sort_by_legend:
        idx = np.argsort(legends).tolist()
        legends = [legends[i] for i in idx]
        mols = [mols[i] for i in idx]

    legends = [f'{l:.{legend_float_decimals}f}' if isinstance(l, float) else str(l) for l in legends]

    img = Draw.MolsToGridImage(mols, maxMols=max_mols, legends=legends, molsPerRow=min(max_mols, mols_per_row),
                         useSVG=svg, returnPNG=False)

    if save_pth:
        with open(save_pth, 'w') as f:
            f.write(img.data)

    return img


class MyopicMCES():
    def __init__(
        self,
        ind: int = 0,  # dummy index
        solver: str = pulp.listSolvers(onlyAvailable=True)[0],  # Use the first available solver
        threshold: int = 15,  # MCES threshold
        always_stronger_bound: bool = True, # "False" makes computations a lot faster, but leads to overall higher MCES values
        solver_options: dict = None
    ):
        self.ind = ind
        self.solver = solver
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        if solver_options is None:
            solver_options = dict(msg=0)  # make ILP solver silent
        self.solver_options = solver_options

    def __call__(self, smiles_1: str, smiles_2: str) -> float:
        retval = MCES(
            s1=smiles_1,
            s2=smiles_2,
            ind=self.ind,
            threshold=self.threshold,
            always_stronger_bound=self.always_stronger_bound,
            solver=self.solver,
            solver_options=self.solver_options
        )
        dist = retval[1]
        return dist


class ReturnScalarBootStrapper(BootStrapper):
    def __init__(
        self,
        base_metric: Metric,
        num_bootstraps: int = 10,
        mean: bool = False,
        std: bool = False,
        quantile: T.Optional[T.Union[float, torch.Tensor]] = None,
        raw: bool = False,
        sampling_strategy: str = "poisson",
        **kwargs: T.Any
    ) -> None:
        """Wrapper for BootStrapper that returns a scalar value in compute instead of a dictionary."""

        if mean + std + bool(quantile) + raw != 1:
            raise ValueError("Exactly one of mean, std, quantile or raw should be True.")

        if std:
            self.compute_key = "std"
        else:
            raise NotImplementedError("Currently only std is implemented.")

        super().__init__(
            base_metric=base_metric,
            num_bootstraps=num_bootstraps,
            mean=mean,
            std=std,
            quantile=quantile,
            raw=raw,
            sampling_strategy=sampling_strategy,
            **kwargs
        )

    def compute(self):
        return super().compute()[self.compute_key]


def batch_ptr_to_batch_idx(batch_ptr: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor of batch pointers to a tensor of batch indexes.
    
    For example [1, 3, 2] -> [0, 1, 1, 1, 2, 2]

    Args:
        batch_ptr (Tensor): Tensor of batch pointers.
    """
    indexes = torch.arange(batch_ptr.size(0), device=batch_ptr.device)
    indexes = torch.repeat_interleave(indexes, batch_ptr)
    return indexes


def unbatch_list(batch_list: list, batch_idx: torch.Tensor) -> list:
    """
    Unbatch a list of items using the batch indexes (i.e., number of samples per batch).
    
    Args:
        batch_list (list): List of items to unbatch.
        batch_idx (Tensor): Tensor of batch indexes.
    """
    return [
        [batch_list[j] for j in range(len(batch_list)) if batch_idx[j] == i]
        for i in range(batch_idx[-1] + 1)
    ]


class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, inputs, targets):
        return 1 - F.cosine_similarity(inputs, targets).mean()

      
def parse_sirius_ms(spectra_file: str) -> T.Tuple[dict, T.List[T.Tuple[str, np.ndarray]]]:
    """
    Parses spectra from the SIRIUS .ms file.

    Copied from the code of Goldman et al.:
    https://github.com/samgoldman97/mist/blob/4c23d34fc82425ad5474a53e10b4622dcdbca479/src/mist/utils/parse_utils.py#LL10C77-L10C77.
    :return T.Tuple[dict, T.List[T.Tuple[str, np.ndarray]]]: metadata and list of spectra tuples containing name and array
    """
    lines = [i.strip() for i in open(spectra_file, "r").readlines()]

    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for index, (start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        # Get spectra
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            # Check if spectra is empty
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                # Add new tuple
                spectras.append((spectra_header, peak_data))
        # Get meta data
        else:
            entries = {}
            for i in group_lines:
                if " " not in i:
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end

            metadata.update(entries)
        group_num += 1

    metadata["_FILE_PATH"] = spectra_file
    metadata["_FILE"] = Path(spectra_file).stem
    return metadata, spectras
