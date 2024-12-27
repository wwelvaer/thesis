from pyteomics import mgf
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import h5torch
import sys

input_file = str(sys.argv[1]) # GNPS MGF
output_file = str(sys.argv[2]) # output h5torch file

reader = mgf.MGF(input_file)

def check_conditions(spectrum):
    if spectrum["params"]["charge"][0] not in [0, 1]:
        return False
    if int(spectrum["params"]["libraryquality"]) > 3:
        return False
    if len(spectrum["m/z array"]) < 6:
        return False
    if np.max(spectrum["m/z array"]) > 1000:
        return False
    if spectrum["params"]["ionmode"] != "Positive":
        return False
    if not spectrum["params"]["name"].rstrip().endswith(" M+H"):
        return False
    if (spectrum["params"]["smiles"] == "N/A") or (spectrum["params"]["smiles"] == "") or (spectrum["params"]["smiles"] == " "):
        return False
    if not all(spectrum["intensity array"] > 0):
        return False
    try:
        molecule = Chem.MolFromSmiles(spectrum["params"]["smiles"])
        AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=4096)
    except:
        print("failed")
        return False
    return True

def filter_highest_peaks(mz, intensity, n = 128):
    indices = np.argsort(-intensity, kind="stable")
    take = np.sort(indices[:n])

    return mz[take], intensity[take]

mzs = []
intensities = []
smiles = []
organisms = []
fingerprints = []
for spectrum in reader:
    if check_conditions(spectrum):
        int_normed = spectrum["intensity array"] / spectrum["intensity array"].sum()
        mz, intensity = filter_highest_peaks(spectrum["m/z array"], int_normed, n = 128)
        mzs.append(mz)
        intensities.append(intensity)
        smiles.append(spectrum["params"]["smiles"])
        organisms.append(spectrum["params"]["organism"])

        molecule = Chem.MolFromSmiles(spectrum["params"]["smiles"])
        fingerprints.append(np.array(
            AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=4096)
        ))

reader.close()


f = h5torch.File(output_file, "w")

f.register([np.array(list(s)).astype(bytes) for s in smiles], "central", mode="vlen")
f.register(intensities, axis=0, name="intensities", mode="vlen", dtype_save="float32", dtype_load="float32")
f.register(mzs, axis=0, name="mzs", mode="vlen", dtype_save="float32", dtype_load="float32")
f.register(np.array(organisms).astype(bytes), axis=0, name="organisms", mode="N-D")
f.register(np.stack(fingerprints), axis=0, name="fingerprint", mode="N-D", dtype_save="bool", dtype_load="int64")

f.close()