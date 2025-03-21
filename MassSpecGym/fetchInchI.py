import requests
import time
import pandas as pd
from pathlib import Path

def fetch_inchi_from_inchikey(inchikey):
    """Fetches the InChI string from PubChem using the InChIKey."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/InChI/TXT"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    else:
        print(f"Error fetching InChI for {inchikey}: HTTP {response.status_code}")
        return None

def batch_fetch_inchi(inchikey_list, delay=0.5):
    """Fetches InChI strings for a list of InChIKeys with rate limiting."""
    results = {}
    for inchikey in inchikey_list:
        results[inchikey] = fetch_inchi_from_inchikey(inchikey)
        time.sleep(delay)  # Prevents API rate-limiting issues
    return results

if __name__ == "__main__":
    pth = utils.hugging_face_download("MassSpecGym.tsv")
    pth = Path(pth)
    data = pd.read_csv(pth, sep="\t")
    # Example list of InChIKeys
    
    cache = {}
    inchis = []

    for key in data.inchikey:
        if key in cache:
            inchi = cache[key]
        else:
            inchi = fetch_inchi_from_inchikey(key)
            cache[key] = inchi
        inchis.append(inchi)

    data['inchi'] = selfies
    data.to_csv("MassSpecGym_with_inchi.tsv", sep="\t")
