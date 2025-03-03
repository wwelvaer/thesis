from smiles_transformer import SmilesTransformer
from mol_tokenizers import SmilesBPETokenizer
import pickle

import sys

cache_dir = "~/scratch_kyukon/cache"

def main():
    # Ensure exactly 2 arguments are provided (excluding the script name)
    if len(sys.argv) != 3:
        print("Error: Two arguments are required: representation and dataset_size.")
        sys.exit(1)
    
    # Get the arguments and process them
    representation = sys.argv[1].lower()
    dataset_size_str = sys.argv[2]
    
    # Validate the representation
    if representation not in ["smiles", "selfies"]:
        print("Error: representation must be 'smiles' or 'selfies'.")
        sys.exit(1)
    
    # Validate and convert dataset_size to an integer
    try:
        dataset_size = int(dataset_size_str)
    except ValueError:
        print("Error: dataset_size must be an integer.")
        sys.exit(1)
    
    if representation == "smiles":
        tokenizer = SmilesBPETokenizer(dataset_size=dataset_size, max_len=200, cache_dir=cache_dir)
    
    # store tokenizer
    filename = f"tokenizers/{representation}_tokenizer_{dataset_size}M.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"saved as {filename}")

if __name__ == "__main__":
    main()
