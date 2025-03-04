from smiles_transformer import SmilesTransformer
from mol_tokenizers import SmilesBPETokenizer
import pickle

import sys

cache_dir = "cache"

def main():
    # Ensure exactly 3 arguments are provided (excluding the script name)
    if len(sys.argv) != 4:
        print("Error: Two arguments are required: representation and dataset_size.")
        sys.exit(1)
    
    # Get the arguments and process them
    representation = sys.argv[1].lower()
    dataset_size_str = sys.argv[2]
    vocab_size_str = sys.argv[3]
    
    # Validate the representation
    if representation not in ["smiles", "selfies"]:
        print("Error: representation must be 'smiles' or 'selfies'.")
        sys.exit(1)
    
    # Validate and convert dataset_size to an integer
    try:
        dataset_size = int(dataset_size_str)
        vocab_size = int(vocab_size_str)
    except ValueError:
        print("Error: dataset_size and vocab_size must be integers.")
        sys.exit(1)

    if representation == "smiles":
        tokenizer = SmilesBPETokenizer(dataset_size=dataset_size, max_len=200, cache_dir=cache_dir, vocab_size=vocab_size)
    
    # store tokenizer
    filename = f"tokenizers/{representation}_tokenizer_{dataset_size}M_vocab_{vocab_size}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"saved as {filename}")

if __name__ == "__main__":
    main()
