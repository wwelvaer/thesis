from smiles_transformer import SmilesTransformer
from mol_tokenizers import SmilesBPETokenizer, SelfiesTokenizer, SelfiesBPETokenizer, DeepSmilesBPETokenizer
import pickle

import sys

import os

# Force the progress bar to show regardless of TTY detection.
os.environ["CARGO_TERM_PROGRESS_WHEN"] = "always"
# Set the progress bar width to 80 columns.
os.environ["CARGO_TERM_PROGRESS_WIDTH"] = "80"

cache_dir = "cache"

def main():
    # Ensure at least 3 arguments are provided (excluding the script name)
    if len(sys.argv) < 3:
        print("Error: 2 Arguments are required: representation and dataset_size.")
        sys.exit(1)
    
    # Get the arguments and process them
    representation = sys.argv[1].lower()
    dataset_size_str = sys.argv[2]
    vocab_size_str = sys.argv[3] if len(sys.argv) > 3 else None
    encoded_selfies_path = sys.argv[4] if len(sys.argv) > 4 else None
    min_frequency_str = sys.argv[5] if len(sys.argv) > 5 else None
    
    # Validate and convert dataset_size to an integer
    try:
        dataset_size = int(dataset_size_str)
        vocab_size = int(vocab_size_str) if vocab_size_str is not None else 30_000
        min_frequency = int(min_frequency_str) if min_frequency_str is not None else 2
    except ValueError:
        print("Error: dataset_size, vocab_size and min_frequency must be integers.")
        sys.exit(1)

    if representation == "smiles_bpe":
        tokenizer = SmilesBPETokenizer(dataset_size=dataset_size, max_len=200, cache_dir=cache_dir, vocab_size=vocab_size)
        filename = f"tokenizers/{representation}_tokenizer_{dataset_size}M_vocab_{tokenizer.get_vocab_size()}.pkl"
    elif representation == "selfies":
        tokenizer = SelfiesTokenizer(max_len=150)
        filename = f"tokenizers/{representation}_tokenizer.pkl"
    elif representation == "selfies_bpe":
        tokenizer = SelfiesBPETokenizer(max_len=150, encoded_selfies=encoded_selfies_path, min_frequency=min_frequency)
        filename = f"tokenizers/{representation}_tokenizer_{dataset_size}M_vocab_{tokenizer.get_vocab_size()}.pkl"
    elif representation == "deepsmiles_bpe":
        tokenizer = DeepSmilesBPETokenizer(dataset_size=dataset_size, max_len=200, cache_dir=cache_dir, vocab_size=vocab_size)
        filename = f"tokenizers/{representation}_tokenizer_{dataset_size}M_vocab_{tokenizer.get_vocab_size()}.pkl"
    else:
        raise AssertionError("Unknown representation")

    # store tokenizer
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"saved as {filename}")

if __name__ == "__main__":
    main()
