from smiles_transformer import SmilesTransformer
from mol_tokenizers import SmilesBPETokenizer, SelfiesTokenizer, SelfiesBPETokenizer, DeepSmilesBPETokenizer, LayeredInchIBPETokenizer
import pickle
import argparse

import os

cache_dir = "cache"

parser = argparse.ArgumentParser()

# Submission
parser.add_argument('--representation', type=str, required=True)
parser.add_argument('--dataset_size', type=int, required=True)
parser.add_argument('--vocab_size', type=int, required=False, default=30_000)
parser.add_argument('--min_frequency', type=int, required=False, default=2)
parser.add_argument('--encoded_selfies_path', type=str, required=False, default=None)
parser.add_argument('--filterStereochemistry', type=bool, required=False, default=False)

def main():
    args = parser.parse_args([] if "__file__" not in globals() else None)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if args.representation == "smiles_bpe":
        tokenizer = SmilesBPETokenizer(dataset_size=args.dataset_size, max_len=200, cache_dir=cache_dir, vocab_size=args.vocab_size, min_frequency=args.min_frequency, filterStereochemistry=args.filterStereochemistry)
        filename = f"tokenizers/{args.representation}_tokenizer_{args.dataset_size}M_vocab_{tokenizer.get_vocab_size()}{'_no-stereo' if args.filterStereochemistry else ''}.pkl"
    elif args.representation == "selfies":
        tokenizer = SelfiesTokenizer(max_len=150)
        filename = f"tokenizers/{args.representation}_tokenizer.pkl"
    elif args.representation == "selfies_bpe":
        tokenizer = SelfiesBPETokenizer(max_len=150, encoded_selfies=args.encoded_selfies_path, min_frequency=args.min_frequency, vocab_size=args.vocab_size)
        filename = f"tokenizers/{args.representation}_tokenizer_{args.dataset_size}M_vocab_{tokenizer.get_vocab_size()}.pkl"
    elif args.representation == "deepsmiles_bpe":
        tokenizer = DeepSmilesBPETokenizer(dataset_size=args.dataset_size, max_len=200, cache_dir=cache_dir, vocab_size=args.vocab_size)
        filename = f"tokenizers/{args.representation}_tokenizer_{args.dataset_size}M_vocab_{tokenizer.get_vocab_size()}.pkl"
    elif args.representation == "layered_inchi":
        tokenizer = LayeredInchIBPETokenizer(dataset_size=args.dataset_size, max_len=200, cache_dir=cache_dir, vocab_size=args.vocab_size)
        filename = f"tokenizers/{args.representation}_tokenizer_{args.dataset_size}M_vocab_{"-".join(tokenizer.get_vocab_sizes())}.pkl"
    else:
        raise AssertionError("Unknown representation")

    # store tokenizer
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"saved as {filename}")

if __name__ == "__main__":
    main()
