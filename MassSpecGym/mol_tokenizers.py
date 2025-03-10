import pandas as pd
import typing as T
import selfies as sf
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer, processors, models, pre_tokenizers
from tokenizers.implementations import BaseTokenizer, ByteLevelBPETokenizer
import utils as utils
from massspecgym.definitions import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


class SpecialTokensBaseTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_len: T.Optional[int] = None,
    ):
        """Initialize the base tokenizer with special tokens performing padding and truncation."""
        super().__init__(tokenizer)

        # Save essential attributes
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        self.max_length = max_len

        # Add special tokens
        self.add_special_tokens([self.pad_token, self.sos_token, self.eos_token, self.unk_token])

        # Get token IDs
        self.pad_token_id = self.token_to_id(self.pad_token)
        self.sos_token_id = self.token_to_id(self.sos_token)
        self.eos_token_id = self.token_to_id(self.eos_token)
        self.unk_token_id = self.token_to_id(self.unk_token)

        # Enable padding
        self.enable_padding(
            direction="right",
            pad_token=self.pad_token,
            pad_id=self.pad_token_id,
            length=max_len,
        )

        # Enable truncation
        self.enable_truncation(max_len)

        # Set post-processing to add SOS and EOS tokens
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.sos_token} $A {self.eos_token}",
            pair=f"{self.sos_token} $A {self.eos_token} {self.sos_token} $B {self.eos_token}",
            special_tokens=[
                (self.sos_token, self.sos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )


class SelfiesTokenizer(SpecialTokensBaseTokenizer):
    def __init__(
            self,
            selfies_train: T.Optional[T.Union[str, T.List[str]]] = None,
            **kwargs
        ):
        """
        Initialize the SELFIES tokenizer with optional training data to build a vocabulary.

        Args:
            selfies_train (str or list of str): Either a list of SELFIES strings to build the vocabulary from,
                or a `semantic_robust_alphabet` string indicating the usahe of `selfies.get_semantic_robust_alphabet()`
                alphabet. If None, the MassSpecGym training molecules will be used.
        """

        if selfies_train == 'semantic_robust_alphabet':
            alphabet = list(sorted(sf.get_semantic_robust_alphabet()))
        else:
            if not selfies_train:
                selfies_train = utils.load_train_mols()
                selfies = [sf.encoder(s, strict=False) for s in selfies_train]
            else:
                selfies = selfies_train
            alphabet = list(sorted(sf.get_alphabet_from_selfies(selfies))) 

        vocab = {symbol: i for i, symbol in enumerate(alphabet)}
        vocab[UNK_TOKEN] = len(vocab)
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))

        super().__init__(tokenizer, **kwargs)

    def encode(self, text: str, add_special_tokens: bool = True) -> Tokenizer:
        """Encodes a SMILES string into a list of SELFIES token IDs."""
        selfies_string = sf.encoder(text, strict=False)
        selfies_tokens = list(sf.split_selfies(selfies_string))
        return super().encode(
            selfies_tokens, is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of SELFIES token IDs back into a SMILES string."""
        selfies_string = super().decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        selfies_string = self._decode_wordlevel_str_to_selfies(selfies_string)
        return sf.decoder(selfies_string)

    def encode_batch(
        self, texts: T.List[str], add_special_tokens: bool = True
    ) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of SELFIES token IDs."""
        selfies_strings = [
            list(sf.split_selfies(sf.encoder(text, strict=False))) for text in texts
        ]
        return super().encode_batch(
            selfies_strings, is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode_batch(
        self, token_ids_batch: T.List[T.List[int]], skip_special_tokens: bool = True
    ) -> T.List[str]:
        """Decodes a batch of SELFIES token IDs back into SMILES strings."""
        selfies_strings = super().decode_batch(
            token_ids_batch, skip_special_tokens=skip_special_tokens
        )
        return [
            sf.decoder(
                self._decode_wordlevel_str_to_selfies(
                    selfies_string
                )
            )
            for selfies_string in selfies_strings
        ]

    def _decode_wordlevel_str_to_selfies(self, text: str) -> str:
        """Converts a WordLevel string back to a SELFIES string."""
        return text.replace(" ", "")


class SmilesBPETokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, dataset_size: int=4, smiles_pth: T.Optional[str] = None, cache_dir=None, vocab_size=30000, **kwargs):
        """
        Initialize the BPE tokenizer for SMILES strings, with optional training data.
        """
        tokenizer = ByteLevelBPETokenizer()
        if smiles_pth:
            tokenizer.train(smiles_pth)
        else:
            smiles = utils.load_unlabeled_mols(col_name="smiles", size=dataset_size, cache_dir=cache_dir).tolist()
            smiles += utils.load_train_mols().tolist()
            print(f"Training tokenizer on {len(smiles)} SMILES strings.")
            tokenizer.train_from_iterator(smiles, vocab_size)

        super().__init__(tokenizer, **kwargs)

class SmilesTokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, **kwargs):
        smiles = utils.load_train_mols().tolist()
        smiles += utils.load_unlabeled_mols(col_name="smiles", size=4, cache_dir=None).tolist()

        vocab = {t: i for i,t in enumerate(set(y for x in smiles for y in list(x)))}

        vocab[UNK_TOKEN] = len(vocab)
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    
        super().__init__(tokenizer, **kwargs)

    def encode(self, text: str, add_special_tokens: bool = True) -> Tokenizer:
        """Encodes a SMILES string into a list of SMILES token IDs."""
        return super().encode(
            list(text), is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of SMILES token IDs back into a SMILES string."""
        smiles_string = super().decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return self._decode_wordlevel_str_to_smiles(smiles_string)

    def encode_batch(
        self, texts: T.List[str], add_special_tokens: bool = True
    ) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of SMILES token IDs."""
        smiles_strings = [
            list(text) for text in texts
        ]
        return super().encode_batch(
            smiles_strings, is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode_batch(
        self, token_ids_batch: T.List[T.List[int]], skip_special_tokens: bool = True
    ) -> T.List[str]:
        """Decodes a batch of SMILES token IDs back into SMILES strings."""
        smiles_strings = super().decode_batch(
            token_ids_batch, skip_special_tokens=skip_special_tokens
        )
        return [
            self._decode_wordlevel_str_to_smiles(smiles_string)
            for smiles_string in smiles_strings
        ]

    def _decode_wordlevel_str_to_smiles(self, text: str) -> str:
        """Converts a WordLevel string back to a SMILES string."""
        return text.replace(" ", "")

# TODO fix
class SelfiesBPETokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, dataset_size: int=4, cache_dir=None, vocab_size=30000, **kwargs):
        """
        Initialize the BPE tokenizer for SELFIES strings
        """
        tokenizer = ByteLevelBPETokenizer()

        smiles = utils.load_train_mols().tolist()
        #smiles += utils.load_unlabeled_mols(col_name="smiles", size=dataset_size, cache_dir=cache_dir).tolist()

        print(f"Converting {len(smiles)} SMILES strings to SELFIES strings")
        selfies_tokens = [
            sf.split_selfies(sf.encoder(s, strict=False)) for s in smiles
        ]

        print(f"Calculating SELFIES token to byte mapping")
        unique_selfies_tokens = sorted(set([y for x in selfies_tokens for y in x]))
        print(unique_selfies_tokens)
        start_code = 0xE000  # starting code point in Private Use Area
        self.selfies_to_byte = {}
        self.byte_to_selfies = {}

        for i, token in enumerate(unique_selfies_tokens):
            char = chr(start_code + i)
            self.selfies_to_byte[token] = char
            self.byte_to_selfies[char] = token
        
        print(self.selfies_to_byte)
            
        print(f"Converting SELFIES tokens from {len(selfies_tokens)} SELFIES strings to single byte")
        byte_tokens = ["".join([self.selfies_to_byte[t] for t in s]) for s in selfies_tokens]

        print(f"Training tokenizer on {len(selfies_tokens)} compressed SELFIES strings.")
        tokenizer.train_from_iterator(byte_tokens, vocab_size)

        super().__init__(tokenizer, **kwargs)

    def encode(self, text: str, add_special_tokens: bool = True) -> Tokenizer:
        """Encodes a SMILES string into a list of byte SELFIES token IDs."""
        selfies_string = sf.encoder(text, strict=False)
        selfies_tokens = list(sf.split_selfies(selfies_string))
        byte_tokens = [self.selfies_to_byte[t] for t in selfies_tokens]
        return super().encode(
            byte_tokens, is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of byte SELFIES token IDs back into a SMILES string."""
        byte_string = super().decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        selfies_string = _decode_byte_str_to_selfies(byte_string)
        return sf.decoder(selfies_string)

    def encode_batch(
        self, texts: T.List[str], add_special_tokens: bool = True
    ) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of byte SELFIES token IDs."""
        byte_tokens = [
            [
                self.selfies_to_byte[c] for c in 
                list(sf.split_selfies(sf.encoder(text, strict=False)))
            ] for text in texts
        ]
        return super().encode_batch(
            byte_tokens, is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode_batch(
        self, token_ids_batch: T.List[T.List[int]], skip_special_tokens: bool = True
    ) -> T.List[str]:
        """Decodes a batch of SELFIES token IDs back into SMILES strings."""
        byte_strings = super().decode_batch(
            token_ids_batch, skip_special_tokens=skip_special_tokens
        )
        return [
            sf.decoder(
                self._decode_byte_str_to_selfies(
                    byte_string
                )
            )
            for byte_string in byte_strings
        ]

    def _decode_byte_str_to_selfies(self, text: str) -> str:
        """Converts a byte string back to a SELFIES string."""
        return "".join([self.byte_to_selfies[b] for b in list(text)])