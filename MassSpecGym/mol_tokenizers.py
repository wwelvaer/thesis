import pandas as pd
import typing as T
import selfies as sf
from tokenizers import Tokenizer, processors, models, pre_tokenizers
from tokenizers.implementations import BaseTokenizer, ByteLevelBPETokenizer, CharBPETokenizer
import utils as utils
from massspecgym.definitions import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
import string
import deepsmiles as ds
from concurrent.futures import ThreadPoolExecutor
import rdkit.Chem as Chem
import re


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
    def __init__(self, dataset_size: int=4, smiles_pth: T.Optional[str] = None, cache_dir=None, vocab_size=30000, min_frequency=2, filterStereochemistry=False, **kwargs):
        """
        Initialize the BPE tokenizer for SMILES strings, with optional training data.
        """
        self.filterStereochemistry = filterStereochemistry
        tokenizer = ByteLevelBPETokenizer()
        if smiles_pth:
            tokenizer.train(smiles_pth)
        else:
            smiles = utils.load_train_mols().tolist()
            if dataset_size > 0:
                smiles += utils.load_unlabeled_mols(col_name="smiles", size=dataset_size, cache_dir=cache_dir).tolist()

            if self.filterStereochemistry:
                print(f"Removing stereochemistry from {len(smiles)} SMILES strings.")
                smiles = [Chem.rdmolfiles.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in smiles]
                
            print(f"Training tokenizer on {len(smiles)} SMILES strings.")
            tokenizer.train_from_iterator(smiles, vocab_size, min_frequency=min_frequency)

        super().__init__(tokenizer, **kwargs)

    def encode(self, smiles: str) -> Tokenizer:
        """Encodes a SMILES string into a list of token IDs."""
        if self.filterStereochemistry:
            smiles = Chem.rdmolfiles.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
        return super().encode(smiles)

    def encode_batch(self, smiles_strings: T.List[str]) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of token IDs."""
        if self.filterStereochemistry:
            smiles_strings = [Chem.rdmolfiles.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in smiles_strings]
        return super().encode_batch(smiles_strings)

class DeepSmilesTokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, dataset_size: int=4, cache_dir=None, **kwargs):
        self.converter = ds.Converter(rings=True, branches=True)

        smiles = utils.load_train_mols().tolist()
        if dataset_size > 0:
                smiles += utils.load_unlabeled_mols(col_name="smiles", size=dataset_size, cache_dir=cache_dir).tolist()
        
        print(f"Converting {len(smiles)} SMILES to DeepSMILES strings.")
        deepsmiles = [self.converter.encode(s) for s in smiles]

        vocab = {t: i for i,t in enumerate(set(y for x in deepsmiles for y in list(x)))}

        vocab[UNK_TOKEN] = len(vocab)
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    
        super().__init__(tokenizer, **kwargs)

    def encode(self, smiles: str, add_special_tokens: bool = True) -> Tokenizer:
        """Encodes a SMILES string into a list of DeepSMILES token IDs."""
        deepsmiles_str = self.converter.encode(smiles)
        return super().encode(
            list(deepsmiles_str), is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of DeepSMILES token IDs back into a SMILES string."""
        deepsmiles_string = super().decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return self.try_deepsmiles_to_smiles(self._decode_wordlevel_str_to_deepsmiles(deepsmiles_string))

    def encode_batch(
        self, smiles: T.List[str], add_special_tokens: bool = True
    ) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of DeepSMILES token IDs."""
        deepsmiles_strings = [
            list(self.converter.encode(s)) for s in smiles
        ]
        return super().encode_batch(
            deepsmiles_strings, is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode_batch(
        self, token_ids_batch: T.List[T.List[int]], skip_special_tokens: bool = True
    ) -> T.List[str]:
        """Decodes a batch of DeepSMILES token IDs back into SMILES strings."""
        deepsmiles_strings = super().decode_batch(
            token_ids_batch, skip_special_tokens=skip_special_tokens
        )
        return [
            self.try_deepsmiles_to_smiles(self._decode_wordlevel_str_to_deepsmiles(deepsmiles_string))
            for deepsmiles_string in deepsmiles_strings
        ]

    def _decode_wordlevel_str_to_deepsmiles(self, text: str) -> str:
        """Converts a WordLevel string back to a SMILES string."""
        return text.replace(" ", "")

    def try_deepsmiles_to_smiles(self, deepsmiles: str) -> str:
        try:
            smiles = self.converter.decode(deepsmiles)
        except:
            smiles = None
        return smiles

class SmilesTokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, dataset_size: int=4, cache_dir=None, filterStereochemistry=False, **kwargs):
        self.filterStereochemistry = filterStereochemistry
        
        smiles = utils.load_train_mols().tolist()
        if dataset_size > 0:
            smiles += utils.load_unlabeled_mols(col_name="smiles", size=dataset_size, cache_dir=cache_dir).tolist()

        if self.filterStereochemistry:
            print(f"Removing stereochemistry from {len(smiles)} SMILES strings.")
            smiles = [Chem.rdmolfiles.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in smiles]

        vocab = {t: i for i,t in enumerate(set(y for x in smiles for y in list(x)))}

        vocab[UNK_TOKEN] = len(vocab)
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    
        super().__init__(tokenizer, **kwargs)

    def encode(self, smiles: str, add_special_tokens: bool = True) -> Tokenizer:
        """Encodes a SMILES string into a list of SMILES token IDs."""
        if self.filterStereochemistry:
            smiles = self._filter_stereo(smiles)
        return super().encode(
            list(smiles), is_pretokenized=True, add_special_tokens=add_special_tokens
        )

    def decode(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of SMILES token IDs back into a SMILES string."""
        smiles_string = super().decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return self._decode_wordlevel_str_to_smiles(smiles_string)

    def encode_batch(
        self, smiles_strings: T.List[str], add_special_tokens: bool = True
    ) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of SMILES token IDs."""
        tokens = [
            list(self._filter_stereo(s) if self.filterStereochemistry else s) 
                for s in smiles_strings
        ]
        return super().encode_batch(
            tokens, is_pretokenized=True, add_special_tokens=add_special_tokens
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
    
    def _filter_stereo(self, smiles:str) -> str:
        return Chem.rdmolfiles.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)

class SelfiesBPETokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, dataset_size: int=4, encoded_selfies=None, cache_dir=None, vocab_size=30000, min_frequency=2, **kwargs):
        """
        Initialize the BPE tokenizer for SELFIES strings
        """
        tokenizer = ByteLevelBPETokenizer()

        smiles = utils.load_train_mols().tolist()

        print(f"Converting {len(smiles)} SMILES strings to SELFIES strings")
        selfies_strings = [sf.encoder(s, strict=False) for s in smiles]

        print(f"Calculating SELFIES token to byte mapping")
        unique_selfies_tokens = list(sorted(sf.get_alphabet_from_selfies(selfies_strings)))

        printable_chars = string.printable.strip()  # Removes whitespace characters
        printable_chars = [c for c in printable_chars if not (c in ['"', "'", "\\", '`'])] # Filter unstable characters

        # Ensure there are enough characters to map each word uniquely
        if len(unique_selfies_tokens) > len(printable_chars):
            raise ValueError(f"Not enough unique characters to map each word. Tyring to use {len(unique_selfies_tokens)} while only {len(printable_chars)} are available")

        self.selfies_to_byte = {}
        self.byte_to_selfies = {}
        self.vocab = []

        for i, token in enumerate(unique_selfies_tokens):
            char = printable_chars[i]
            self.selfies_to_byte[token] = char
            self.byte_to_selfies[char] = token
            self.vocab.append(char)

        if encoded_selfies is None:
            unlabaled_smiles = utils.load_unlabeled_mols(col_name="smiles", size=dataset_size, cache_dir=cache_dir).tolist()
            print(f"Converting {len(unlabaled_smiles)} SMILES strings to SELFIES strings")
            selfies_strings += [sf.encoder(s, strict=False) for s in unlabaled_smiles]
        
            print(f"Converting SELFIES tokens from {len(selfies_strings)} SELFIES strings to single byte")
            with open("encoded_selfies.txt", "w") as file:
                for s in selfies_strings:
                    if all([t in self.selfies_to_byte for t in sf.split_selfies(s)]):
                        enc = self._encode_selfies_to_byte_str(s)
                        file.write(enc + "\n")
            print("Encoded selfies written to encoded_selfies.txt")

        print(f"Training tokenizer")
        tokenizer.train("encoded_selfies.txt" if encoded_selfies is None else encoded_selfies, vocab_size, min_frequency=min_frequency, special_tokens=[])

        super().__init__(tokenizer, **kwargs)

    def encode(self, text: str, add_special_tokens: bool = True) -> Tokenizer:
        """Encodes a SMILES string into a list of byte SELFIES token IDs."""
        selfies_string = sf.encoder(text, strict=False)
        byte_str = self._encode_selfies_to_byte_str(selfies_string)
        return super().encode(
            byte_str, add_special_tokens=add_special_tokens
        )

    def decode(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of byte SELFIES token IDs back into a SMILES string."""
        byte_string = super().decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        selfies_string = self._decode_byte_str_to_selfies(byte_string)
        return sf.decoder(selfies_string)

    def encode_batch(
        self, texts: T.List[str], add_special_tokens: bool = True
    ) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of byte SELFIES token IDs."""
        byte_tokens = [
            [
                self.selfies_to_byte[c] if c in self.selfies_to_byte else self.unk_token for c in 
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
        return "".join([self.byte_to_selfies[b] if b in self.byte_to_selfies else "" for b in list(text)])

    def _encode_selfies_to_byte_str(self, selfies_str: str) -> str:
        """Converts a SELFIES string back to a byte string."""
        return "".join([self.selfies_to_byte[t] if t in self.selfies_to_byte else self.unk_token for t in sf.split_selfies(selfies_str)])

class DeepSmilesBPETokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, dataset_size: int=4, deepsmiles_pth: T.Optional[str] = None, cache_dir=None, vocab_size=30000, **kwargs):
        """
        Initialize the BPE tokenizer for SMILES strings, with optional training data.
        """
        self.converter = ds.Converter(rings=True, branches=True)

        tokenizer = ByteLevelBPETokenizer()
        if deepsmiles_pth:
            tokenizer.train(deepsmiles_pth)
        else:
            smiles = utils.load_train_mols().tolist()

            if dataset_size > 0:
                smiles += utils.load_unlabeled_mols(col_name="smiles", size=dataset_size, cache_dir=cache_dir).tolist()

            print(f"Converting {len(smiles)} SMILES to DeepSMILES strings.")
            deepsmiles = [self.converter.encode(s) for s in smiles]

            print(f"Training tokenizer on {len(deepsmiles)} DeepSMILES strings.")
            tokenizer.train_from_iterator(deepsmiles, vocab_size)

        super().__init__(tokenizer, **kwargs)

    def encode(self, smiles: str) -> Tokenizer:
        """Encodes a SMILES string into a list of DeepSMILES token IDs."""
        deepsmiles_str = self.converter.encode(smiles)
        return super().encode(deepsmiles_str)

    def decode(self, token_ids: T.List[int]) -> str:
        """Decodes a list of DeepSMILES token IDs back into a SMILES string."""
        deepsmiles_str = super().decode(token_ids)
        return self.try_deepsmiles_to_smiles(deepsmiles_str)

    def encode_batch(self, smiles: T.List[str]) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of DeepSMILES token IDs."""
        deepsmiles_strings = [self.converter.encode(s) for s in smiles]
        return super().encode_batch(deepsmiles_strings)

    def decode_batch(
        self, token_ids_batch: T.List[T.List[int]], skip_special_tokens: bool = True
    ) -> T.List[str]:
        """Decodes a batch of DeepSMILES token IDs back into SMILES strings."""
        deepsmiles_strings = super().decode_batch(token_ids_batch)
        return [self.try_deepsmiles_to_smiles(s) for s in deepsmiles_strings]

    def try_deepsmiles_to_smiles(self, deepsmiles: str) -> str:
        try:
            smiles = self.converter.decode(deepsmiles)
        except:
            smiles = None
        return smiles

class InchIBPETokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, dataset_size: int=4, inchi_pth: T.Optional[str] = None, cache_dir=None, vocab_size=30000, **kwargs):
        """
        Initialize the BPE tokenizer for InchI strings, with optional training data.
        """

        tokenizer = ByteLevelBPETokenizer()
        if inchi_pth:
            tokenizer.train(inchi_pth)
        else:
            smiles = utils.load_train_mols().tolist()

            if dataset_size > 0:
                smiles += utils.load_unlabeled_mols(col_name="smiles", size=dataset_size, cache_dir=cache_dir).tolist()

            print(f"Converting {len(smiles)} SMILES to InchI strings.")
            self.unk_token = UNK_TOKEN
            inchis = []

            for i, s in enumerate(smiles):
                inchis.append(self.smiles_to_inchi(s))
                if i % (len(smiles) // 1000) == 0:
                    print("Conversion:\t{}% Complete".format(round(i / len(smiles) * 100, 2)), end = "\r", flush = True)

            print(f"Training tokenizer on {len(inchis)} InchI strings.")
            tokenizer.train_from_iterator(inchis, vocab_size)

        super().__init__(tokenizer, **kwargs)

    def encode(self, smiles: str) -> Tokenizer:
        """Encodes a SMILES string into a list of InchI token IDs."""
        return super().encode(self.smiles_to_inchi(smiles))

    def decode(self, token_ids: T.List[int]) -> str:
        """Decodes a list of InchI token IDs back into a SMILES string."""
        return self.inchi_to_smiles(super().decode(token_ids))

    def encode_batch(self, smiles: T.List[str]) -> T.List[Tokenizer]:
        """Encodes a batch of SMILES strings into a list of InchI token IDs."""
        return super().encode_batch([self.smiles_to_inchi(s) for s in smiles])

    def decode_batch(self, token_ids_batch: T.List[T.List[int]]) -> T.List[str]:
        """Decodes a batch of InchI token IDs back into SMILES strings."""
        return [self.inchi_to_smiles(s) for s in super().decode_batch(token_ids_batch)]

    def smiles_to_inchi(self, smiles: str) -> str:
        try:
            inchi = Chem.inchi.MolToInchi(Chem.MolFromSmiles(smiles))
        except:
            inchi = self.unk_token
        return inchi
    
    def inchi_to_smiles(self, inchi: str) -> str:
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromInchi(inchi))
        except:
            smiles = None
        return smiles

class DummyBPETokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, iterator, vocab_size=30000, min_frequency=2, **kwargs):
        """
        Initialize the BPE tokenizer for strings
        """
        tokenizer = ByteLevelBPETokenizer()
        print(f"Training tokenizer on {len(iterator)} strings.")
        tokenizer.train_from_iterator(iterator, vocab_size, min_frequency=min_frequency)
        super().__init__(tokenizer, **kwargs)

class LayeredInchIBPETokenizer():
    def __init__(self, dataset_size: int=4, inchi_pth: T.Optional[str] = None, cache_dir=None, vocab_size=30000, min_frequency=2, **kwargs):
        """
        Initialize the BPE tokenizer for InchI strings, with optional training data.
        """
        self.num_layers = 3
        self.inchi_pattern = re.compile(r"^InChI=1S/([^/]*)/c([^/]*)/h([^/]*)/?.*$")

        if inchi_pth:
            assert False, "Not implemented"

        smiles = utils.load_train_mols().tolist()

        if dataset_size > 0:
            smiles += utils.load_unlabeled_mols(col_name="smiles", size=dataset_size, cache_dir=cache_dir).tolist()

        print(f"Converting {len(smiles)} SMILES to InchI strings.")
        self.unk_token = UNK_TOKEN
        inchis_f = []
        inchis_c = []
        inchis_h = []

        for i, s in enumerate(smiles):
            inchi = self.smiles_to_inchi(s)
            match = self.inchi_pattern.match(inchi) if inchi else None
            if match:
                f, c, h = match.groups()
                inchis_f.append(f)
                inchis_c.append(c)
                inchis_h.append(h)

            if i % (len(smiles) // 1000) == 0:
                print("Conversion:\t{}% Complete".format(round(i / len(smiles) * 100, 2)), end = "\r", flush = True)
        print("Conversion SMILES to InchI strings Complete")

        
        print(f"Training tokenizer f")
        self.tokenizer_f = DummyBPETokenizer(inchis_f, vocab_size=vocab_size, min_frequency=min_frequency, **kwargs)
        print(f"Training tokenizer c")
        self.tokenizer_c = DummyBPETokenizer(inchis_c, vocab_size=vocab_size, min_frequency=min_frequency, **kwargs)
        print(f"Training tokenizer h")
        self.tokenizer_h = DummyBPETokenizer(inchis_h, vocab_size=vocab_size, min_frequency=min_frequency, **kwargs)

        self.tokenizers = [self.tokenizer_f, self.tokenizer_c, self.tokenizer_h]

        self.max_length = self.tokenizer_f.max_length

    def get_vocab_sizes(self):
        return [t.get_vocab_size() for t in self.tokenizers]

    def get_max_lengths(self):
        return [t.max_length for t in self.tokenizers]

    def get_inchi(self, f, c, h):
        return f"InChI=1S/{f}/c{c}/h{h}"

    def encode(self, smiles: str) -> T.List[Tokenizer]:
        """Encodes a SMILES string into a list of InchI layers each containing token IDs."""
        inchi = self.smiles_to_inchi(smiles)
        m = self.inchi_pattern.match(inchi) if inchi else None
        if m is None:
            return None
        layers = m.groups()
        assert len(layers) == len(self.tokenizers), f"invalid number of layers: {len(layers)}, expected {len(self.tokenizers)}"
        return [t.encode(s) for s, t in zip(layers, self.tokenizers)]

    def decode(self, token_ids_layers: T.List[T.List[int]]) -> str:
        """Decodes a list of InchI layers each containing token IDs back into a SMILES string."""
        assert len(token_ids_layers) == len(self.tokenizers), f"invalid number of layers: {len(token_ids_layers)}, expected {len(self.tokenizers)}"
        inchi = self.get_inchi(*[t.decode(ids) for ids, t in zip(token_ids_layers, self.tokenizers)])
        return self.inchi_to_smiles(inchi)

    def encode_batch(self, smiles: T.List[str]) -> T.List[T.List[Tokenizer]]:
        return [self.encode(s) for s in smiles]

    def decode_batch(self, token_ids_layers_batch: T.List[T.List[T.List[int]]]) -> T.List[str]:
        """Decodes tensors for each InchI layer containing token IDs back into a SMILES string."""
        assert len(set([(x.size(0), x.size(1)) for x in token_ids_layers_batch])) == 1, "Layers do not have same batch size or number of predictions"
        return [
            [self.decode(x) for x in zip(bl1.tolist(), bl2.tolist(), bl3.tolist())] 
                for bl1, bl2, bl3 in zip(*token_ids_layers_batch)
        ]
    
    def smiles_to_inchi(self, smiles: str) -> str:
        try:
            inchi = Chem.inchi.MolToInchi(Chem.MolFromSmiles(smiles))
        except:
            inchi = None
        return inchi
    
    def inchi_to_smiles(self, inchi: str) -> str:
        try:
            smiles = Chem.MolToSmiles(Chem.MolFromInchi(inchi))
        except:
            smiles = None
        return smiles
