import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T
from torch_geometric.nn import MLP
from mol_tokenizers import LayeredInchIBPETokenizer
from massspecgym.data.transforms import MolToFormulaVector
from massspecgym.models.base import Stage
from transformer_base import DeNovoMassSpecGymModel
from massspecgym.definitions import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
import pytorch_lightning as pl

import pickle
from decimal import Decimal

class LayeredInchiTransformer(DeNovoMassSpecGymModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        layered_inchi_tokenizer: LayeredInchIBPETokenizer,
        start_token: str = SOS_TOKEN,
        end_token: str = EOS_TOKEN,
        pad_token: str = PAD_TOKEN,
        dropout: float = 0.1,
        k_predictions: int = 1,
        temperature: T.Optional[float] = 1.0,
        pre_norm: bool = False,
        chemical_formula: bool = False,
        sampler: str = "naive",
        mz_scaling: bool = False,
        embedding_norm: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layered_inchi_tokenizer = layered_inchi_tokenizer
        self.vocab_sizes = layered_inchi_tokenizer.get_vocab_sizes()
        for token in [start_token, end_token, pad_token]:
            for tokenizer in layered_inchi_tokenizer.tokenizers:
                assert token in tokenizer.get_vocab(), f"Token {token} not found in tokenizer vocabulary."
        self.start_token_ids = [t.token_to_id(start_token) for t in layered_inchi_tokenizer.tokenizers]
        self.end_token_ids = [t.token_to_id(end_token) for t in layered_inchi_tokenizer.tokenizers]
        self.pad_token_ids = [t.token_to_id(pad_token) for t in layered_inchi_tokenizer.tokenizers]

        self.d_model = d_model
        self.max_lengths = layered_inchi_tokenizer.get_max_lengths()
        self.k_predictions = k_predictions
        self.temperature = temperature

        samplers = ["greedy", "naive-parallel"]
        assert sampler in samplers, f"Unknown sampler {sampler}, known samplers: {samplers}"

        self.input_dim = input_dim
        self.mz_scaling = mz_scaling
        self.embedding_norm = embedding_norm

        self.src_encoder = nn.Linear(input_dim, d_model)
        self.tgt_embeddings = nn.ModuleList([nn.Embedding(vocab_size, d_model) for vocab_size in self.vocab_sizes])
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            norm_first=pre_norm
        )

        self.tgt_decoders = nn.ModuleList([nn.Linear(d_model, vocab_size) for vocab_size in self.vocab_sizes])

        self.chemical_formula = chemical_formula
        if self.chemical_formula:
            self.formula_mlp = MLP(
                in_channels=MolToFormulaVector.num_elements(),
                hidden_channels=MolToFormulaVector.num_elements(),
                out_channels=d_model,
                num_layers=1,
                dropout=dropout,
                norm=None
            )

        self.criterion = nn.CrossEntropyLoss()

    def scale_spec_batch(self, batch):
        spec = batch["spec"]
        if self.mz_scaling:
            spec[:,:,0] = spec[:,:,0] / 1000 # scale m/z down
        if self.input_dim == 1:
            return spec[:,:,[0]]
        else:
            return spec
        

    def forward(self, batch):
        spec = self.scale_spec_batch(batch)  # (batch_size, seq_len, in_dim)
        
        smiles = batch["mol"]  # List of SMILES of length batch_size

        encoded_layered_inchis = self.layered_inchi_tokenizer.encode_batch(smiles)
        encoded_layered_inchis = [[l.ids for l in inchi] for inchi in encoded_layered_inchis]
        encoded_layered_inchis = torch.tensor(encoded_layered_inchis, device=spec.device)  # (batch_size, inchi layers, seq_len)

        # Generating padding masks for variable-length sequences
        src_key_padding_mask = self.generate_src_padding_mask(spec)
        tgt_key_padding_masks = [self.generate_tgt_padding_mask(encoded_layered_inchis[:,layer,:], layer) for layer in range(self.layered_inchi_tokenizer.num_layers)]
        # Create target mask (causal mask)
        tgt_seq_len = encoded_layered_inchis.size(2)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(spec.device)
        # Prepare inputs for transformer teacher forcing
        src = spec.permute(1, 0, 2)  # (seq_len, batch_size, in_dim)
        encoded_layered_inchis = encoded_layered_inchis.permute(2, 1, 0)  # (seq_len, batch_size)
        tgts = [encoded_layered_inchis[:-1, layer,:] for layer in range(self.layered_inchi_tokenizer.num_layers)]
        tgt_mask = tgt_mask[:-1, :-1]
        src_key_padding_mask = src_key_padding_mask
        tgt_key_padding_masks = [m[:, :-1] for m in tgt_key_padding_masks]
        # Input and output embeddings
        src_encoded = self.src_encoder(src)  # (seq_len, batch_size, d_model)
        if self.embedding_norm:
            src_encoded = src_encoded / src_encoded.norm(dim=-1, keepdim=True)  # Normalize to unit norm

        if self.chemical_formula:
            formula_emb = self.formula_mlp(batch["formula"])  # (batch_size, d_model)
            src = src + formula_emb.unsqueeze(0)  # (seq_len, batch_size, d_model) + (1, batch_size, d_model)
        src_encoded_scaled = src_encoded # * (self.d_model**0.5)
        tgts = [embedding(tgt) for embedding, tgt in zip(self.tgt_embeddings, tgts)]
        # Transformer forward pass
        memory = self.transformer.encoder(src_encoded_scaled, src_key_padding_mask=src_key_padding_mask)
        outputs = [
            self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask) 
                for tgt, tgt_key_padding_mask in zip(tgts, tgt_key_padding_masks)
        ]

        # Logits to vocabulary and reshape for returning
        
        preds = [tgt_decoder(output).view(-1, vocab_size) for tgt_decoder, output, vocab_size in zip(self.tgt_decoders, outputs, self.vocab_sizes)]
        encoded_layered_inchis = [encoded_layered_inchis[1:,l,:].contiguous().view(-1) for l in range(self.layered_inchi_tokenizer.num_layers)]

        return preds, encoded_layered_inchis
        

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        if stage == Stage('train'):
            self.train()
        else:
            self.eval()

        # Forward pass
        layers_pred, encoded_layered_inchis = self.forward(batch)
        # Compute loss (sum of decoder losses)
        loss = sum(self.criterion(layer_pred, encoded_inchi_layer) for layer_pred, encoded_inchi_layer in zip(layers_pred, encoded_layered_inchis))

        # Generate SMILES strings
        if stage in self.log_only_loss_at_stages:
            mols_pred = None
        else:
            # Alway put model in eval mode for sampling
            self.eval()
            if self.sampler == "naive-parallel":
                mols_pred = self.decode_smiles_parallel(batch)
            elif self.sampler == "greedy":
                mols_pred = self.decode_smiles_beam_search(batch)
            else:
                raise "unkown decoder"
        return dict(loss=loss, mols_pred=mols_pred)

    def generate_src_padding_mask(self, spec):
        return spec.sum(-1) == 0

    def generate_tgt_padding_mask(self, smiles, layer):
        return smiles == self.pad_token_ids[layer]

    def sanitize_decimal(self, x, p):
        a, b = (f"%.{p-1}E" % Decimal(x)).split("E")
        return str(int(Decimal(a) * (10 ** (p-1)))) + "e" + str(int(b)-(p-1))

    def decode_smiles_parallel(self, batch):
        decoded_smiles = self.naive_decode_parallel(
                                batch,
                                nr_preds=self.k_predictions,
                                temperature=self.temperature
            )            
        return [self.smiles_tokenizer.decode_batch(b) for b in decoded_smiles.tolist()]

    def _encode_spec(self, spec, nr_preds):
        # repeat input "beam_width" times for predictions in one decode
        repeated_spec = spec.repeat_interleave(nr_preds, dim=0) # (batch_size * beam_width, seq_len, in_dim)

        #### Calculate Memory (code snippet from greedy_decode function)
        src_key_padding_mask = self.generate_src_padding_mask(repeated_spec)

        repeated_spec = repeated_spec.permute(1, 0, 2)  # (seq_len, batch_size * beam_width, in_dim)
        src = self.src_encoder(repeated_spec)  # (seq_len, batch_size * beam_width, d_model)
        if self.embedding_norm:
            src = src / src.norm(dim=-1, keepdim=True)  # Normalize to unit norm
        #src = src * (self.d_model**0.5)
        memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return memory

    def _decode_step(self, memory, preds, batch_size, nr_preds, i, d):
        last_tokens = preds[:,:,:i+1].reshape(batch_size * nr_preds, i+1).T # (batch_size, nr_preds, seq_length) to (seq_length, batch_size * nr_preds)
        tgt = self.tgt_embeddings[d](last_tokens)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(self.device)
        out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        out = self.tgt_decoders[d](out[-1, :]) # (batch_size * nr_preds, vocab_size)
        return out


    def naive_decode_parallel(self, batch, nr_preds, temperature):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim)
            batch_size = spec.shape[0]
            memory = self._encode_spec(spec, nr_preds)

            preds_layers = []

            for d in range(self.layered_inchi_tokenizer.num_layers):
                # tensor that holds predicted tokens for each prediction
                preds = torch.full((batch_size, nr_preds, self.max_lengths[d]), self.pad_token_ids[d], device=self.device)
                preds[:,:,0] = self.start_token_ids[d]

                stopped = torch.full((batch_size, nr_preds), False, device=self.device)

                for i in range(self.max_lengths[d] - 1):
                    logits = self._decode_step(memory, preds, batch_size, nr_preds, i, d)

                    ### Naive sampling
                    # Scale logits wrt temperature
                    scaled_logits = logits / temperature if temperature != None else logits
                    probs = F.softmax(scaled_logits, dim=-1)

                    # Sample from the adjusted distribution
                    next_tokens = torch.multinomial(probs, num_samples=1).reshape(batch_size, nr_preds) # (batch_size, nr_preds)
                    
                    # store values
                    preds[:,:,i+1] = torch.where(stopped, self.pad_token_ids[d], next_tokens)
                    stopped[next_tokens == self.end_token_ids[d]] = True

                    if torch.all(stopped):
                        break
                preds_layers.append(preds)

            return preds_layers

    def greedy_decode_parallel(self, batch, nr_preds):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim)
            batch_size = spec.shape[0]
            memory = self._encode_spec(spec, nr_preds)

            preds_layers = []

            for d in range(self.layered_inchi_tokenizer.num_layers):
                # tensor that holds predicted tokens for each prediction
                preds = torch.full((batch_size, nr_preds, self.max_lengths[d]), self.pad_token_ids[d], device=self.device)
                preds[:,:,0] = self.start_token_ids[d]

                stopped = torch.full((batch_size, nr_preds), False, device=self.device)

                for i in range(self.max_lengths[d] - 1):
                    logits = self._decode_step(memory, preds, batch_size, nr_preds, i, d)

                    ### Greedy sampling
                    # Get token with highest logit
                    next_tokens = torch.argmax(logits, dim=-1).reshape((batch_size, nr_preds)) # (batch_size, nr_preds)
                    # store values
                    preds[:,:,i+1] = torch.where(stopped, self.pad_token_ids[d], next_tokens)
                    stopped[next_tokens == self.end_token_ids[d]] = True

                    if torch.all(stopped):
                        break
                preds_layers.append(preds)

            return preds_layers

