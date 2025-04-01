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

class LayeredInchIBPETokenizer(DeNovoMassSpecGymModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        inchi_tokenizer: LayeredInchIBPETokenizer,
        start_token: str = SOS_TOKEN,
        end_token: str = EOS_TOKEN,
        pad_token: str = PAD_TOKEN,
        dropout: float = 0.1,
        max_smiles_len: int = 200,
        k_predictions: int = 1,
        temperature: T.Optional[float] = 1.0,
        pre_norm: bool = False,
        chemical_formula: bool = False,
        sampler: str = "naive",
        k: int = 10,
        q: float = 0.8,
        beam_width: int = 20,
        alpha: float = 1.0,
        mz_scaling: bool = False,
        embedding_norm: bool = False,
        store_metadata: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.smiles_tokenizer = smiles_tokenizer
        self.vocab_size = smiles_tokenizer.get_vocab_size()
        for token in [start_token, end_token, pad_token]:
            assert token in smiles_tokenizer.get_vocab(), f"Token {token} not found in tokenizer vocabulary."
        self.start_token_id = smiles_tokenizer.token_to_id(start_token)
        self.end_token_id = smiles_tokenizer.token_to_id(end_token)
        self.pad_token_id = smiles_tokenizer.token_to_id(pad_token)

        self.d_model = d_model
        self.max_smiles_len = max_smiles_len
        self.k_predictions = k_predictions
        self.temperature = temperature
        if self.k_predictions == 1:  # TODO: this logic should be changed because sampling with k = 1 also makes sense
            self.temperature = None

        samplers = ["greedy", "naive-parallel"]
        assert sampler in samplers, f"Unknown sampler {sampler}, known samplers: {samplers}"

        self.store_metadata = store_metadata

        self.input_dim = input_dim
        self.mz_scaling = mz_scaling
        self.embedding_norm = embedding_norm

        self.src_encoder = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(self.vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            norm_first=pre_norm
        )

        # TODO convert to module list with 3 decoders
        self.tgt_decoder = nn.Linear(d_model, self.vocab_size)

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

        smiles = self.smiles_tokenizer.encode_batch(smiles)
        smiles = [s.ids for s in smiles]
        smiles = torch.tensor(smiles, device=spec.device)  # (batch_size, seq_len)

        # Generating padding masks for variable-length sequences
        src_key_padding_mask = self.generate_src_padding_mask(spec)
        tgt_key_padding_mask = self.generate_tgt_padding_mask(smiles)
        # Create target mask (causal mask)
        tgt_seq_len = smiles.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(smiles.device)
        # Preapre inputs for transformer teacher forcing
        src = spec.permute(1, 0, 2)  # (seq_len, batch_size, in_dim)
        smiles = smiles.permute(1, 0)  # (seq_len, batch_size)
        tgt = smiles[:-1, :]
        tgt_mask = tgt_mask[:-1, :-1]
        src_key_padding_mask = src_key_padding_mask
        tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        # Input and output embeddings
        src_encoded = self.src_encoder(src)  # (seq_len, batch_size, d_model)
        if self.embedding_norm:
            src_encoded = src_encoded / src_encoded.norm(dim=-1, keepdim=True)  # Normalize to unit norm

        if self.chemical_formula:
            formula_emb = self.formula_mlp(batch["formula"])  # (batch_size, d_model)
            src = src + formula_emb.unsqueeze(0)  # (seq_len, batch_size, d_model) + (1, batch_size, d_model)
        src_encoded_scaled = src_encoded # * (self.d_model**0.5)
        tgt = self.tgt_embedding(tgt) #* (self.d_model**0.5)  # (seq_len, batch_size, d_model)
        # Transformer forward pass
        memory = self.transformer.encoder(src_encoded_scaled, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # Logits to vocabulary
        output_decoded = self.tgt_decoder(output)  # (seq_len, batch_size, vocab_size)
        # Reshape before returning
        smiles_pred = output_decoded.view(-1, self.vocab_size)

        smiles = smiles[1:, :].contiguous().view(-1)
        return smiles_pred, smiles
        

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        if stage == Stage('train'):
            self.train()
        else:
            self.eval()

        # Forward pass
        smiles_pred, smiles = self.forward(batch)
        # Compute loss
        loss = self.criterion(smiles_pred, smiles)

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

    def generate_tgt_padding_mask(self, smiles):
        return smiles == self.pad_token_id

    def save_metadata(self, name, meta_data):
        # Load metadata
        try:
            with open(f'{name}.pkl', 'rb') as f:
                batch_meta_data = pickle.load(f)
        except:
            batch_meta_data = []

        batch_meta_data.append(meta_data)
        # Save metadata
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(batch_meta_data, f)

    def sanitize_decimal(self, x, p):
        a, b = (f"%.{p-1}E" % Decimal(x)).split("E")
        return str(int(Decimal(a) * (10 ** (p-1)))) + "e" + str(int(b)-(p-1))

    def decode_smiles_parallel(self, batch):
        decoded_smiles, i = self.naive_decode_parallel(
                                batch,
                                nr_preds=self.k_predictions,
                                max_len=self.max_smiles_len,
                                temperature=self.temperature
            )

        if self.store_metadata:
            meta_data = {
                'forward_passes': i,
                'prediction_lengths': [
                    [x.index(self.end_token_id) if self.end_token_id in x else self.max_smiles_len for x in b] for b in decoded_smiles.tolist()
                ],
                'temp': self.temperature,
            }

            self.save_metadata(f'meta_data/Naive_parallel_metadata_temp-{self.sanitize_decimal(self.temperature, 2)}', meta_data)
            
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

    def _decode_step(self, memory, preds, batch_size, nr_preds, i):
        last_tokens = preds[:,:,:i+1].reshape(batch_size * nr_preds, i+1).T # (batch_size, nr_preds, seq_length) to (seq_length, batch_size * nr_preds)
        tgt = self.tgt_embedding(last_tokens)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(self.device)
        out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        out = self.tgt_decoder(out[-1, :]) # (batch_size * nr_preds, vocab_size)
        return out


    def naive_decode_parallel(self, batch, nr_preds, max_len, temperature):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim)
            batch_size = spec.shape[0]
            memory = self._encode_spec(spec, nr_preds)

            # tensor that holds predicted tokens for each prediction
            preds = torch.full((batch_size, nr_preds, max_len), self.pad_token_id, device=self.device)
            preds[:,:,0] = self.start_token_id

            stopped = torch.full((batch_size, nr_preds), False, device=self.device)

            for i in range(max_len - 1):
                logits = self._decode_step(memory, preds, batch_size, nr_preds, i)

                ### Naive sampling
                # Scale logits wrt temperature
                scaled_logits = logits / temperature if temperature != None else logits
                probs = F.softmax(scaled_logits, dim=-1)

                # Sample from the adjusted distribution
                next_tokens = torch.multinomial(probs, num_samples=1).reshape(batch_size, nr_preds) # (batch_size, nr_preds)
                
                # store values
                preds[:,:,i+1] = torch.where(stopped, self.pad_token_id, next_tokens)
                stopped[next_tokens == self.end_token_id] = True

                if torch.all(stopped):
                    return preds, i+1

            return preds, max_len

    def greedy_decode_parallel(self, batch, nr_preds, max_len, temperature):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim)
            batch_size = spec.shape[0]
            memory = self._encode_spec(spec, nr_preds)

            # tensor that holds predicted tokens for each prediction
            preds = torch.full((batch_size, nr_preds, max_len), self.pad_token_id, device=self.device)
            preds[:,:,0] = self.start_token_id

            stopped = torch.full((batch_size, nr_preds), False, device=self.device)

            for i in range(max_len - 1):
                logits = self._decode_step(memory, preds, batch_size, nr_preds, i)

                ### Greedy sampling
                # Get token with highest logit
                next_tokens = torch.argmax(logits, dim=-1) # (batch_size, nr_preds)
                
                # store values
                preds[:,:,i+1] = torch.where(stopped, self.pad_token_id, next_tokens)
                stopped[next_tokens == self.end_token_id] = True

                if torch.all(stopped):
                    return preds, i+1

            return preds, max_len

