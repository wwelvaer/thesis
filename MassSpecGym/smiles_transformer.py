import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T
from torch_geometric.nn import MLP
from massspecgym.models.tokenizers import SpecialTokensBaseTokenizer
from massspecgym.data.transforms import MolToFormulaVector
from massspecgym.models.base import Stage
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel
from massspecgym.definitions import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
import pytorch_lightning as pl

import pickle
from decimal import Decimal

class SmilesTransformer(DeNovoMassSpecGymModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        smiles_tokenizer: SpecialTokensBaseTokenizer,
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
        mz_scaling: bool = False,
        embedding_norm: bool = False,
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

        samplers = ["greedy", "naive", "naive-parallel", "top-k", "top-k-parallel", "top-q",  "top-q-parallel"]
        assert sampler in samplers, f"Unknown sampler {sampler}, known samplers: {samplers}"
        self.sampler = sampler
        self.k = k
        self.q = q
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
        # Forward pass
        smiles_pred, smiles = self.forward(batch)
        # Compute loss
        loss = self.criterion(smiles_pred, smiles)

        # Generate SMILES strings
        if stage in self.log_only_loss_at_stages:
            mols_pred = None
        else:
            if self.sampler == "naive":
                mols_pred = self.decode_smiles(batch)
            elif self.sampler == "naive-parallel":
                mols_pred = self.decode_smiles_parallel(batch)
            elif self.sampler == "top-k":
                mols_pred = self.decode_smiles_top_k(batch)
            elif self.sampler == "top-k-parallel":
                mols_pred = self.decode_smiles_top_k_parallel(batch)
            elif self.sampler == "top-q":
                mols_pred = self.decode_smiles_top_q(batch)
            elif self.sampler == "top-q-parallel":
                mols_pred = self.decode_smiles_top_q_parallel(batch)
            else:
                raise "unkown decoder"
        return dict(loss=loss, mols_pred=mols_pred)

    def generate_src_padding_mask(self, spec):
        return spec.sum(-1) == 0

    def generate_tgt_padding_mask(self, smiles):
        return smiles == self.pad_token_id

    def decode_smiles(self, batch):
        decoded_smiles_str = []

        meta_data = {
            'forward_passes': [],
            'prediction_lengths': [],
            'temp': self.temperature
        }
        
        for _ in range(self.k_predictions):
            decoded_smiles = self.naive_decode(
                batch,
                max_len=self.max_smiles_len,
                temperature=self.temperature,
            )

            decoded_smiles = [seq.tolist() for seq in decoded_smiles]
            
            # Store prediction length and number of forward passes
            meta_data['forward_passes'].append(len(decoded_smiles[0]) - 1)
            meta_data['prediction_lengths'].append([x.index(self.end_token_id) if self.end_token_id in x else -1 for x in decoded_smiles])            

            decoded_smiles_str.append(self.smiles_tokenizer.decode_batch(decoded_smiles))

        # Transpose from (k, batch_size) to (batch_size, k)
        decoded_smiles_str = list(map(list, zip(*decoded_smiles_str)))

        # Load metadata
        try:
            with open(f'NAIVE_metadata_temp-{self.sanitize_decimal(self.temperature, 2)}.pkl', 'rb') as f:
                batch_meta_data = pickle.load(f)
        except:
            batch_meta_data = []

        batch_meta_data.append(meta_data)
        # Save metadata
        with open(f'NAIVE_metadata_temp-{self.sanitize_decimal(self.temperature, 2)}.pkl', 'wb') as f:
            pickle.dump(batch_meta_data, f)

        return decoded_smiles_str

    def sanitize_decimal(self, x, p):
        a, b = (f"%.{p-1}E" % Decimal(x)).split("E")
        return str(int(Decimal(a) * (10 ** (p-1)))) + "e" + str(int(b)-(p-1))

    def decode_smiles_parallel(self, batch):
        decoded_smiles = self.naive_decode_parallel(
                                batch,
                                nr_preds=self.k_predictions,
                                max_len=self.max_smiles_len,
                                temperature=self.temperature
            )
        return [self.smiles_tokenizer.decode_batch(b) for b in decoded_smiles.tolist()]

    def naive_decode(self, batch, max_len, temperature):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim) 
            src_key_padding_mask = self.generate_src_padding_mask(spec)   

            spec = spec.permute(1, 0, 2)  # (seq_len, batch_size, in_dim)
            src = self.src_encoder(spec)  # (seq_len, batch_size, d_model)
            if self.embedding_norm:
                src = src / src.norm(dim=-1, keepdim=True)  # Normalize to unit norm
            if self.chemical_formula:
                formula_emb = self.formula_mlp(batch["formula"])  # (batch_size, d_model)
                src = src + formula_emb.unsqueeze(0)  # (seq_len, batch_size, d_model) + (1, batch_size, d_model)
            #src = src * (self.d_model**0.5)
            memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
            batch_size = src.size(1)
            out_tokens = torch.ones(1, batch_size).fill_(self.start_token_id).type(torch.long).to(spec.device)
            for _ in range(max_len - 1):
                tgt = self.tgt_embedding(out_tokens)# * (self.d_model**0.5)
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
                out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
                out = self.tgt_decoder(out[-1, :])  # (batch_size, vocab_size)

                # Select next token
                if self.temperature is None:
                    probs = F.softmax(out, dim=-1)
                    next_token = torch.argmax(probs, dim=-1)  # (batch_size,)
                else:
                    probs = F.softmax(out / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # (batch_size,)

                next_token = next_token.unsqueeze(0)  # (1, batch_size)
                out_tokens = torch.cat([out_tokens, next_token], dim=0)
                
                if torch.all(torch.logical_or(next_token == self.end_token_id, next_token == self.pad_token_id)):
                    break

            out_tokens = out_tokens.permute(1, 0)  # (batch_size, seq_len)
            return out_tokens

    def naive_decode_parallel(self, batch, nr_preds, max_len, temperature):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim)
            batch_size = spec.shape[0]
            # repeat input "nr_preds" times for predictions in one decode
            spec = spec.repeat_interleave(nr_preds, dim=0) # (batch_size * nr_preds, seq_len, in_dim)

            #### Calculate Memory (code snippet from greedy_decode function)
            src_key_padding_mask = self.generate_src_padding_mask(spec)
            
            spec = spec.permute(1, 0, 2)  # (seq_len, batch_size * nr_preds, in_dim)
            src = self.src_encoder(spec)  # (seq_len, batch_size * nr_preds, d_model)
            if self.embedding_norm:
                src = src / src.norm(dim=-1, keepdim=True)  # Normalize to unit norm
            #src = src * (self.d_model**0.5)
            memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
            ####

            # tensor that holds predicted tokens for each prediction
            preds = torch.full((batch_size, nr_preds, max_len), self.start_token_id, device=self.device)

            for i in range(max_len - 1):
                last_tokens = preds[:,:,:i+1].reshape(batch_size * nr_preds, i+1).T # (batch_size, nr_preds, seq_length) to (seq_length, batch_size * nr_preds)
                tgt = self.tgt_embedding(last_tokens)# * (self.d_model**0.5) # embedding scaling
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
                out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
                out = self.tgt_decoder(out[-1, :]) # (batch_size * nr_preds, vocab_size)

                ### Naive sampling
                # Scale logits wrt temperature
                scaled_logits = out / temperature if temperature != None else out
                
                # Compute probabilities
                probabilities = F.softmax(scaled_logits, dim=-1)

                # Sample from the adjusted distribution
                next_tokens = torch.multinomial(probabilities, num_samples=1) # (batch_size * nr_preds)
                preds[:,:,i+1] = next_tokens.reshape(batch_size, nr_preds)

                # End loop when all sequences have end token
                if torch.all(next_tokens == self.end_token_id):
                    break

            return preds
    
    def decode_smiles_top_k(self, batch):
        decoded_smiles_str = []
        for _ in range(self.k_predictions):
            decoded_smiles = self.top_k_decode_one(
                                batch,
                                max_len=self.max_smiles_len,
                                temperature=self.temperature,
                                k=self.k
            )
            decoded_smiles = [seq.tolist() for seq in decoded_smiles]
            decoded_smiles_str.append(self.smiles_tokenizer.decode_batch(decoded_smiles))

        # Transpose from (nr_preds, batch_size) to (batch_size, nr_preds)
        decoded_smiles_str = list(map(list, zip(*decoded_smiles_str)))

        return decoded_smiles_str

    def decode_smiles_top_k_parallel(self, batch):
        decoded_smiles = self.top_k_decode(
                                batch,
                                nr_preds=self.k_predictions,
                                max_len=self.max_smiles_len,
                                temperature=self.temperature,
                                k=self.k
            )
        return [self.smiles_tokenizer.decode_batch(b) for b in decoded_smiles.tolist()]
    
    def decode_smiles_top_q(self, batch):
        decoded_smiles_str = []
        for _ in range(self.k_predictions):
            decoded_smiles = self.top_q_decode_one(
                                batch,
                                max_len=self.max_smiles_len,
                                temperature=self.temperature,
                                q=self.q
            )
            decoded_smiles = [seq.tolist() for seq in decoded_smiles]
            decoded_smiles_str.append(self.smiles_tokenizer.decode_batch(decoded_smiles))

        # Transpose from (nr_preds, batch_size) to (batch_size, nr_preds)
        decoded_smiles_str = list(map(list, zip(*decoded_smiles_str)))

        return decoded_smiles_str

    def decode_smiles_top_q_parallel(self, batch):
        decoded_smiles = self.top_q_decode(
                                batch,
                                nr_preds=self.k_predictions,
                                max_len=self.max_smiles_len,
                                temperature=self.temperature,
                                q=self.q
            )
        return [self.smiles_tokenizer.decode_batch(b) for b in decoded_smiles.tolist()]

    def top_k_decode(self, batch, nr_preds, max_len, temperature, k):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim)
            batch_size = spec.shape[0]
            # repeat input "nr_preds" times for predictions in one decode
            spec = spec.repeat_interleave(nr_preds, dim=0) # (batch_size * nr_preds, seq_len, in_dim)

            #### Calculate Memory (code snippet from greedy_decode function)
            src_key_padding_mask = self.generate_src_padding_mask(spec)
            
            spec = spec.permute(1, 0, 2)  # (seq_len, batch_size * nr_preds, in_dim)
            src = self.src_encoder(spec)  # (seq_len, batch_size * nr_preds, d_model)
            if self.embedding_norm:
                src = src / src.norm(dim=-1, keepdim=True)  # Normalize to unit norm
            #src = src * (self.d_model**0.5)
            memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
            ####

            # tensor that holds predicted tokens for each prediction
            preds = torch.full((batch_size, nr_preds, max_len), self.start_token_id, device=self.device)

            for i in range(max_len - 1):
                last_tokens = preds[:,:,:i+1].reshape(batch_size * nr_preds, i+1).T # (batch_size, nr_preds, seq_length) to (seq_length, batch_size * nr_preds)
                tgt = self.tgt_embedding(last_tokens)# * (self.d_model**0.5) # embedding scaling
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
                out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
                out = self.tgt_decoder(out[-1, :]) # (batch_size * nr_preds, vocab_size)

                ### Top-k sampling
                # Scale logits wrt temperature
                scaled_logits = out / temperature if temperature != None else out

                ## Apply top-k filtering
                # Select top-k tokens
                top_k_values, top_k_indices = torch.topk(scaled_logits, k, dim=-1)

                # Set logits for unselected tokens to -inf
                mask = torch.full_like(scaled_logits, float('-inf'))
                mask.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
                
                # Compute probabilities
                probabilities = F.softmax(mask, dim=-1)

                # Sample from the adjusted distribution
                next_tokens = torch.multinomial(probabilities, num_samples=1) # (batch_size * nr_preds)
                preds[:,:,i+1] = next_tokens.reshape(batch_size, nr_preds)

                # End loop when all sequences have end token
                if torch.all(next_tokens == self.end_token_id):
                    break

            return preds

    def top_q_decode(self, batch, nr_preds, max_len, temperature, q):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim)
            batch_size = spec.shape[0]

            # repeat input "nr_preds" times for predictions in one decode
            spec = spec.repeat_interleave(nr_preds, dim=0) # (batch_size * nr_preds, seq_len, in_dim)

            #### Calculate Memory (code snippet from greedy_decode function)
            src_key_padding_mask = self.generate_src_padding_mask(spec)
            
            spec = spec.permute(1, 0, 2)  # (seq_len, batch_size * nr_preds, in_dim)
            src = self.src_encoder(spec)  # (seq_len, batch_size * nr_preds, d_model)
            if self.embedding_norm:
                src = src / src.norm(dim=-1, keepdim=True)  # Normalize to unit norm
            memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
            ####

            # tensor that holds predicted tokens for each prediction
            preds = torch.full((batch_size, nr_preds, max_len), self.start_token_id, device=self.device)

            for i in range(max_len - 1):
                last_tokens = preds[:,:,:i+1].reshape(batch_size * nr_preds, i+1).T # (batch_size, nr_preds, seq_length) to (seq_length, batch_size * nr_preds)
                tgt = self.tgt_embedding(last_tokens)
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
                out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
                out = self.tgt_decoder(out[-1, :]) # (batch_size * nr_preds, vocab_size)

                ### Top-q sampling
                # Scale logits wrt temperature
                scaled_logits = out / temperature if temperature != None else out
                probs = F.softmax(scaled_logits, dim=-1)

                ## Apply top-q filtering
                # Select top-q tokens
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                top_q = torch.cumsum(sorted_probs, dim=-1) < q
                # Force most common token to always be selected (in case its prob > q)
                top_q[:,0] = torch.full_like(top_q[:,0], True)

                # Set logits for unselected tokens to -inf
                mask = torch.full_like(probs, float('-inf'))
                mask.scatter_(dim=-1, index=sorted_indices, src=sorted_probs.where(top_q, float('-inf')))
                
                # Compute probabilities
                probabilities = F.softmax(mask, dim=-1)

                # Sample from the adjusted distribution
                next_tokens = torch.multinomial(probabilities, num_samples=1) # (batch_size * nr_preds)
                preds[:,:,i+1] = next_tokens.reshape(batch_size, nr_preds)

                # End loop when all sequences have end token
                if torch.all(next_tokens == self.end_token_id):
                    break

            return preds

    def top_k_decode_one(self, batch, max_len, temperature, k):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim)
            batch_size = spec.shape[0]

            #### Calculate Memory (code snippet from greedy_decode function)
            src_key_padding_mask = self.generate_src_padding_mask(spec)

            spec = spec.permute(1, 0, 2)  # (seq_len, batch_size, in_dim)
            src = self.src_encoder(spec)  # (seq_len, batch_size, d_model)
            if self.embedding_norm:
                src = src / src.norm(dim=-1, keepdim=True)  # Normalize to unit norm
            memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
            ####

            # tensor that holds predicted tokens for each prediction
            preds = torch.full((batch_size, max_len), self.start_token_id, device=self.device)

            for i in range(max_len - 1):
                last_tokens = preds[:,:i+1].T # (batch_size, seq_length) to (seq_length, batch_size)
                tgt = self.tgt_embedding(last_tokens)# * (self.d_model**0.5) # embedding scaling
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

                out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
                out = self.tgt_decoder(out[-1, :]) # (batch_size, vocab_size)

                ### Top-k sampling
                # Scale logits wrt temperature
                scaled_logits = out / temperature if temperature != None else out

                ## Apply top-k filtering
                # Select top-k tokens
                top_k_values, top_k_indices = torch.topk(scaled_logits, k, dim=-1)

                # Set logits for unselected tokens to -inf
                mask = torch.full_like(scaled_logits, float('-inf'))
                mask.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
                
                # Compute probabilities
                probabilities = F.softmax(mask, dim=-1)
                
                # Sample from the adjusted distribution
                next_tokens = torch.multinomial(probabilities, num_samples=1) # (batch_size)
                preds[:,i+1] = next_tokens.squeeze(1)

                # End loop when all sequences have end token
                if torch.all(next_tokens == self.end_token_id):
                    break

            return preds

    def top_q_decode_one(self, batch, max_len, temperature, q):
        with torch.inference_mode():
            spec = self.scale_spec_batch(batch)    # (batch_size, seq_len, in_dim)
            batch_size = spec.shape[0]

            #### Calculate Memory (code snippet from greedy_decode function)
            src_key_padding_mask = self.generate_src_padding_mask(spec)

            spec = spec.permute(1, 0, 2)  # (seq_len, batch_size, in_dim)
            src = self.src_encoder(spec)  # (seq_len, batch_size, d_model)
            if self.embedding_norm:
                src = src / src.norm(dim=-1, keepdim=True)  # Normalize to unit norm
            memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
            ####

            # tensor that holds predicted tokens for each prediction
            preds = torch.full((batch_size, max_len), self.start_token_id, device=self.device)

            for i in range(max_len - 1):
                last_tokens = preds[:,:i+1].T # (batch_size, seq_length) to (seq_length, batch_size)
                tgt = self.tgt_embedding(last_tokens)# * (self.d_model**0.5) # embedding scaling
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

                out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
                out = self.tgt_decoder(out[-1, :]) # (batch_size, vocab_size)

                ### Top-q sampling
                # Scale logits wrt temperature
                scaled_logits = out / temperature if temperature != None else out
                probs = F.softmax(scaled_logits, dim=-1)

                ## Apply top-q filtering
                # Select top-q tokens
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                top_q = torch.cumsum(sorted_probs, dim=-1) < q
                # Force most common token to always be selected (in case its prob > q)
                top_q[:,0] = torch.full_like(top_q[:,0], True)

                # Set logits for unselected tokens to -inf
                mask = torch.full_like(probs, float('-inf'))
                mask.scatter_(dim=-1, index=sorted_indices, src=sorted_probs.where(top_q, float('-inf')))
                
                # Compute probabilities
                probabilities = F.softmax(mask, dim=-1)
                
                # Sample from the adjusted distribution
                next_tokens = torch.multinomial(probabilities, num_samples=1) # (batch_size)
                preds[:,i+1] = next_tokens.squeeze(1)

                # End loop when all sequences have end token
                if torch.all(next_tokens == self.end_token_id):
                    break

            return preds

