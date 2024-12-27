#!/usr/bin/env python
# coding: utf-8


# In[2]:


import random
import numpy as np
from rdkit import RDLogger
import pytorch_lightning as pl
from massspecgym.models.tokenizers import SmilesBPETokenizer
from massspecgym.data.transforms import SpecTokenizer
from massspecgym.data import MassSpecDataset, MassSpecDataModule
from massspecgym.models.base import Stage
from smiles_transformer import SmilesTransformer
import pickle
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
from pathlib import Path
import time

# In[3]:


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


# Suppress RDKit warnings and errors
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# In[5]:


# seed for reproducability
pl.seed_everything(0)


# # Setup

# In[6]:


tokenizer = SmilesBPETokenizer(max_len=200)


# In[7]:


dataset = MassSpecDataset(
            pth=None, # use massSpecGym dataset
            spec_transform = SpecTokenizer(n_peaks=60, matchms_kwargs=dict(mz_to=1005)),
            mol_transform= None
        )


# In[8]:


data_module = MassSpecDataModule(
        dataset=dataset,
        split_pth=None,
        batch_size=128,
        num_workers=1,
    )


# In[9]:


data_module.prepare_data()
data_module.setup()


# In[10]:


batch = next(iter(data_module.train_dataloader()))

batch["spec"] = batch["spec"].to(device)

# In[11]:


common_kwargs = dict(
    lr=1e-4,
    weight_decay=0.0,
    log_only_loss_at_stages=[Stage('train'), Stage('val')],
    #df_test_path=f"./testresults/de_novo/{run_name}_{now_formatted}.pkl",
)


# In[12]:


def createTransFormer(nr_preds, max_smiles_len):
    return SmilesTransformer(
        input_dim=2,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout=0.1,
        smiles_tokenizer=tokenizer,
        k_predictions=nr_preds,
        pre_norm=False,
        max_smiles_len=max_smiles_len,
        k=50,
        **common_kwargs
    )


# # Benchmark (Execution Time)

# In[13]:


max_len = [10*(i+1) for i in range(30)] # 10, 20, 30, ..., 200
nr_preds = [1, 10]

def timeDecoder(decoder):
    times = {}
    for _nr_preds in nr_preds:
        print(f"nr_preds: {_nr_preds}")
        times[_nr_preds] = []
        for _max_len in max_len:
            print(f"max_len: {_max_len}")
            
            m = createTransFormer(_nr_preds, _max_len)
            m.cuda(device)
            
            start_time = time.time()
            
            if decoder == 'greedy':
                m.decode_smiles(batch)
            elif decoder == 'top-k':
                m.decode_smiles_top_k(batch)
            elif decoder == 'top-k-parallel':
                m.decode_smiles_top_k_parallel(batch)
            else:
                raise Exception('Unknown decoder')
                
            times[_nr_preds].append(time.time()-start_time)

    return times


# ## Greedy Decode


# In[ ]:

greedy_times = timeDecoder('greedy')
topk_times = timeDecoder('top-k')
topk_parallel_times = timeDecoder('top-k-parallel')

with open('samplersBenchmarkResults.pkl', 'wb') as f:
    pickle.dump({
        "greedy_times": greedy_times,
        "topk_times": topk_times,
        "topk_parallel_times": topk_parallel_times,
    }, f)

# In[ ]:



