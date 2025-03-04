import numpy as np
from decimal import Decimal
import itertools
from collections import OrderedDict

extra_params = [
 ("smiles_tokenizer", "tokenizers/smiles_tokenizer_4M.pkl"),
 #("full_selfies_vocab", ""),
 ("store_metadata", ""),
 #("k_predictions", 1)
]

hyperparams = {
    "temperature": [0.9, 1.0],
    "k": [3],# 5, 10, 20, 50],
    "q": [0.8, 0.9, 0.95],#[0.7, 0.8, 0.9, 0.95, 0.99],
    "beam_width": [10, 20],
    "alpha": [1.0],
    "sampler": ["top-q-parallel"],#, "top-k", "top-q"]
    "checkpoint_pth": ["models/smiles_lowest_val_loss.ckpt"] #["models/smiles_augmented_1.ckpt", "models/smiles_augmented_2.ckpt", "models/smiles_augmented_5.ckpt"]
}

samplers = hyperparams["sampler"]
samplers_allowed_params = {
    "naive": ["checkpoint_pth", "temperature"],
    "naive-parallel": ["checkpoint_pth", "temperature"],
    "top-k": ["checkpoint_pth", "temperature", "k"],
    "top-q": ["checkpoint_pth", "temperature", "q"],
    "top-q-parallel": ["checkpoint_pth", "temperature", "q"],
    "beam-search": ["checkpoint_pth", "beam_width", "alpha"]
}

shortened_params = {
    "temperature": "temp",
    "k": "k",
    "q": "q",
    "beam_width": "bw",
    "alpha": "alph",
    "sampler": "",
    "checkpoint_pth": ""
}


def sanitize_decimal(x, p):
    a, b = (f"%.{p-1}E" % Decimal(x)).split("E")
    return str(int(Decimal(a) * (10 ** (p-1)))) + "e" + str(int(b)-(p-1))

sanitized_values = {
    "temperature": lambda x: sanitize_decimal(x, 2),
    "k": lambda x:str(x),
    "q": lambda x: sanitize_decimal(x, 2),
    "beam_width": lambda x:str(x),
    "alpha": lambda x:sanitize_decimal(x, 2),
    "sampler": lambda x:x,
    "checkpoint_pth": lambda x: x.split("/")[1].split(".")[0],
}


for k, v in hyperparams.items():
    print("DEBUG:", k, ":", [sanitized_values[k](x) for x in v])

for s in samplers:
    params = samplers_allowed_params[s]
    for values in itertools.product(*[hyperparams[p] for p in params]):
        job_args = [("sampler", s)] + list(zip(params, values))
        args = " ".join(["--" + k + " " + str(v) for k, v in extra_params + job_args])
        jobname = "_".join([shortened_params[k] + sanitized_values[k](v) for k, v in job_args]) 
        print(jobname + "|" + args)

