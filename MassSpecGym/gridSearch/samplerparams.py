import numpy as np
from decimal import Decimal
import itertools
from collections import OrderedDict

# TODO change to real path
model_path = "smiles_lowest_val_loss"
model_name = "SMILES_LOW_VAL_LOSS"

extra_params = [
 ("smiles_tokenizer", "smiles"),
 ("full_selfies_vocab", "False"),
]

hyperparams = {
    "temperature": [0.7, 0.9, 1.0, 1.1, 1.3, 1.5],
    "k": [3, 5, 10, 20, 50],
    "q": [0.7, 0.8, 0.9, 0.95, 0.99],
    "sampler": ["greedy", "top-k", "top-q"]
}

samplers = hyperparams["sampler"]
samplers_allowed_params = {
    "greedy": ["temperature"],
    "top-k": ["temperature", "k"],
    "top-q": ["temperature", "q"]
}

shortened_params = {
    "temperature": "temp",
    "k": "k",
    "q": "q",
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
    "sampler": lambda x:x,
    "checkpoint_pth": lambda x: model_name
}


for k, v in hyperparams.items():
    print("DEBUG:", k, ":", [sanitized_values[k](x) for x in v])

for s in samplers:
    params = samplers_allowed_params[s]
    for values in itertools.product(*[hyperparams[p] for p in params]):
        job_args = [("checkpoint_pth", model_path), ("sampler", s)] + list(zip(params, values))
        args = " ".join(["--" + k + " " + str(v) for k, v in extra_params + job_args])
        jobname = "_".join([shortened_params[k] + sanitized_values[k](v) for k, v in job_args])
        print(jobname + "|" + args)

