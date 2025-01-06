import numpy as np
from decimal import Decimal
import itertools
from collections import OrderedDict

hyperparams = OrderedDict({
    "lr": [10 ** -x for x in np.arange(3, 4.01, 0.5)],
    "batch_size": [512, 1024, 2048],
    "d_model": [128, 256, 512],
    "nhead": [2, 4, 8],
    "num_encoder_layers": [2, 3, 4],
    "num_decoder_layers": [2, 3, 4],
    "temperature": [1.0],
    "sampler": ["top-k"],
    "k": [10],
})

for k, v in hyperparams.items():
    print("DEBUG:", k, ":", (v if k != "lr" else ['%.2E' % Decimal(x) for x in v]))

shortened_params = {
    "lr": "lr",
    "batch_size": "bs",
    "d_model": "dmod",
    "nhead": "nh",
    "num_encoder_layers": "enc",
    "num_decoder_layers": "dec",
}

for values in itertools.product(*hyperparams.values()):
    args = " ".join(["--" + k + " " + str(v) for k, v in zip(hyperparams.keys(), values)])
    jobname = "_".join([shortened_params[k] + ((str(round(v)) if type(v) != str else v) if k != "lr" else '%.0E' % Decimal(v)) for k, v in zip(hyperparams.keys(), values) if k in shortened_params])
    print(jobname + "|" + args)
