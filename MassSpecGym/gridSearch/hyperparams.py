import numpy as np
from decimal import Decimal
import itertools

lr = [10 ** -x for x in np.arange(3, 4.01, 0.5)]
batch_size = [512, 1024, 2048]
d_model = [128, 256, 512]
nhead = [2, 4, 8]
num_encoder_layers = [2, 3, 4]
num_decoder_layers = [2, 3, 4]
#temperature = [0.8, 1.0, 1.2]

print(f"DEBUG:lr {['%.2E' % Decimal(x) for x in lr]}")
print(f"DEBUG:batch size {batch_size}")
print(f"DEBUG:d_model {d_model}")
print(f"DEBUG:nhead {nhead}")
print(f"DEBUG:num_encoder_layers {num_encoder_layers}")
print(f"DEBUG:num_decoder_layers {num_decoder_layers}")
#print(f"DEBUG:temperature {temperature}")
assert False
for _lr, _batch_size, _d_model, _nhead, _num_enc, _num_dec in itertools.product(lr, batch_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
    _temp = 1.0
    print(f"--lr {_lr} --batch_size {_batch_size} --d_model {_d_model} --nhead {_nhead} --num_encoder_layers {_num_enc} --num_decoder_layers {_num_dec} --temperature {_temp}")
