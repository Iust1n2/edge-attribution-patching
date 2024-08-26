import os
import sys
sys.path.append('..')
sys.path.append('../../Automatic-Circuit-Discovery/')
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
import torch as t
from torch import Tensor
import json

from acdc.hybridretrieval.utils import (
    get_all_hybrid_retrieval_things,
    get_gpt2_small
)

from transformer_lens import HookedTransformer, ActivationCache

import tqdm.notebook as tqdm
import plotly

device = t.device("cuda" if t.cuda.is_available() else "CPU")
print(device)

all_kbicr_items = get_all_hybrid_retrieval_things(num_examples=20, device=device, metric_name='logit_diff')

tl_model = all_kbicr_items.tl_model
validation_metric = all_kbicr_items.validation_metric
validation_data = all_kbicr_items.validation_data
validation_labels = all_kbicr_items.validation_labels
validation_patch_data = all_kbicr_items.validation_patch_data
test_metrics = all_kbicr_items.test_metrics
test_data = all_kbicr_items.test_data
test_labels = all_kbicr_items.test_labels
test_patch_data = all_kbicr_items.test_patch_data

model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

def abs_kbicr_metric(logits):
    return -abs(test_metrics['logit_diff'](logits))


# os.chdir(os.getcwd())
os.getcwd()

from ACDCPPExperiment import ACDCPPExperiment
import numpy as np
THRESHOLD = [0.001] # np.arange(0.04, 0.16, 0.005)
# I'm just using one threshold so I can move fast!

tl_model.reset_hooks()
RUN_NAME = f'abs_edges/{THRESHOLD}'
acdcpp_exp = ACDCPPExperiment(
    tl_model,
    test_data,
    test_patch_data,
    test_metrics['logit_diff'],
    abs_kbicr_metric,
    thresholds=THRESHOLD,
    local_dir=RUN_NAME,
    verbose=False,
    attr_absolute_val=True,
    save_graphs_after=0,
    pruning_mode='edge',
    no_pruned_nodes_attr=1,
)

import datetime
exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
start_time = datetime.datetime.now()

pruned_heads, pruned_attrs = acdcpp_exp.run()

elapsed_time = datetime.datetime.now() -  start_time
print(f'Elapsed time: {elapsed_time}')