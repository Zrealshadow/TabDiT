Overview

During training, TabICL generates synthetic datasets on-the-fly using the prior module. Here's the flow:

PriorDataset → DataLoader → Training Loop → Model

Configuration in Training (run.py:226-276)

## On-the-fly generation (default)
dataset = PriorDataset(
   batch_size=512,
   batch_size_per_gp=4,      # Datasets sharing same hyperparameters
   prior_type='mix_scm',     # MLP + Tree SCMs
   max_features=100,
   max_seq_len=1024,
   min_train_size=0.1,
   max_train_size=0.9,
   n_jobs=1,                 # Avoid nested parallelism with DDP
)

dataloader = DataLoader(dataset, batch_size=None, prefetch_factor=4)

Generation Pipeline (per batch)

1. Sample hyperparameters per group
   └─ batch_size_per_gp datasets share: seq_len, train_size, SCM config

2. For each dataset:
   ├─ Create random SCM (MLPSCM or TreeSCM with random weights)
   ├─ Generate X, y via forward pass (no training!)
   ├─ Reg2Cls: regression → classification
   │   ├─ Categorize ~20% of features
   │   ├─ Normalize features (outlier removal + standardization)
   │   └─ Bin targets into classes
   └─ Validate: remove constant features, check class balance

3. Stack into batch tensors
   └─ X: (batch_size, seq_len, max_features)
   └─ y: (batch_size, seq_len)

Training Loop (run.py:413-668)

for step in range(max_steps):
   # 1. Get batch (infinite iterator)
   X, y, d, seq_lens, train_sizes = next(dataloader)

   # 2. Split into micro-batches for gradient accumulation
   for micro_X, micro_y, micro_d in micro_batches:
         # 3. Split train/test
         y_train = micro_y[:, :train_size]
         y_test = micro_y[:, train_size:]

         # 4. Forward pass - model sees train data, predicts test
         pred = model(micro_X, y_train, micro_d)
         loss = cross_entropy(pred, y_test)

         # 5. Backward
         loss.backward()

      optimizer.step()
   


## Pre-generate and save to disk:
python -m tabicl.prior.genload \
   --save_dir ./prior_data \
   --num_batches 1000 \
   --batch_size 512 \
   --prior_type mix_scm \
   --max_features 100 \
   --max_seq_len 1024 \
   --n_jobs -1

## Load pre-generated data:
from tabicl.prior.genload import LoadPriorDataset

loader = LoadPriorDataset(
   data_dir='./prior_data',
   batch_size=256,
   ddp_rank=0,
   ddp_world_size=1
)

for batch in loader:
   X, y, d, seq_lens, train_sizes = batch
   # train your model