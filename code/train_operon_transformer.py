SUBSAMPLE_LENGTH = 300
#more depth = better performance but slower and more memory required
DEPTH = 16
NUM_HEADS = 16
BATCH_SIZE = 1
# DIM OF LANGUAGE MODEL
DIM = 512
DEVICE = 'cuda:2'
EPOCHS = 200
#"bsubtilis" or "ecoli"
IMG_SET = "bsubtilis"
#odd number
MAX_NUM_GENES = 13
INCLUDE_POSITION = False
INCLUDE_SEQUENCE = False
FLIP_ORIENTATION = False

import torch
from torch.utils.data import DataLoader
from operon_transformer import operon_transformer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataloader import OperonLoader
import math
from tqdm import tqdm
import numpy as np


torch.random.manual_seed(42)

train_data = OperonLoader('data.csv',
                          subsample_length=SUBSAMPLE_LENGTH,
                          subsampling_method="regular_random",
                          shuffle_rows=False,
                          flip_orientation=FLIP_ORIENTATION,
                          include_position=INCLUDE_POSITION,
                          include_sequence=INCLUDE_SEQUENCE,
                          max_num_genes=MAX_NUM_GENES,
                          split_key="train",
                          im_set=IMG_SET,
                          device=DEVICE)
val_data = OperonLoader('data.csv',
                        subsample_length=SUBSAMPLE_LENGTH,
                        include_position=INCLUDE_POSITION,
                        include_sequence=INCLUDE_SEQUENCE,
                        max_num_genes=MAX_NUM_GENES,
                        split_key="valid",
                        im_set=IMG_SET,
                        device=DEVICE)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

model = operon_transformer(depth=DEPTH,
                           max_num_genes=MAX_NUM_GENES,
                           heads = NUM_HEADS,
                           alignment_length=SUBSAMPLE_LENGTH,
                           include_position=INCLUDE_POSITION,
                           include_sequence=INCLUDE_SEQUENCE,
                           attn_dropout=.1,
                           ff_dropout=.1).to(DEVICE)

optimizer = AdamW(model.parameters(),
                  lr=3e-4,
                  betas=(0.9, 0.96),
                  weight_decay=4.5e-2,
                  amsgrad=True)

epoch_train_losses = []
epoch_val_losses = []
best_val_loss = 1e1000

torch.cuda.empty_cache()


for i in range(EPOCHS):
    train_losses = []
    val_losses = []
    model.train()
    for j, batch in tqdm(enumerate(train_dataloader)):

        msa, _, _ = batch

        optimizer.zero_grad()

        loss = model(msa, return_loss=True)

        if j % 100 == 0:
            print(f'{j}: {loss.item()}')
        loss.backward()

        optimizer.step()

        train_losses.append(loss.item())

    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            msa, _, _ = batch
            loss = model(msa, return_loss=True)
            val_losses.append(loss.item())
    train_losses = [
        v for v in train_losses if not math.isnan(v) and not math.isinf(v)
    ]
    val_losses = [
        v for v in val_losses if not math.isnan(v) and not math.isinf(v)
    ]
    epoch_train_losses.append(np.sum(train_losses) / len(train_dataloader))
    epoch_val_losses.append(np.sum(val_losses) / len(val_dataloader))

    print(
        f'Epoch {i} - Train Loss: {epoch_train_losses[-1]:.4f} - Val Loss: {epoch_val_losses[-1]:.4f}'
    )

    if epoch_val_losses[-1] < best_val_loss:
        best_val_loss = epoch_val_losses[-1]
        torch.save(model.state_dict(),
                   f'saved_models/operon_transformer_stable_{IMG_SET}_{i}.pt')

    torch.save(model.state_dict(),
               f'saved_models/operon_transformer_stable_last.pt')

    with open(f"saved_models/{IMG_SET}.txt", "w") as outfile:
        outfile.write('Train: ' + ",".join(str(i)
                                           for i in epoch_train_losses) + '\n')
        outfile.write('Val: ' + ",".join(str(i) for i in epoch_val_losses))