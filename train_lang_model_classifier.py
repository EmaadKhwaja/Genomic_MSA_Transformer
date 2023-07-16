SUBSAMPLE_LENGTH = 700
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
LANG_EMBEDDING = True
FLIP_ORIENTATION = False

import torch
from torch.utils.data import DataLoader
from operon_transformer import operon_transformer, LearnedPositionalEmbedding
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataloader import OperonLoader
import math
from tqdm import tqdm
import numpy as np
import gpn.mlm
from transformers import AutoModelForMaskedLM
from torch import nn




torch.random.manual_seed(42)

train_data = OperonLoader('data.csv',
                          subsample_length=SUBSAMPLE_LENGTH,
                          subsampling_method="regular_random",
                          shuffle_rows=False,
                          flip_orientation=FLIP_ORIENTATION,
                          include_position=INCLUDE_POSITION,
                          include_sequence=INCLUDE_SEQUENCE,
                          max_num_genes=MAX_NUM_GENES,
                          lang_embedding = LANG_EMBEDDING,
                          split_key="train",
                          im_set=IMG_SET,
                          device=DEVICE)
val_data = OperonLoader('data.csv',
                        subsample_length=SUBSAMPLE_LENGTH,
                        include_position=INCLUDE_POSITION,
                        include_sequence=INCLUDE_SEQUENCE,
                        lang_embedding=LANG_EMBEDDING,
                        max_num_genes=MAX_NUM_GENES,
                        split_key="valid",
                        im_set=IMG_SET,
                        device=DEVICE)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

model = operon_transformer_classifier(
    depth=DEPTH,
    ckpt_path='saved_models/operon_transformer_stable_last.pt',
    max_num_genes=MAX_NUM_GENES,
    alignment_length=SUBSAMPLE_LENGTH,
    include_position=INCLUDE_POSITION,
    include_sequence=INCLUDE_SEQUENCE,
    stable=False,
    attn_dropout=.1,
    ff_dropout=.1,
    lang_model = LANG_EMBEDDING).to(DEVICE)

optimizer = AdamW(model.parameters(),
                  lr=1e-3,
                  betas=(0.9, 0.96),
                  weight_decay=4.5e-2,
                  amsgrad=True)

epoch_train_losses = []
epoch_val_losses = []
best_val_loss = 1e1000

torch.cuda.empty_cache()

orientation_emb = nn.Embedding(3 + 2, dim)
orientation_pos_emb = LearnedPositionalEmbedding(SUBSAMPLE_LENGTH, DIM)

for i in range(EPOCHS):
    train_losses = []
    val_losses = []
    model.train()
    for j, batch in tqdm(enumerate(train_dataloader)):

        msa, score, label = batch

        optimizer.zero_grad()

        out = model(msa, score)

        loss = torch.nn.MSELoss()(out, label)

        loss.backward()

        if j % 500 == 0:
            print(
                f'{j}: Mean Loss: {np.mean([v for v in train_losses if not math.isnan(v) and not math.isinf(v)])}'
            )

        optimizer.step()

        train_losses.append(loss.item())

    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_dataloader):

            msa, score, label = batch

            out = model(msa, score)

            loss = torch.nn.MSELoss()(out, label)

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
        torch.save(
            model.state_dict(),
            f'saved_models/stable_operon_transformer_classifier_lr.001_{IMG_SET}_{i}.pt'
        )

    torch.save(
        model.state_dict(),
        f'saved_models/stable_operon_transformer_classifier_lr.001_last.pt')

    with open(f"saved_models/stable_classifier_lr.001_{IMG_SET}.txt",
              "w") as outfile:
        outfile.write('Train: ' + ",".join(str(i)
                                           for i in epoch_train_losses) + '\n')
        outfile.write('Val: ' + ",".join(str(i) for i in epoch_val_losses))