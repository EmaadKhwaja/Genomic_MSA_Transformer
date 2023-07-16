from xml.etree.ElementInclude import include
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import gpn.mlm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


class OperonLoader(Dataset):

    def __init__(self,
                 config_file,
                 full_sequence_length=5000,
                 num_alignments=20,
                 subsample_length=1000,
                 subsampling_method="regular",
                 split_key=None,
                 im_set=None,
                 shuffle_rows=True,
                 flip_orientation=False,
                 include_position=False,
                 include_sequence=False,
                 max_num_genes=float("inf"),
                 lang_embedding =False,
                 device="cpu"):

        data = pd.read_csv(config_file)

        self.config_file = config_file
        self.full_sequence_length = full_sequence_length
        self.num_alignments = num_alignments
        self.include_sequence = include_sequence
        self.subsample_length = subsample_length
        self.split_key = split_key
        self.device = device
        self.lang_embedding = lang_embedding
        if max_num_genes % 2 == 0:
            max_num_genes -= 1
        self.max_num_genes = max_num_genes

        # dont do data aug if val or test
        if self.split_key == "val" or self.split_key == "test":
            self.subsampling_method = "regular"
            self.shuffle_rows = False
            self.flip_orientation = False

        else:
            self.subsampling_method = subsampling_method
            self.shuffle_rows = shuffle_rows
            self.flip_orientation = flip_orientation

        if split_key == "train":
            self.data = data[data["split"] == "train"]
        elif split_key == "val":
            self.data = data[data["split"] == "val"]
        elif split_key == "test":
            self.data = data[data["split"] == "test"]
        else:
            self.data = data

        if im_set:
            self.data = self.data[self.data["im_set"] == im_set]

        self.num_channels = 4

        self.include_position = include_position
        self.include_sequence = include_sequence
        model_path = "gonzalobenegas/gpn-arabidopsis"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path).to(self.device)
        self.embedding_model = AutoModel.from_pretrained(model_path).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        json_path = self.data.iloc[idx]["path"]
        score = self.data.iloc[idx]["score"]
        label = self.data.iloc[idx]["label"]

        operon_msa = torch.ones(self.num_channels,
                                self.num_alignments,
                                self.full_sequence_length,
                                device=self.device)
        operon_msa[0] = 0

        with open(json_path, "r") as read_file:
            operon_data = json.load(read_file)["result"][0]
            for idx, genes in enumerate(operon_data):
                sub_fids = {}
                count = 1
                pinned_peg = genes["pinned_peg"]
                for fid_features in genes["features"]:
                    fid = fid_features["fid"]
                    if fid not in sub_fids:
                        sub_fids[fid] = count
                        count += 1

                if pinned_peg in sub_fids:
                    pinned_peg_fid_value = sub_fids[pinned_peg]
                    for key, value in sub_fids.items():
                        sub_fids[key] = max(
                            -self.max_num_genes // 2,
                            min(value - pinned_peg_fid_value,
                                self.max_num_genes // 2))

                fids.append(sub_fids)

            # shift entire position dict to be > 0
            min_pos = min(min(fids[i].values()) for i in range(len(fids)))
            for jdx, sub_fids in enumerate(fids):
                for key, value in sub_fids.items():
                    fids[jdx][key] = value - min_pos + 1

            # Go through again to assign values to the MSA
            for idx, genes in enumerate(operon_data):
                min_position = min(genes["beg"], genes["end"])
                for fid_features in genes["features"]:
                    fid = fid_features["fid"]
                    if "strand" in fid_features.keys():
                        strand = fid_features["strand"]
                        if strand == "+":
                            start = int(fid_features["beg"]) - min_position
                            end = int(fid_features["end"]) - min_position
                            operon_msa[0, idx, start:end] = 1
                        elif strand == "-":
                            start = int(fid_features["end"]) - min_position
                            end = int(fid_features["beg"]) - min_position
                            operon_msa[0, idx, start:end] = 2

                        operon_msa[1, idx, start:end] = fids[idx][fid]

        if self.include_position:
            # add indices so transformer knows positions after subsampling
            indices_tensor = (torch.tensor(
                range(1, self.full_sequence_length + 1),
                device=self.device).repeat(self.num_alignments,
                                           1).unsqueeze(0))
        else:
            indices_tensor = torch.zeros(1,
                                         self.num_alignments,
                                         self.full_sequence_length,
                                         device=self.device)

        if self.include_sequence:

            # TO-DO (Retrieve sequence from metadata)
            #sequence = 
            sequence = torch.stack([torch.nn.functional.pad(self.tokenizer(seq, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"],self.full_sequence_length,'constant',1) for seq in sequence])
            
            sequence = sequence.to(device)


        else:
            sequence_tensor = torch.zeros(1,
                                          self.num_alignments,
                                          self.full_sequence_length,
                                          device=self.device)

        if self.lang_embedding:

            with torch.no_grad():
                embedding = self.embedding_model(input_ids=sequence).last_hidden_state

            return (operon_msa[0], operon_msa[1], indices_tensor, embedding), torch.tensor(
            score, device=self.device), torch.tensor(
                label, device=self.device).unsqueeze(0).double()

        # randomly select rows to downsample tensor
        if self.subsampling_method == "random":
            indices = torch.randperm(
                operon_msa.shape[-1])[:self.subsample_length]
            operon_msa = operon_msa[:, :, indices]

        # downsample tensor to subsample_length
        elif self.subsampling_method == "regular":
            scale_factor = operon_msa.shape[-1] // self.subsample_length
            operon_msa = operon_msa[:, :, ::scale_factor]

        # downsample at regular intervals but with random offset
        elif self.subsampling_method == "regular_random":
            scale_factor = operon_msa.shape[-1] // self.subsample_length
            operon_msa = operon_msa[:, :,
                                    torch.
                                    randint(low=0, high=scale_factor, size=(
                                    )).item()::scale_factor, ]

        # rearrange rows
        if self.shuffle_rows:
            operon_msa = operon_msa[:,
                                    torch.randperm(operon_msa.shape[1]).sort(
                                    )[0], :]

        # randomly flip entire horizontally
        if self.flip_orientation:
            if torch.rand(1).item() > 0.5:
                operon_msa = torch.flip(operon_msa, [-1])

        # make sure correct length after subsampling
        operon_msa = operon_msa[:, :, :self.subsample_length]

        return operon_msa, torch.tensor(
            score, device=self.device), torch.tensor(
                label, device=self.device).unsqueeze(0).double()
