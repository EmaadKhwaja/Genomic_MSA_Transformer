import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from operon_transformer.transformer import MSATransformer


class DivideMax(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True).detach()
        return x / maxes


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = 0):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) *
                     mask).long() + self.padding_idx

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class operon_transformer(nn.Module):

    def __init__(self,
                 *,
                 depth,
                 dim=768,
                 stable=True,
                 max_num_genes=83,
                 num_alignments=20,
                 alignment_length=1000,
                 original_base_pair_length=5000,
                 heads=8,
                 attn_dropout=0.0,
                 ff_dropout=0,
                 include_position=False,
                 include_sequence=False):
        super().__init__()

        if max_num_genes % 2 == 0:
            max_num_genes -= 1
        self.max_num_genes = max_num_genes
        self.original_base_pair_length = original_base_pair_length
        self.alignment_length = alignment_length
        self.num_alignments = num_alignments
        self.include_position = include_position
        self.include_sequence = include_sequence
        self.stable = stable

        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        size = (alignment_length + 2) * num_alignments

        self.orientation_emb = nn.Embedding(3 + 2, dim)
        self.orientation_pos_emb = LearnedPositionalEmbedding(
            original_base_pair_length + 2, dim)
        self.orientation_transformer = MSATransformer(
            embed_dim=dim,
            num_layers=depth,
            num_attention_heads=heads,
            dropout=.1,
            attention_dropout=attn_dropout,
            activation_dropout=ff_dropout,
            max_tokens_per_msa=size,
            num_tokens=3 + 2,
            embed_tokens=self.orientation_emb)

        self.gene_id_emb = nn.Embedding(max_num_genes + 2, dim)
        self.gene_id_pos_emb = LearnedPositionalEmbedding(
            original_base_pair_length + 2, dim)
        self.gene_id_transformer = MSATransformer(
            embed_dim=dim,
            num_layers=depth,
            num_attention_heads=heads,
            dropout=.1,
            attention_dropout=attn_dropout,
            activation_dropout=ff_dropout,
            max_tokens_per_msa=size,
            num_tokens=max_num_genes + 2,
            embed_tokens=self.gene_id_emb)

        if include_position:

            self.position_emb = nn.Embedding(original_base_pair_length + 2,
                                             dim)
            self.position_pos_emb = LearnedPositionalEmbedding(
                original_base_pair_length + 2, dim)
            self.position_transformer = MSATransformer(
                embed_dim=dim,
                num_layers=depth,
                num_attention_heads=heads,
                dropout=.1,
                attention_dropout=attn_dropout,
                activation_dropout=ff_dropout,
                max_tokens_per_msa=size,
                num_tokens=original_base_pair_length + 2,
                embed_tokens=self.position_emb)

        if include_sequence:

            self.sequence_emb = nn.Embedding(4 + 2, dim)
            self.sequence_pos_emb = LearnedPositionalEmbedding(
                original_base_pair_length + 2, dim)
            self.sequence_transformer = MSATransformer(
                embed_dim=dim,
                num_layers=depth,
                num_attention_heads=heads,
                dropout=.1,
                attention_dropout=attn_dropout,
                activation_dropout=ff_dropout,
                max_tokens_per_msa=size,
                num_tokens=4 + 2,
                embed_tokens=self.position_emb)

    def forward(self, msa, return_loss=False):

        # add <bos>

        orientation, gene_id, position, sequence = msa[:,
                                                       0], msa[:,
                                                               1], msa[:,
                                                                       2], msa[:,
                                                                               3]

        orientation = F.pad(F.pad(orientation, (1, 0), value=3), (0, 1),
                            value=4)
        orientation_tokens = self.orientation_emb(orientation)
        orientation_tokens += self.orientation_pos_emb(orientation)
        orientation_out = self.orientation_transformer(orientation_tokens)
        orientation_logits = orientation_out['logits']
        orientation_logits = orientation_logits.permute(2, 3, 0, 1)


        gene_id = F.pad(F.pad(gene_id, (1, 0), value=self.max_num_genes),
                        (0, 1),
                        value=self.max_num_genes + 1)
        gene_id_tokens = self.gene_id_emb(gene_id)
        gene_id_tokens += self.gene_id_pos_emb(gene_id)
        gene_id_out = self.gene_id_transformer(gene_id_tokens)
        gene_id_logits = gene_id_out['logits']
        gene_id_logits = gene_id_logits.permute(2, 3, 0, 1)

        if self.stable:
            orientation_logits = self.norm_by_max(orientation_logits)
            gene_id_logits = self.norm_by_max(gene_id_logits)

        if self.include_position:

            position = F.pad(F.pad(position, (1, 0),
                                   value=self.original_base_pair_length),
                             (0, 1),
                             value=self.original_base_pair_length + 1)
            position_tokens = self.position_emb(position)
            position_tokens += self.position_pos_emb(position)
            position_out = self.position_transformer(position_tokens)
            position_logits = position_out['logits']
            position_logits = position_logits.permute(2, 3, 0, 1)

            if self.stable:
                position_logits = self.norm_by_max(position_logits)

        else:
            position_logits = torch.zeros(gene_id_logits.shape)

        if self.include_sequence:
            sequence = F.pad(F.pad(sequence, (1, 0), value=4), (0, 1),
                             value=4 + 1)
            sequence_tokens = self.sequence_emb(sequence)
            sequence_tokens += self.sequence_pos_emb(sequence)
            sequence_out = self.sequence_transformer(sequence_tokens)
            sequence_logits = sequence_out['logits']
            sequence_logits = sequence_logits.permute(2, 3, 0, 1)

            if self.stable:
                sequence_logits = self.norm_by_max(sequence_logits)

        else:
            sequence_logits = torch.zeros(gene_id_logits.shape)

        if not return_loss:
            return orientation_logits, gene_id_logits, position_logits, sequence_logits

        orientation_logits = orientation_logits[:, :, :, 1:-1]
        gene_id_logits = gene_id_logits[:, :, :, 1:-1]

        if self.include_position:
            position_logits = position_logits[:, :, :, 1:-1]

        if self.include_sequence:
            sequence_logits = sequence_logits[:, :, :, 1:-1]

        orientation_logits[:, -2:] = -torch.finfo(orientation_logits.dtype).max
        orientation_loss = F.cross_entropy(orientation_logits, msa[:, 0])
        self.orientation_logits = orientation_logits
        loss = orientation_loss

        gene_id_logits[:, -2:] = -torch.finfo(gene_id_logits.dtype).max
        gene_id_loss = F.cross_entropy(gene_id_logits, msa[:, 1])
        self.gene_id_logits = gene_id_logits
        loss += gene_id_loss

        if self.include_position:
            position_logits[:, -2:] = -torch.finfo(position_logits.dtype).max
            position_loss = F.cross_entropy(position_logits, msa[:, 2])

            loss += position_loss

        if self.include_sequence:
            sequence_logits[:, -2:] = -torch.finfo(sequence_logits.dtype).max
            sequence_loss = F.cross_entropy(sequence_logits, msa[:, 3])

            loss += sequence_loss

        return loss


class operon_transformer_classifier(nn.Module):

    def __init__(self,
                 *,
                 depth,
                 ckpt_path=False,
                 dim=768,
                 stable=True,
                 max_num_genes=83,
                 num_alignments=20,
                 alignment_length=1000,
                 original_base_pair_length=5000,
                 heads=8,
                 dropout=.1,
                 attn_dropout=0.1,
                 ff_dropout=.1,
                 include_position=False,
                 include_sequence=False,
                 lang_model=False):

        super(operon_transformer_classifier, self).__init__()

        self.lang_model = lang_model

        if not lang_model:

            self.model = operon_transformer(depth=depth,
                                            stable=stable,
                                            max_num_genes=max_num_genes,
                                            alignment_length=alignment_length,
                                            include_position=include_position,
                                            include_sequence=include_sequence,
                                            attn_dropout=attn_dropout,
                                            ff_dropout=ff_dropout)
            if ckpt_path:
                self.model.load_state_dict(torch.load(ckpt_path), strict=False)
            self.model.eval()

            self.num_channels = 2
            if include_position:
                self.num_channels += 1
            if include_sequence:
                self.num_channels += 1

        self.conv2d = nn.Conv2d(in_channels=20,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.linear = nn.Linear(in_features=302 * 20, out_features=1)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, msa, score):

        if not self.lang_model:

            out = self.model(msa, return_loss=False)
            out = torch.cat(out[:self.num_channels], dim=1)
            out = self.conv2d(out)
            out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        out = self.dropout(out)
        out = self.softmax(out)
        out = out * score

        return out
