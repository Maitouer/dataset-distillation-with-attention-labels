from dataclasses import dataclass

import torch
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from torch import nn

MODEL_ATTRS = {
    "SASRec": {
        "dropout_keys": [
            "attention_probs_dropout_prob",
            "hidden_dropout_prob",
            "classifier_dropout",
        ],
    },
}


@dataclass
class ModelConfig:
    """Config for Learner Model"""

    task_name: str
    model_name: str = "SASRec"
    use_pretrained_model: bool = True
    disable_dropout: bool = True
    n_layers: int = 2
    n_heads: int = 2
    hidden_size: int = 64
    inner_size: int = 256
    hidden_dropout_prob: float = 0.5
    attn_dropout_prob: float = 0.5
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    loss_type: str = "CE"

    def __post_init__(self):
        assert self.model_name in MODEL_ATTRS


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attn_dropout_prob = config.attn_dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps

        self.initializer_range = config.initializer_range
        self.loss_type = config.loss_type

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def initialize_parameters(self):
        self.apply(self._init_weights)

    def _get_output(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def forward(self, interaction):
        """Synthetic data input"""
        if isinstance(interaction, torch.Tensor) and interaction.dim() == 3:
            extended_attention_mask = self.get_attention_mask(interaction[:, :-1, 0])
            item_seq_len = (
                torch.tensor(interaction.size(1) - 1)
                .unsqueeze(0)
                .expand_as(interaction[:, 0, 0])
                .to(interaction.device)
            )
            position_ids = (
                torch.arange(
                    interaction.size(1),
                    dtype=torch.long,
                    device=interaction.device,
                )
                .unsqueeze(0)
                .expand_as(interaction[:, :, 0])
            )
            position_embedding = self.position_embedding(position_ids)
            item_emb = interaction @ self.item_embedding.weight
            item_emb += position_embedding

            if self.loss_type == "BPR":
                pos_items_emb = item_emb[:, -2, :]
                neg_items_emb = item_emb[:, -1, :]
                input_emb = item_emb[:, :-2, :]
            elif self.loss_type == "CE":
                pos_items = (
                    torch.tensor(interaction.size(1) - 1)
                    .unsqueeze(0)
                    .expand_as(interaction[:, 0, 0])
                    .to(interaction.device)
                )
                input_emb = item_emb[:, :-1, :]

        """ Real data input"""
        if not (isinstance(interaction, torch.Tensor) and interaction.dim() == 3):
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]

            position_ids = torch.arange(
                item_seq.size(1), dtype=torch.long, device=item_seq.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)

            item_emb = self.item_embedding(item_seq)
            input_emb = item_emb + position_embedding
            input_emb = self.LayerNorm(input_emb)
            input_emb = self.dropout(input_emb)

            extended_attention_mask = self.get_attention_mask(item_seq)

            if self.loss_type == "BPR":
                pos_items = interaction[self.POS_ITEM_ID]
                pos_items_emb = self.item_embedding(pos_items)
                neg_items = interaction[self.NEG_ITEM_ID]
                neg_items_emb = self.item_embedding(neg_items)
            elif self.loss_type == "CE":
                pos_items = interaction[self.POS_ITEM_ID]

        """ Generate output and calculate loss """
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)

        if self.loss_type == "BPR":
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        elif self.loss_type == "CE":
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self._get_output(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self._get_output(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self._get_output(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
