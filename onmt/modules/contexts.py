from onmt.utils.misc import block_eye, tile

import torch


class ContainsNaN(Exception):
    pass


def check_object_for_nan(obj):
    if isinstance(obj, torch.nn.Module):
        for name, tensor in obj.named_parameters():
            if (tensor != tensor).any():
                raise ContainsNaN(name)
    elif isinstance(obj, torch.Tensor):
        if (obj != obj).any():
            raise ContainsNaN()
    elif isinstance(obj, (list, tuple)):
        for _obj in obj:
            check_object_for_nan(_obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            try:
                check_object_for_nan(value)
            except ContainsNaN:
                raise ContainsNaN(key)


class ContextPredictor(torch.nn.Module):
    """
    Predicts a context from a decoder state.
    """
    def __init__(self, decoder_hidden_size, elaboration_vocab):
        super().__init__()

        self.elaboration_predictor = torch.nn.Linear(decoder_hidden_size,
                                                     len(elaboration_vocab))

    def forward(self, ):
        pass


class Aggregation(torch.nn.Module):
    """
    Uses self attention to aggregate a variable number of representations.
    """
    def __init__(self, dim, heads=1, dropout=0, do_proj=False):
        super(Aggregation, self).__init__()

        self.dim = dim
        self.heads = heads
        self._do_proj = do_proj

        if self._do_proj:
            self.key_linear = torch.nn.Linear(dim, dim)
            self.val_linear = torch.nn.Linear(dim, dim)

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=heads,
            dropout=dropout,
        )

        query = torch.nn.Parameter(torch.Tensor(1, 1, self.dim))
        self.register_parameter('query', query)

        self._init_parameters_manually()

        # once the module is initialized, check for NaNs
        check_object_for_nan(self)

    def _init_parameters_manually(self):
        torch.nn.init.uniform_(self.query, -1, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def make_context_query_mask(self, n_sents, n_ents, batch_size):
        """
        Buils a padding mask, for the entity aggregation step in computing
        context representations. This is a bit convoluted, so here's an
        explanation:

        1) Given entities, we want to aggregate their repr using MultiHeadAttn
        2) However, we'll do it for all sentences, so we have n_ents * n_sents
           as a batch_size
        3) But we also have this for each example of the batch
        --> We need to deal with this "double batch" situation
        4) We put everything as one big batch, but now we need to make n_sents
           distinct queries. This means that the first query has access to the
           first n_ents lines, the second has access to 5-8, etc.
        """

        mask = block_eye(n_sents, n_ents, dtype=torch.bool, device=self.device)
        mask = mask[range(0, n_sents * n_ents, n_ents)].unsqueeze(0)
        mask = mask.expand(batch_size, -1, -1)

        # check_object_for_nan(mask)

        return ~mask

    def forward(self, entities, padding_mask, n_sents):
        """
        padding_mask (BoolTensor) [n_sents * n_ents, batch_size, dim]
            positions with True won't be attended to.
        n_sents (int)
            Used to reshape and enhanced the padding_mask
        """
        # check_object_for_nan(entities)
        # check_object_for_nan(padding_mask)

        # Sanity check n°1
        sents_x_ents, batch_size, dim = entities.shape
        assert dim == self.dim

        # Sanity check n°2
        _sents_x_ents, _batch_size = padding_mask.shape
        assert _batch_size == batch_size and _sents_x_ents == sents_x_ents

        n_ents = sents_x_ents // n_sents

        # Expanding query for all example in the batch and for all sentences
        query = self.query.expand(n_sents, batch_size, -1)

        # check_object_for_nan(query)

        # Formatting the paddding_mask
        if padding_mask.shape == (sents_x_ents, batch_size):
            padding_mask = padding_mask.transpose(0, 1)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, n_sents, -1)

        # Sanity check n°3
        assert padding_mask.shape == (batch_size, n_sents, sents_x_ents)

        # Building mask for the multiple queries, so that each query only
        # attends its assigned entities (0-4, 5-8, 9-12, etc.)
        query_mask = self.make_context_query_mask(n_sents, n_ents, batch_size)

        # check_object_for_nan(query_mask)

        # Merging both masks
        attn_mask = padding_mask | query_mask

        # Repeating for each head of the MultiHeadAttention layer
        attn_mask = tile(attn_mask, self.heads, 0)

        # check_object_for_nan(attn_mask)

        # Maybe compute projected repr of keys and values
        if self._do_proj:
            keys = self.key_linear(entities)
            vals = self.val_linear(entities)
        else:
            keys, vals = entities, entities

        # check_object_for_nan(keys)
        # check_object_for_nan(vals)

        # We are not returning attention scores.
        result = self.attention(query, keys, vals, attn_mask=attn_mask)[0]

        # Not that there could be some NaN in this tensor. This is expected as
        # some sentences have zero grounding entities, which means that the
        # attn_mask will be all -inf, leading to division by zero in softmax.
        # We get those position and simply zero them out.
        # The sanity check here is that the sum of NaNs should be exactly equal
        # to the hidden_dim, else NaNs come from somewhere else.
        nan_mask = (result != result).sum(dim=2)  # Getting rows with NaNs

        # Extensive Sanity check
        unique_vals = sorted(list(nan_mask.unique()))
        if len(unique_vals) == 1 and unique_vals[0] not in {0, dim}:
            raise ContainsNaN('Result of aggregation contains unexpected NaNs')
        if len(unique_vals) == 2 and unique_vals != [0, dim]:
            raise ContainsNaN('Result of aggregation contains unexpected NaNs')
        elif len(unique_vals) > 2:
            raise ContainsNaN('Result of aggregation contains unexpected NaNs')

        result = result.masked_fill(nan_mask.ne(0).unsqueeze(2), 0)
        check_object_for_nan(result)

        return result
