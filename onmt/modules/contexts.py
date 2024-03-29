from onmt.utils.misc import block_eye, tile, check_object_for_nan

import torch


class ContextPredictor(torch.nn.Module):
    """
    Predicts a context from a decoder state.
    """
    def __init__(self, decoder_hidden_size, elaboration_vocab):
        super().__init__()

        self.elab_predictor = torch.nn.Linear(decoder_hidden_size,
                                              len(elaboration_vocab))

        self.states_to_queries = torch.nn.Linear(decoder_hidden_size,
                                                 2 * decoder_hidden_size)

    def forward(self, decoder_hidden_state):
        raise NotImplementedError("Never call ContextPredictor.forward!")


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

        return ~mask

    def forward(self, entities, n_sents):
        """
        INPUTS:
        -------
        entities (Tensor) [n_sents * n_ents, batch_size, dim]
            The list of grounding entities for each sentence of each example.
            They will be aggregated to return one repr per sentence per ex.

        padding_mask (BoolTensor) [n_sents * n_ents, batch_size, dim]
            positions with True won't be attended to.

        n_sents (int)
            Used to reshape and enhanced the padding_mask

        OUTPUTS:
        --------
        entities (Tensor) [n_sents, batch_size, dim]
        """

        # Sanity check n°1
        sents_x_ents, batch_size, dim = entities.shape
        assert dim == self.dim

        # Sanity check n°2
        assert sents_x_ents % n_sents == 0
        n_ents = sents_x_ents // n_sents

        # Expanding query for all example in the batch and for all sentences
        query = self.query.expand(n_sents, batch_size, -1)

        # Building mask for the multiple queries, so that each query only
        # attends its assigned entities (0-4, 5-8, 9-12, etc.)
        attn_mask = self.make_context_query_mask(n_sents, n_ents, batch_size)

        # Repeating for each head of the MultiHeadAttention layer
        attn_mask = tile(attn_mask, self.heads, 0)

        # Maybe compute projected repr of keys and values
        if self._do_proj:
            keys = self.key_linear(entities)
            vals = self.val_linear(entities)
        else:
            keys, vals = entities, entities

        # We are not returning attention scores.
        return self.attention(query, keys, vals, attn_mask=attn_mask)[0]
