"""Same as normal RNNDecoder but using hierarchical attention"""

from onmt.modules import HierarchicalAttention, Aggregation
from onmt.models.stacked_rnn import StackedLSTM
from onmt.utils.misc import aeq

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


class HierarchicalRNNDecoder(torch.nn.Module):
    """
    Custom RNN decoder. This decoder decodes sentence by sentence.
    For each sentence, the decoder first predicts which part of the data is
    relevant for its attention/copy mechanisms; then decodes all words until '.'

    This decoder builds on top of the Hierarchical Encoder: its attention is
    also hierarchical and attends first to entities, then to their values.

    Since examples don't have the same number of sentences to be decoded,
    that have been padded by None sentences. These are removed to avoid extra
    computation.
    """
    def __init__(self,
                 embeddings=None,
                 dataset_config=None,
                 num_layers=2,
                 dropout=0.0,

                 attn_type="general",
                 attn_func="softmax",
                 copy_attn_type="general",
                 use_cols_in_attention=True,
                 separate_copy_mechanism=False,

                 entity_aggregation_heads=1,
                 entity_aggregation_do_proj=True,
                 elaboration_dim=5):

        self._check_arg_types_and_values(
            embeddings=embeddings,
            dataset_config=dataset_config,
            num_layers=num_layers,
            dropout=dropout,
            attn_type=attn_type,
            attn_func=attn_func,
            copy_attn_type=copy_attn_type,
            use_cols_in_attention=use_cols_in_attention,
            separate_copy_mechanism=separate_copy_mechanism,
            entity_aggregation_heads=entity_aggregation_heads,
            entity_aggregation_do_proj=entity_aggregation_do_proj,
            elaboration_dim=elaboration_dim)

        super().__init__()

        # Gather all useful parameters
        self.entity_size = dataset_config.entity_size
        self.hidden_size = embeddings.embedding_size
        self._separate_copy_mechanism = separate_copy_mechanism
        self.num_layers = num_layers

        # Make sure that self.init_state is called before running
        self._state_is_init = False
        self.state = dict()

        # 0. Configure dropout
        self.dropout = torch.nn.Dropout(dropout)

        # 1. Build embeddings
        self.embeddings = embeddings.value_embeddings

        # 2. Build the LSTM and initialize input feed.
        _input_feed = torch.nn.Parameter(torch.Tensor(1, 1, self.hidden_size))
        self.register_parameter('_input_feed', _input_feed)
        self.rnn = StackedLSTM(self.num_layers,
                               self._input_size,
                               self.hidden_size,
                               dropout)

        # 3. Setup the attention layers.
        units_size = embeddings.embedding_size
        if use_cols_in_attention:
            units_size = embeddings.pos_embeddings.embedding_dim

        # 3.1 Set up the standard attention.
        self.attn = HierarchicalAttention(
            (self.hidden_size, units_size),
            entity_size=self.entity_size,
            attn_type=attn_type, attn_func=attn_func,
            use_pos=use_cols_in_attention)

        # 3.2 Set up a distinct copy mechanism if asked by user
        if self._separate_copy_mechanism:
            self.copy_attn = HierarchicalAttention(
                (self.hidden_size, units_size),
                entity_size=self.entity_size,
                attn_type=copy_attn_type, attn_func=attn_func,
                use_pos=use_cols_in_attention)

        # 4.1 Set up the aggregation layer. It'll be used to aggregate the
        # context entity representations
        self.aggregation = Aggregation(dim=self.hidden_size,
                                       heads=entity_aggregation_heads,
                                       dropout=dropout,
                                       do_proj=entity_aggregation_do_proj)

        # 4.2 Set up the elaboration embedding layer
        n_elaborations = len(dataset_config.elaboration_vocab)
        self.elaboration_embeddings = torch.nn.Embedding(n_elaborations,
                                                         elaboration_dim)

        in_dim, out_dim = self.hidden_size + elaboration_dim, self.hidden_size
        self.merge_entity_and_elaborations = torch.nn.Linear(in_dim, out_dim)

        # Eventually initialize manually registered parameters
        self._init_parameters_manually()

        # Once the decoder is initialized, check for NaNs
        check_object_for_nan(self)

    def _init_parameters_manually(self):
        torch.nn.init.uniform_(self._input_feed, -1, 1)

    def _check_arg_types_and_values(self, embeddings=None, dataset_config=None,
            num_layers=2, dropout=0.0, attn_type="general", attn_func="softmax",
            copy_attn_type="general", use_cols_in_attention=True,
            separate_copy_mechanism=False, entity_aggregation_heads=1,
            entity_aggregation_do_proj=True, elaboration_dim=5):

        assert embeddings is not None
        assert dataset_config is not None
        assert isinstance(num_layers, int) and num_layers > 0
        assert isinstance(dropout, float) and 0 <= dropout < 1
        assert attn_type in {"dot", "general", "mlp"}
        assert attn_func == 'softmax'
        assert copy_attn_type in {"dot", "general", "mlp"}
        assert isinstance(use_cols_in_attention, bool)
        assert isinstance(separate_copy_mechanism, bool)
        assert isinstance(entity_aggregation_heads, int) and entity_aggregation_heads > 0
        assert isinstance(entity_aggregation_do_proj, bool)
        assert isinstance(elaboration_dim, int) and elaboration_dim > 1

    @property
    def device(self):
        return next(self.parameters()).device

    def init_state(self, encoder_final):
        """
        Here we initialize the hidden state of the hierarchical_decoder
        This function only works with the hierarchical_transformer.

        encoder_final is [1, bsz, dim]. We need to:
            - Duplicate it to mimic a multi-layer encoder
            - Convert it to a tuple because decoder.rnn is an LSTM
        """
        # Create container for state objects
        self._state_is_init = True

        hidden = encoder_final.repeat(self.num_layers, 1, 1)
        self.state["hidden"] = (hidden, hidden)

        # Init the input feed with a learnt parameter, repeated for each
        # examples of the batch.
        batch_size = encoder_final.size(1)
        self.state["input_feed"] = self._input_feed.expand(-1, batch_size, -1)

        # Init a useless state, to debug tracking states
        self.state['tracking'] = torch.zeros(1, batch_size, 1, device=self.device)

        # check_object_for_nan(self.state)

    def set_state(self, state):
        self.state = state

    def map_state(self, func):
        """
        Applies function to all states.
        func is always torch.index_select or tile for now.
        For this reason, it's important that dim=1 be batch dim.
        """
        def fn(state):
            if isinstance(state, tuple):
                return tuple(func(s, 1) for s in state)
            return func(state, 1)

        self.state = {
            key: fn(state) for key, state in self.state.items()
        }

        # check_object_for_nan(self.state)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    @classmethod
    def from_opt(cls, opt, embeddings, dataset_config):
        """
        embeddings should be an instance of TableEmbeddings.
        """

        return cls(
            embeddings=embeddings,
            dataset_config=dataset_config,
            num_layers=opt.decoder_layers,
            attn_type=opt.global_attention,
            attn_func=opt.global_attention_function,
            dropout=opt.dropout,
            separate_copy_mechanism=opt.separate_copy_mechanism,
            copy_attn_type=opt.copy_attn_type,
            use_cols_in_attention=opt.use_cols_in_attention,
            entity_aggregation_heads=opt.entity_aggregation_heads,
            entity_aggregation_do_proj=opt.entity_aggregation_do_proj,
            elaboration_dim=opt.elaboration_dim)

    def forward(self, sentences=None, memory_bank=None,
                      contexts=None, elaborations=None,
                      action=None):
        """
        Action should be from [decode_full, decode_once, predict_context]
        Set to None to force explicit choice.
        """

        if not self._state_is_init:
            raise RuntimeError('You MUST call self.init_state before decoding.')

        if action in ['decode_full']:

            if sentences is None or memory_bank is None or contexts is None:
                raise RuntimeError('sentences, contexts and memory bank must be'
                                   ' given when decoding sentences.')

            dec_state, dec_outs, attns = self._run_forward_pass(sentences,
                                                                contexts,
                                                                memory_bank)

            # check_object_for_nan(self)
            # check_object_for_nan(dec_state)
            # check_object_for_nan(dec_outs)
            # check_object_for_nan(attns)

            # dec_outs is list of [batch_size, hidden_dim]
            # We stack to get [tgt_lengths+1, batch_size, hidden_dim]
            # Note that we get length + 1 because we also get init states.
            dec_outs = torch.stack(dec_outs)

            self.state["hidden"] = dec_state
            self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
            self.state['tracking'] += 1

            # check_object_for_nan(self.state)

            return dec_outs, attns

        elif action == 'compute_context_representation':

            # This action has two steps:
            #   1) Aggregating the grounding entities
            #   2) Embedding elaboration in latent space
            # Both the aggregation and emb are cat + mlp to self.hidden_size

            # check_object_for_nan(memory_bank)

            if memory_bank is None:  # or elaborations is None:
                raise RuntimeError('memory bank & elaborations must be given '
                                   'to compute context representations!')
            if contexts is None:
                err = 'contexts must be given to compute context repr...'
                raise RuntimeError(err)

            # 1.1 Shaping the mask for attention in the aggregation step

            # Simply add the sentence dimension. Many ways to do this.
            if contexts.dim() == 2:
                contexts.unsqueeze(0)

            n_sents, n_ents, batch_size = contexts.shape
            contexts = contexts.view(n_sents * n_ents, batch_size)

            # check_object_for_nan(contexts)

            # Compute the padding mask, using -1 as padding index
            padding_mask = contexts == -1

            # torch.gather do not support negative index, so we replace them by
            # 0. It's not a problem, given that we still retain their locations
            # and will ignore them later on.
            contexts = contexts.masked_fill(padding_mask, 0)

            # check_object_for_nan(contexts)

            # Create the index for torch.gather
            index = contexts.unsqueeze(2).expand(-1, -1, self.aggregation.dim)

            # 1.2 Gather & Aggregate grounding entities
            entities = memory_bank['high_level_repr']
            entities = entities.gather(dim=0, index=index)
            entities = self.aggregation(entities, padding_mask, n_sents)

            # check_object_for_nan(entities)

            # Reshaping everything to its correct shape
            entities = entities.view(n_sents, batch_size, self.aggregation.dim)

            # 2.1 Embedding elaborations
            # We skip the last one, because it is <eod>
            elaborations = self.elaboration_embeddings(elaborations[:-1])

            # check_object_for_nan(elaborations)

            result = torch.cat([entities, elaborations], dim=2)
            result = self.merge_entity_and_elaborations(result)

            # check_object_for_nan(result)

            return result

        else:
            raise RuntimeError(f'Unknown decoder action: {action}')

    def _run_forward_pass(self, tgt, ctx, memory_bank):
        """
        memory_bank is a tuple (chunks, units, pos_embs, unit_mask, chunk_mask)
        """
        # Additional args check.
        input_feed_batch = self.state["input_feed"].size(1)
        tgt_len, tgt_batch = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # It's important to keep track of all outputs/states. This way,
        # when decoding is done for all sentences, we can extract relevant
        # state at each change of sentences, to try and predict next slices.
        decoder_outputs = [self.state["input_feed"].squeeze(0)]
        decoder_states = self.state["hidden"]

        attns = dict()

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        # Sanity check that we have the same number of embeddings and context
        # and they have same batch_size and repr_dim
        assert ctx.shape == emb.shape

        # Input feed concatenates hidden state with input at every time step.
        for emb_t, ctx_t in zip(emb.split(1, dim=0), ctx.split(1, dim=0)):

            dec_in = [emb_t.squeeze(0), ctx_t.squeeze(0), decoder_outputs[-1]]
            rnn_output, dec_state = self.rnn(torch.cat(dec_in, 1), decoder_states)

            # If the RNN has several layers, we only use the last one to compute
            # the attention scores. In pytorch, the outs of the rnn are:
            #     - rnn_output [seq_len, bsz, n-directions * hidden_size]
            #     - dec_state [n-layers * n-directions, bsz, hidden_size] * 2
            # We unpack the rnn_output on dim 2 and keep the last layer

            decoder_output, ret = self.attn(rnn_output, memory_bank)
            for postfix, tensor in ret.items():
                key = 'std' + postfix
                attns.setdefault(key, list())
                attns[key].append(tensor)

            decoder_output = self.dropout(decoder_output)

            decoder_outputs.append(decoder_output)
            decoder_states = dec_state

            if self._separate_copy_mechanism:
                _, copy_attn = self.copy_attn(decoder_output, memory_bank)
                for postfix, tensor in copy_attn.items():
                    key = 'copy' + postfix
                    attns.setdefault(key, list())
                    attns[key].append(tensor)

        # this trick should save memory because torch.stack creates a new
        # object. Here we use torch.stack before duplicating the attn keys,
        # to ensure create the object once.
        for key in list(attns):
            if key.startswith('std'):
                attns[key] = torch.stack(attns[key])
                if not self._separate_copy_mechanism:
                    attns[key.replace('std', 'copy')] = attns[key]

        return decoder_states, decoder_outputs, attns

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return 3 * self.hidden_size
