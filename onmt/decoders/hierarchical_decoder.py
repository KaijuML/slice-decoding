"""Same as normal RNNDecoder but using hierarchical attention"""

from onmt.models.stacked_rnn import StackedLSTM
from onmt.modules import HierarchicalAttention
from onmt.utils.misc import aeq

import torch


class ContainsNaN(Exception):
    pass


def _check_for_nan(tensor: torch.Tensor):
    if (tensor != tensor).any():
        raise ContainsNaN


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
    def __init__(self, embeddings, dataset_config, num_layers,
                 attn_type="general", attn_func="softmax", dropout=0.0,
                 separate_copy_mechanism=False, copy_attn_type="general",
                 use_cols_in_attention=True):

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
        self.register_parameter('_start_input_feed',
            torch.nn.Parameter(torch.Tensor(1, 1, self.hidden_size)))
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
        self.state["input_feed"] = self._start_input_feed.repeat(1, batch_size, 1)

        # Init a useless state, to debug tracking states
        self.state['tracking'] = torch.zeros(1, batch_size, 1)

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
            use_cols_in_attention=opt.use_cols_in_attention)

    def forward(self, sentences=None, memory_bank=None, action=None):
        """
        Action should be from [decode_full, decode_once, predict_context]
        Set to None to force explicit choice.
        """

        if not self._state_is_init:
            raise RuntimeError('You MUST call self.init_state before decoding.')

        if action in ['decode_full']:

            if sentences is None or memory_bank is None:
                err = 'sentences & memory bank must be given in decode_full'
                raise RuntimeError(err)

            dec_state, dec_outs, attns = self._run_forward_pass(sentences,
                                                                memory_bank)

            # dec_outs is list of [batch_size, hidden_dim]
            # We stack to get [tgt_lengths+1, batch_size, hidden_dim]
            # Note that we get length + 1 because we also get init states.
            dec_outs = torch.stack(dec_outs)

            self.state["hidden"] = dec_state
            self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
            self.state['tracking'] += 1

            return dec_outs, attns

        else:
            raise RuntimeError(f'Unknown decoder action: {action}')

    def _run_forward_pass(self, tgt, memory_bank):
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

        # Input feed concatenates hidden state with input at every time step.
        for emb_t in emb.split(1):

            # We get the last output / state

            dec_in = torch.cat([emb_t.squeeze(0), decoder_outputs[-1]], 1)
            rnn_output, dec_state = self.rnn(dec_in, decoder_states)

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
        return self.embeddings.embedding_dim + self.hidden_size
