from onmt.inference.decode_strategy import DecodeStrategy
from onmt.inference import penalties
from onmt.utils.misc import tile

import warnings
import torch


class BeamSearch(DecodeStrategy):
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        vocab (torchtext.Vocab): see base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B x beam_size,)``. These
            are the scores used for the topk operation.
        memory_lengths (LongTensor): Lengths of encodings. Used for
            masking attentions.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__(self, vocab, batch_size, beam_size, min_length, max_length,
                 block_ngram_repeat, exclusion_tokens, init_token='<s>',
                 global_scorer=None, n_best=1, ratio=0):

        super().__init__(
            vocab=vocab,
            batch_size=batch_size,
            parallel_paths=beam_size,
            min_length=min_length,
            max_length=max_length,
            block_ngram_repeat=block_ngram_repeat,
            exclusion_tokens=exclusion_tokens,
            init_token=init_token,
            return_attention=True,
        )

        # beam parameters
        self.global_scorer = global_scorer
        self.n_best = n_best
        self.ratio = ratio

        assert n_best == 1, "our current version only supports n_best==1"

        # result caching
        self.hypotheses = [list() for _ in range(batch_size)]

        # beam state shenanigans
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.bool)
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)

        self.best_scores = torch.full([self.batch_size], -1e10, dtype=torch.float)
        self._beam_offset = torch.arange(
            0, self.n_paths, step=self.beam_size, dtype=torch.long)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (self.beam_size - 1)
        ).repeat(self.batch_size)

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty(self.shape, dtype=torch.float)
        self.topk_ids = torch.empty(self.shape, dtype=torch.long)
        self._batch_index = torch.empty(self.shape, dtype=torch.long)

        self.memory_lengths = None

    @property
    def beam_size(self):
        return self.parallel_paths

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None):
        """
        Repeat src objects `beam_size` times.
        """

        def fn_map_state(state, dim):
            return tile(state, self.beam_size, dim=dim)

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, self.beam_size, dim=1)
                                for x in memory_bank)
            mb_device = memory_bank[0].device
        elif isinstance(memory_bank, dict):
            memory_bank = {name: tile(tensor, self.beam_size, dim=1)
                           for name, tensor in memory_bank.items()}
            _tmp_key = next(iter(memory_bank))  # raises if empty dict
            mb_device = memory_bank[_tmp_key].device
        else:
            memory_bank = tile(memory_bank, self.beam_size, dim=1)
            mb_device = memory_bank.device

        if src_map is not None:
            src_map = tile(src_map, self.beam_size, dim=1)

        if device is None:
            device = mb_device

        super().initialize(device=device)

        self.best_scores = self.best_scores.to(device)
        self._beam_offset = self._beam_offset.to(device)
        self.topk_log_probs = self.topk_log_probs.to(device)

        self.memory_lengths = tile(src_lengths, self.beam_size)

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = self.topk_scores.to(device)
        self.topk_ids = self.topk_ids.to(device)
        self._batch_index = self._batch_index.to(device)

        return fn_map_state, memory_bank, self.memory_lengths, src_map

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    @property
    def batch_offset(self):
        return self._batch_offset

    def _pick(self, log_probs):
        """Return token decision for a step.

        Args:
            log_probs (FloatTensor): (B, vocab_size)

        Returns:
            topk_scores (FloatTensor): (B, beam_size)
            topk_ids (LongTensor): (B, beam_size)
        """
        vocab_size = log_probs.size(-1)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs.reshape(-1, self.beam_size * vocab_size)
        topk_scores, topk_ids = torch.topk(curr_scores, self.beam_size, dim=-1)
        return topk_scores, topk_ids

    def advance(self, log_probs, attns, states):

        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        # _B is the current batch size (which gets smaller as beams end)
        _B = log_probs.shape[0] // self.beam_size

        # force the output to be longer than self.min_length
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        # noinspection PyCallingNonCallable
        length_penalty = self.global_scorer.length_penalty(
            len(self) + 1, alpha=self.global_scorer.alpha)

        curr_scores = log_probs / length_penalty

        # Avoid any direction that would repeat unwanted ngrams
        self.block_ngram_repeats(curr_scores)

        # Pick up candidate token by curr_scores
        self.topk_scores, self.topk_ids = self._pick(curr_scores)

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        self.topk_log_probs = torch.mul(self.topk_scores, length_penalty)

        # Resolve beam origin and map to batch index flat representation.
        self._batch_index = self.topk_ids // vocab_size
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)
        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)

        self.maybe_update_forbidden_tokens()

        # Now we update the tracking of attention and decoder states

        def index_select(state, dim=1):
            if isinstance(state, tuple):
                return tuple(index_select(s, dim=dim) for s in state)
            return state.index_select(dim, self.select_indices)

        current_attn = {
            name: index_select(attn, dim=1)
            for name, attn in attns.items()
        }

        current_states = {
            'hidden': tuple(index_select(state.unsqueeze(0), dim=2)
                            for state in states['hidden'])
        }
        current_states.update({
            name: index_select(state, dim=1)
            for name, state in states.items()
            if name != 'hidden'
        })

        # When len(self) == 2, meaning we are at the first token after <s>
        # We initialize self.alive_state|attn (before they were None)
        if len(self) - 1 == 1:
            self.alive_attn = current_attn
            self.alive_states = current_states

        # Otherwise, we simply cat on dim 0 (i.e. sequence length dim)
        else:
            # Reordering previous tracked items
            self.alive_attn = {
                name: index_select(attn)
                for name, attn in self.alive_attn.items()
            }
            self.alive_states = {
                name: index_select(state, dim=int(name == 'hidden') + 1)
                for name, state in self.alive_states.items()
            }

            # torch.cat (on dim=0) previous with current
            def cat(prev, current):
                if isinstance(prev, tuple):
                    assert isinstance(current, tuple)
                    return tuple(cat(p, c) for p, c in zip(prev, current))
                return torch.cat([prev, current], 0)

            self.alive_attn = {
                name: cat(attn, current_attn[name])
                for name, attn in self.alive_attn.items()
            }
            self.alive_states = {
                name: cat(state, current_states[name])
                for name, state in self.alive_states.items()
            }

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)

        def reshape(state, extra_dim=False):
            if isinstance(state, tuple):
                return tuple(reshape(s, extra_dim=extra_dim) for s in state)
            shape = [step - 1, _B_old, self.beam_size, state.size(-1)]
            if extra_dim:
                shape.insert(1, -1)
            return state.view(*shape)

        attention = {
            name: reshape(attn)
            for name, attn in self.alive_attn.items()
        }
        states = {
            name: reshape(state, extra_dim=name == 'hidden')
            for name, state in self.alive_states.items()
        }

        non_finished_batch = list()
        for i in range(self.is_finished.size(0)):  # Batch level
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero(as_tuple=False).view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:  # Beam level: finished beam j in batch i
                if self.ratio > 0:
                    s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s

                def grab_attn(attn, i, j):
                    return attn[:, i, j, :self.memory_lengths[i]]

                def grab_state(state, i, j, is_hidden=False):
                    if isinstance(state, tuple):
                        return tuple(grab_state(s, i, j, is_hidden) for s in state)
                    grabs = [
                        slice(None, None, None),
                        i, j, slice(None, None, None)
                    ]
                    if is_hidden:
                        grabs.insert(1, slice(None, None, None))

                    return state[tuple(grabs)]

                self.hypotheses[b].append([
                    self.topk_scores[i, j],
                    predictions[i, j, 1:],  # Ignore start_token.
                    {
                        name: grab_attn(attn, i, j)
                        for name, attn in attention.items()
                    },
                    {
                        name: grab_state(state, i, j, name == 'hidden')
                        for name, state in states.items()
                    },
                ])
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self.memory_lengths[i] * self.ratio
                finish_flag = ((self.topk_scores[i, 0] / pred_len)
                               <= self.best_scores[b]) or \
                    self.is_finished[i].all()
            else:
                finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (_score, _pred, _attn, _states) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(_score)
                    self.predictions[b].append(_pred)  # ``(batch, n_best,)``
                    self.attention[b].append(_attn)
                    self.states[b].append(_states)
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)

        # Reorder attentions and states for other non finished examples.
        # Unfortunately, no elegant way to deal with hidden decoder state.
        def index_select(state, is_hidden=False):
            if isinstance(state, tuple):
                return tuple(index_select(s, is_hidden=is_hidden) for s in state)

            state = state.index_select(int(is_hidden)+1, non_finished)
            shape = [step - 1, _B_new * self.beam_size, state.size(-1)]
            if is_hidden:
                shape.insert(1, -1)
            return state.view(*shape)

        self.alive_attn = {
            name: index_select(attn)
            for name, attn in attention.items()
        }
        self.alive_states = {
            name: index_select(state, is_hidden=name == 'hidden')
            for name, state in states.items()
        }


class GNMTGlobalScorer(object):
    """NMT re-ranking.

    Args:
       alpha (float): Length parameter.
       beta (float):  Coverage parameter.
       length_penalty (str): Length penalty strategy.
       coverage_penalty (str): Coverage penalty strategy.

    Attributes:
        alpha (float): See above.
        beta (float): See above.
        length_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        coverage_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        has_cov_pen (bool): See :class:`penalties.PenaltyBuilder`.
        has_len_pen (bool): See :class:`penalties.PenaltyBuilder`.
    """

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.alpha,
            opt.beta,
            opt.length_penalty,
            opt.coverage_penalty)

    def __init__(self, alpha, beta, length_penalty, coverage_penalty):
        self._validate(alpha, beta, length_penalty, coverage_penalty)
        self.alpha = alpha
        self.beta = beta
        penalty_builder = penalties.PenaltyBuilder(coverage_penalty,
                                                   length_penalty)
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is None or length_penalty == "none":
            if alpha != 0:
                warnings.warn("Non-default `alpha` with no length penalty. "
                              "`alpha` has no effect.")
        else:
            # using some length penalty
            if length_penalty == "wu" and alpha == 0.:
                warnings.warn("Using length penalty Wu with alpha==0 "
                              "is equivalent to using length penalty none.")
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                warnings.warn("Non-default `beta` with no coverage penalty. "
                              "`beta` has no effect.")
        else:
            # using some coverage penalty
            if beta == 0.:
                warnings.warn("Non-default coverage penalty with beta==0 "
                              "is equivalent to using coverage penalty none.")
