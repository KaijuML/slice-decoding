"""
Everything is has been heavily edited, as elsewhere.

The idea here is that the loss from OpenNMT is pretty much perfect, but does
one or two things that go against what I'm trying to achieve with this code.

 1) I have removed the .backward() from the __call__ method. This is done so
    that all losses (sentence, context, elab) can be computed and summed before
    backward.
 2) I have removed the normalization (dividing by size of batch). This is done
    so that the Trainer handles the normalization, which I find more easy to
    deal with.
"""
from onmt.modules.copy_generator import collapse_copy_scores

import torch
import onmt


def build_loss_computes(model, vocabs, opt):
    """
    Builds my version of the CopyLossCompute from onmt.
    Also builds a context loss, to predict next grounding views.
    """
    
    # 0. Get the device, to which we'll move the loss computes
    device = torch.device("cuda" if opt.use_gpu else "cpu")

    # 1. Build the main loss compute, for sentence decoding
    padding_idx = vocabs['main_vocab']['<pad>']
    unk_idx = vocabs['main_vocab']['<unk>']

    criterion = CopyGeneratorLoss(
        len(vocabs['main_vocab']), opt.copy_attn_force,
        unk_index=unk_idx, ignore_index=padding_idx
    )
    
    sentence_loss = CopyGeneratorLossCompute(
        criterion, model.generator, 
        vocabs['main_vocab'],
        opt.copy_loss_by_seqlength
    )
    
    # 2. Build the context loss, for predicting sentence context
    context_loss = ContextLossCompute(model.context_predictor,
                                      vocabs['elab_vocab'])

    return sentence_loss.to(device), context_loss.to(device)


class ContextLossCompute(torch.nn.Module):
    """
    A context is defined as two entities + an elaboration type.
    There are therefore two steps in predicting context:
        1) Pointing towards primary entities
        2) Selecting elaboration type
    """
    def __init__(self, context_predictor, vocab):
        super().__init__()

        self.end_of_document_index = vocab['<eod>']
        self.elab_pad_idx = vocab['<pad>']
        self.ents_pad_idx = -1  # hard coded

        self.elab_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.elab_pad_idx)
        self.ents_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.ents_pad_idx)
        self.context_predictor = context_predictor

    def forward(self, decoder_outputs, memory_bank,
                sentence_starts, n_primaries,
                target_elaborations, target_contexts):

        # Sanity checks.
        tgt_len, batch_size, dim = decoder_outputs.shape
        n_sents, _batch_size = sentence_starts.shape
        assert batch_size == _batch_size

        # Selecting only the outputs related to starts of sentence
        sentence_starts = sentence_starts.unsqueeze(-1).repeat(1, 1, dim)
        decoder_states = torch.gather(decoder_outputs,
                                      index=sentence_starts,
                                      dim=0)

        elaboration_loss = self._compute_elaboration_loss(target_elaborations,
                                                          decoder_states)

        ents_loss = self._compute_entity_loss(sentence_starts,
                                              target_contexts,
                                              n_primaries,
                                              decoder_states,
                                              memory_bank)
        
        return ents_loss, elaboration_loss

    def _compute_elaboration_loss(self, targets, dec_states):
        scores = self.context_predictor.elab_predictor(self._bottle(dec_states))
        return self.elab_criterion(scores, targets.view(-1))

    def _compute_entity_loss(self, sentence_starts, targets, n_primaries,
                             dec_states, memory_bank):

        # First compute the two queries and split 'em
        queries = self.context_predictor.states_to_queries(dec_states)
        first_query, second_query = queries.chunk(2, dim=2)

        # Next, compute the attention scores over all entities
        candidates = memory_bank['high_level_repr'].permute(1, 2, 0)
        first_scr = torch.bmm(first_query.transpose(0, 1), candidates)[:, :-1]
        second_scr = torch.bmm(second_query.transpose(0, 1), candidates)[:, :-1]

        # ALso, shape the targets as needed and use two masks:
        #     1) Replace elaborations views by zero
        #     2) fill with -1 when outside a sentence
        # Note that targets have no reason to have a specific size, except that
        # they should be at least size 4 on dim=1. In practice, no example has
        # a target with size < 2 so this code runs, but leaving this comment
        # for when a later bug will arise for whatever reason.

        # First mask, remove elaborations from targets
        sent_len, n_elab, batch_size = targets.shape
        elab_mask = n_primaries.view(1, 1, -1).expand(sent_len, n_elab, -1)
        targets = targets.masked_fill(targets > elab_mask, 0)

        # Split contexts to get the first and second
        first_ctx, second_ctx = (targets + 1).split(1, 1)[:2]

        # Then, mask padding (i.e. 0s that are outstide of sentences
        pad_mask = sentence_starts[:-1, :, 0].eq(0)
        pad_mask[0] = False  # first sentence start is always 0 but its not pad
        first_ctx = first_ctx.squeeze(1).masked_fill(pad_mask, -1).transpose(0, 1)
        second_ctx = second_ctx.squeeze(1).masked_fill(pad_mask, -1).transpose(0, 1)

        # finally, compute and average the losses:
        l1 = self.ents_criterion(first_scr.transpose(1, 2), first_ctx)
        l2 = self.ents_criterion(second_scr.transpose(1, 2), second_ctx)

        return (l1 + l2) / 2

    @staticmethod
    def _bottle(tensor):
        return tensor.view(-1, tensor.size(2))
        

class CopyGeneratorLoss(torch.nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super().__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(torch.nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`torch.nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length):
        super().__init__()
        self.criterion = criterion
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _compute_loss(self, batch, output, target, copy_attn, align,
                      std_attn=None, coverage_attn=None):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        
        assert target.dim() == 2
        assert align.dim() == 2
        
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        loss = self.criterion(scores, align, target)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, None)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt.ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats

    def __call__(self,
                 batch,
                 output,
                 attns):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """

        # removing the last token of output and first of tgt is important for
        # aligning predicted tokens and target tokens.
        # Remember that given token <t>, the decoder predicts <p>
        # So <p> should be compared to the token AFTER <t>
        #
        # tgt is '<s> <t1> <t2> ... <tn> </s> <pad> <pad> ...'.
        # pred is '<p1> <p2> ... <pn+1> <pad> ...'
        #
        # We want to align tgt and pred, so that we can compare
        # <t1> against <p1>, <t2> against <p2>, etc.
        #
        # We therefore predict for tgt[:-1] (excluding either prediction
        # for </s> or <pad>, which is irrelevant)
        # given <s>, predicted token is <p1>; given <t1> predicted
        # token is <t2>; etc.
        #
        # The sentence_loss will then compare prediction with tgt[1:]
        # (excluding the <s> token) to make the alignment complete!
        # Note that padding is automatically ignored in the loss.
        
        loss, stats = self._compute_loss(batch=batch,
                                         output=output[1:-1],
                                         target=batch.tgt[1:], 
                                         copy_attn=attns.get("copy")[:-1],
                                         align=batch.alignment[1:])
        return loss, stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
