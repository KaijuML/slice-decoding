from onmt.modules.copy_generator import collapse_copy_scores
from onmt.inference import BeamSearch, GNMTGlobalScorer
from onmt.utils.misc import Container, sequence_mask
from onmt.rotowire.dataset import numericalize
from onmt.model_builder import load_test_model
from onmt.rotowire.utils import MultiOpen

from onmt.rotowire import (
    RotowireGuidedInferenceDataset,
    RotowireInferenceDataset,
    build_dataset_iter
)

import torch
import tqdm
import os


PREP_SENTENCE_GENERATION_N_OUPTUS = 6


def build_inference(args, logger=None):
    vocabs, model, model_opts = load_test_model(args.model_path, args.gpu)

    inference_cls = GuidedInference if args.guided_inference else Inference
    return inference_cls.from_opts(
        model,
        vocabs,
        args,
        logger=logger
    )


class BaseInference:
    def __init__(self,
                 model,
                 vocabs,

                 seed,

                 beam_size,
                 min_sent_length,
                 max_sent_length,

                 block_ngram_repeat,
                 ignore_when_blocking,

                 desc_dest,
                 plan_dest,
                 logger):

        self.model = model
        self.vocabs = vocabs

        self.seed = seed

        self.beam_size = beam_size
        self.min_sent_length = min_sent_length
        self.max_sent_length = max_sent_length

        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking

        self.desc_dest = desc_dest
        self.plan_dest = plan_dest
        self.logger = logger

        # This should be specified by children of this base class
        self.is_guided_inference = None

    @property
    def device(self):
        return self.model.device

    @property
    def entity_size(self):
        return self.model.config.entity_size

    @classmethod
    def from_opts(cls, model, vocabs, opts, logger):
        return cls(
            model=model,
            vocabs=vocabs,

            seed=opts.seed,

            beam_size=opts.beam_size,
            min_sent_length=opts.min_sent_length,
            max_sent_length=opts.max_sent_length,

            block_ngram_repeat=opts.block_ngram_repeat,
            ignore_when_blocking=set(opts.ignore_when_blocking),

            desc_dest=opts.desc_dest,
            plan_dest=opts.plan_dest,
            logger=logger,
        )

    def run(self, filename, batch_size, if_file_exists='raise'):

        if if_file_exists not in {'raise', 'overwrite', 'append'}:
            raise ValueError(f'Unknown instruction {if_file_exists=}')

        if os.path.exists(self.dest):
            if if_file_exists == 'raise':
                raise RuntimeError(f'{self.dest} already exists!')
            elif if_file_exists == 'overwrite':
                self.logger.info(f'Overwrite destination file: {self.dest}')
                with open(self.dest, mode="w", encoding='utf8') as f:
                    pass  # overwrites
            else:
                self.logger.info('Appending new generations to existing file: '
                                 f'{self.dest}')

        dataset_cls = RotowireInferenceDataset
        if self.is_guided_inference:
            dataset_cls = RotowireGuidedInferenceDataset

        dataset = dataset_cls.build_from_raw_json(
            filename, config=self.model.config, vocabs=self.vocabs)

        opt = Container(batch_size=batch_size, num_threads=1)
        inference_iter = build_dataset_iter(dataset, opt, self.device)

        for batch in tqdm.tqdm(inference_iter, desc="Running Inference"):
            batch_predicted_descriptions = self.run_on_batch(batch)

            kwargs = {"mode": "a", "encoding": "utf8"}
            with MultiOpen(self.desc_dest, self.plan_dest, **kwargs) as files:
                desc_file, plan_file = files
                for desc, plan in zip(*batch_predicted_descriptions):
                    assert len(desc) == len(plan)

                    for sentence, (entities, elab) in zip(desc, plan):
                        desc_file.write(sentence.strip('<s> ').replace('_', ' ') + '\n')
                        plan_file.write(f'{entities} {elab}\n')

                    desc_file.write('\n')
                    plan_file.write('\n')

    def run_on_batch(self, batch):

        # Used in warnings
        warn_indices = batch.indices.clone()

        # Count number of planned sentences
        n_sentences = self.count_nb_planned_sentences(batch)

        # prepare final outputs
        all_sentences = [list() for _ in range(batch.batch_size)]
        all_plans = [list() for _ in range(batch.batch_size)]

        # tracking predictions that are still not done
        current_sentences, current_plans = all_sentences, all_plans

        with torch.no_grad():
            # 1. Run the encoder on the src.
            true_memory_bank = self.model.encoder(*batch.src, batch.n_primaries)
            true_lengths = batch.src[1]

        # Init previous states to facilitate code
        prev_states = None

        # Generate sentences one by one using beam search
        for sent_idx in range(n_sentences):

            packed_preped_sentence_gen = self.prep_sentence_generation(
                batch, sent_idx, current_sentences, current_plans,
                prev_states, true_lengths, true_memory_bank)

            current_sentences, current_plans = packed_preped_sentence_gen[:2]
            true_memory_bank, true_lengths = packed_preped_sentence_gen[2:4]
            memory_bank, src_lengths = packed_preped_sentence_gen[4:6]

            if current_sentences is None:
                assert memory_bank is None
                assert src_lengths is None
                if sent_idx != n_sentences - 1 and self.is_guided_inference:
                    self.logger.warn('Stopping generation earlier than '
                                     f'expected, for batch={warn_indices}')
                break

            # Keep track of (maybe) predicted entities and elaborations
            predicted_ctx = memory_bank['contexts'][0, :, :2].tolist()
            predicted_elab = memory_bank['elaborations'][0].tolist()

            assert len(predicted_elab) == len(predicted_ctx) == len(current_plans)
            for pidx in range(len(current_plans)):
                current_plans[pidx].append([predicted_ctx[pidx], predicted_elab[pidx]])

            # We will decode each sentence using beam search
            scorer_opt = Container(
                alpha=0,
                beta=0,
                length_penalty='none',
                coverage_penalty='none'
            )
            decode_strategy = BeamSearch(
                vocab=self.vocabs['main_vocab'],
                batch_size=batch.batch_size,
                beam_size=self.beam_size,
                min_length=self.min_sent_length,
                max_length=self.max_sent_length,
                block_ngram_repeat=self.block_ngram_repeat,
                previous_tokens=[' '.join(sents) for sents in current_sentences],
                exclusion_tokens=self.ignore_when_blocking,
                init_token='<s>' if not sent_idx else '</s>',
                global_scorer=GNMTGlobalScorer.from_opt(scorer_opt),
                n_best=1,
                ratio=0
            )

            self.predict_one_sentence(decode_strategy, batch,
                                      memory_bank, src_lengths)

            decoded_sentences = self.build_translated_sentences(decode_strategy,
                                                                batch)

            for sents, new_sent in zip(current_sentences, decoded_sentences):
                sents.append(new_sent)

            prev_states = self.reset_states(decode_strategy)

        return all_sentences, all_plans

    def count_nb_planned_sentences(self, batch):
        raise NotImplementedError()

    def prep_sentence_generation(self, batch, sent_idx,
                                 current_sentences, current_plans,
                                 prev_states, true_lengths, true_memory_bank):
        raise NotImplementedError()

    def predict_one_sentence(self, decode_strategy, batch,
                             memory_bank, src_lengths):

        with torch.no_grad():
            # 2. prep decode_strategy. Repeat src objects for each beam.
            fn_map_state, memory_bank, memory_lengths, src_map = \
                decode_strategy.initialize(memory_bank, src_lengths, batch.src_map)

            # Also repeat the decoder state
            if fn_map_state is not None:
                self.model.decoder.map_state(fn_map_state)

            # 3. Compute context for the sentence at hand
            context_repr, contexts = self.model.decoder(
                action="compute_context_representation",
                memory_bank=memory_bank,
                contexts=memory_bank.pop('contexts').transpose(1, 2).contiguous(),
                elaborations=memory_bank.pop('elaborations')
            )

            # Repack contexts and context_repr into memory_bank
            memory_bank['context_repr'] = context_repr
            memory_bank['contexts'] = contexts.transpose(1, 2).contiguous()

            src_vocabs = None  # Using a different src_vocab for each example

            # 4. Begin decoding step by step:
            for step in range(decode_strategy.max_length):
                decoder_input = decode_strategy.current_predictions.view(1, -1)

                log_probs, attn = self._decode_and_generate(
                    self.model,
                    decoder_input,
                    memory_bank,
                    batch=batch,
                    src_map=src_map,
                    batch_offset=decode_strategy.batch_offset)

                decode_strategy.advance(log_probs, attn, self.model.decoder.state)
                any_finished = decode_strategy.is_finished.any()
                if any_finished:

                    decode_strategy.update_finished()
                    if decode_strategy.done:
                        break

                select_indices = decode_strategy.select_indices

                if any_finished:
                    # Reorder states.
                    if isinstance(memory_bank, tuple):
                        memory_bank = tuple(x.index_select(1, select_indices)
                                            for x in memory_bank)
                    elif isinstance(memory_bank, dict):
                        memory_bank = {name: tensor.index_select(1, select_indices)
                                       for name, tensor in memory_bank.items()}
                    else:
                        memory_bank = memory_bank.index_select(1, select_indices)

                    memory_lengths = memory_lengths.index_select(0, select_indices)

                    if src_map is not None:
                        src_map = src_map.index_select(1, select_indices)

                if decode_strategy.parallel_paths > 1 or any_finished:
                    self.model.decoder.map_state(
                        lambda state, dim: state.index_select(dim, select_indices))

    def _decode_and_generate(
            self,
            model,
            decoder_in,
            memory_bank,
            batch,
            src_map=None,
            batch_offset=None):

        # Turn any copied words into UNKs.
        decoder_in = decoder_in.masked_fill(
            decoder_in.gt(len(self.vocabs['main_vocab']) - 1), 0
        )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = model.decoder(action='decode_full',
                                          sentences=decoder_in,
                                          context_repr=memory_bank['context_repr'],
                                          contexts=memory_bank['contexts'].transpose(1, 2).contiguous(),
                                          memory_bank=memory_bank)

        attn = dec_attn["copy"]
        scores = model.generator(dec_out[1],
                                 attn[0],
                                 src_map)
        # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
        if batch_offset is None:
            scores = scores.view(-1, batch.batch_size, scores.size(-1))
            scores = scores.transpose(0, 1).contiguous()
        else:
            scores = scores.view(-1, self.beam_size, scores.size(-1))
        scores = collapse_copy_scores(
            scores,
            batch,
            self.vocabs['main_vocab'],
            src_vocabs=None,
            batch_dim=0,
            batch_offset=batch_offset
        )
        scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
        log_probs = scores.squeeze(0).log()

        return log_probs, dec_attn

    @staticmethod
    def reset_states(decode_strategy):
        states = {
            'hidden': [list(), list()],
            'input_feed': list(),
            'primary_mask': list(),
            'tracking': list()
        }

        for beam in decode_strategy.states:
            states['hidden'][0].append(beam[0]['hidden'][0][-1])
            states['hidden'][1].append(beam[0]['hidden'][1][-1])
            states['input_feed'].append(beam[0]['input_feed'][-1])
            states['primary_mask'].append(beam[0]['primary_mask'][-1])
            states['tracking'].append(beam[0]['tracking'][-1])

        return {
            'hidden': tuple([
                torch.stack(states['hidden'][0], dim=1),
                torch.stack(states['hidden'][1], dim=1),
            ]),
            'input_feed': torch.stack(states['input_feed']).unsqueeze(0),
            'primary_mask': torch.stack(states['primary_mask']).unsqueeze(0),
            'tracking': torch.stack(states['tracking']).unsqueeze(0),
        }

    def _build_target_tokens(self, src_length, src_vocab, src_map, pred, attn):
        vocab = self.vocabs['main_vocab']
        tokens = list()

        for idx, tok in enumerate(pred):

            if tok < len(vocab):
                lexicalized_tok = vocab.itos[tok]
            else:
                lexicalized_tok = src_vocab.itos[tok - len(vocab)]

            if lexicalized_tok == '<unk>':
                _, max_index = attn['copy'][idx][:src_length].max(0)
                tok = src_map[max_index].nonzero().item()
                lexicalized_tok = src_vocab.itos[tok]
            elif lexicalized_tok == '</s>':
                break

            tokens.append(lexicalized_tok)

        return tokens

    def build_translated_sentences(self,
                                   decode_strategy,
                                   batch):
        results = dict()
        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention

        # Reordering beams using the sorted indices
        preds, pred_score, attn, indices = list(zip(
            *sorted(zip(results["predictions"],
                        results["scores"],
                        results["attention"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        translations = list()
        for b in range(batch.batch_size):
            src_lengths = batch.src[1][b]
            src_vocab = batch.src_ex_vocab[b]
            src_map = batch.src_map[:, b]

            pred_sents = [self._build_target_tokens(
                src_lengths,
                src_vocab,
                src_map,
                preds[b][n],
                attn[b][n])
                for n in range(1)]

            translations.append(' '.join(pred_sents[0]))
        return translations


class Inference(BaseInference):

    def __init__(self, *args, **kwargs):
        super(Inference, self).__init__(*args, **kwargs)
        self.is_guided_inference = False

    def count_nb_planned_sentences(self, batch):
        return 25

    def add_elaboration_to_batch(self, batch, batch_idx, ent, elab_str,
                                 elaborations_query_mapping,
                                 reverse_elaboration_mapping):

        main_voc, cols_voc = self.vocabs['main_vocab'], self.vocabs['cols_vocab']

        # Unpack original source
        src, src_lengths = batch.src

        assert ent.item() - 1 in elaborations_query_mapping, batch_idx
        vals = elaborations_query_mapping[ent.item() - 1][elab_str][0]
        cols = elaborations_query_mapping[ent.item() - 1][elab_str][1]

        assert len(vals) == len(cols) == self.entity_size

        # Adding litteral values to src_ex_vocab
        vocab = batch.src_ex_vocab[batch_idx]
        for val in vals:
            vocab.freqs[val] += 1
            if val not in vocab.stoi:
                vocab.itos.append(val)
                vocab.stoi[val] = len(vocab.itos) - 1

        kwargs = {'dtype': torch.long, "device": self.device}

        # Padding src and src_map
        if (pad := len(vocab) - batch.src_map[:, batch_idx].size(1)) > 0:
            dims = list(batch.src_map.shape); dims[2] = pad
            padding = torch.zeros(*dims, **kwargs)
            batch.src_map = torch.cat([batch.src_map, padding], dim=2)

        if (pad := src_lengths[batch_idx].item() + len(vals) - src_lengths.max()) > 0:
            dims = [pad, batch.src[0].size(1), 1]
            padding = torch.cat([
                torch.full(size=dims, fill_value=self.vocabs['main_vocab']['<pad>'], **kwargs),
                torch.full(size=dims, fill_value=self.vocabs['cols_vocab']['<pad>'], **kwargs),
            ], dim=2)
            src = torch.cat([src, padding], dim=0)

            dims = list(batch.src_map.shape); dims[0] = pad
            padding = torch.zeros(*dims, **kwargs)
            batch.src_map = torch.cat([batch.src_map, padding], dim=0)

        start = src_lengths[batch_idx].item()

        # adding values to src_map for the copy mechanism
        for j, tok in enumerate(vals, start):
            batch.src_map[j, batch_idx, vocab[tok]] = 1

        # adding values to src
        _src = [numericalize(seq, voc) for seq, voc in zip([vals, cols],
                                                           [main_voc, cols_voc])]
        _src = torch.tensor(_src, **kwargs).transpose(0, 1)
        src[start:start + len(vals), batch_idx] = _src

        # incrementing lengths
        src_lengths[batch_idx] += _src.size(0)

        # pack modified source
        batch.src = src, src_lengths

        # Add this elaboration to the reverse mapping
        ridx = (src_lengths[batch_idx] // self.entity_size) - 1
        reverse_elaboration_mapping[ent, elab_str] = ridx

    def prep_sentence_generation(self, batch, sent_idx,
                                 current_sentences, current_plans,
                                 prev_states, true_lengths, true_memory_bank):
        """
        Here, we need to predict the grounding entities, as well as the
        elaboration type. Further, if elaboration type is <eod> we prune
        the terminated descriptions.
        """

        # Initialize decoder with game_repr or states at the end of prev sent
        if not sent_idx:
            self.model.decoder.init_state(true_memory_bank['game_repr'],
                                          true_memory_bank['primary_mask'])
        else:
            self.model.decoder.set_state(prev_states)

        # Predict elaborations based on last decoder states
        dec_states = self.model.decoder.state['input_feed'].squeeze(0)
        with torch.no_grad():
            elab_logits = self.model.context_predictor.elab_predictor(dec_states)
            elaborations = elab_logits.topk(1, dim=-1).indices.squeeze(1)

        # Create list of examples that are not done. We know an example is done
        # when its elaboration is 2 (<eod>) or 1 (<pad>).
        valid_examples = elaborations.ge(3).nonzero().squeeze(1)
        if (batch_size := valid_examples.size(0)) == 0:
            return [None] * PREP_SENTENCE_GENERATION_N_OUPTUS

        # Keep track of current sentences & plans
        current_sentences = [current_sentences[idx] for idx in valid_examples]
        current_plans = [current_plans[idx] for idx in valid_examples]

        # Filter batch examples
        batch.index_select(valid_examples)

        # Filter elaborations
        elaborations = elaborations.index_select(dim=0, index=valid_examples)

        assert batch.batch_size == batch_size

        # Also filter the true_memory_bank and true_src_lengths
        true_lengths = true_lengths.index_select(dim=0, index=valid_examples)
        true_memory_bank = {name: tensor.index_select(dim=1, index=valid_examples)
                            for name, tensor in true_memory_bank.items()}

        # We trim objects to remove extra padding. This is not really useful
        # to optimize compute, but since I have assertion errors everywhere
        # down the line that check for optimal size, it's the only way to re-run
        # the encoding process

        src, src_lengths = batch.src
        if (pad := src.size(0) - src_lengths.max().item()) > 0:

            # We check that we do not remove any thing else than padding in src
            assert src[-pad:, :, 0].eq(self.vocabs['main_vocab']['<pad>']).all()
            assert src[-pad:, :, 1].eq(self.vocabs['cols_vocab']['<pad>']).all()

            batch.src = src[:-pad], src_lengths

            # same check in src_map
            assert batch.src_map[-pad:].eq(0).all()
            _pad = max(len(voc) for voc in batch.src_ex_vocab)
            assert batch.src_map[:, :, _pad:].eq(0).all()

            batch.src_map = batch.src_map[:-pad, :, :_pad]

            # Also trim true_memory_bank, which contains embedded padding
            name2trim = {
                'high_level_repr': 1 + (true_lengths.max() // self.entity_size),
                'low_level_repr': true_lengths.max(),
                'low_level_mask': true_lengths.max() // self.entity_size,
                'pos_embs': true_lengths.max(),
            }
            for name, trim in name2trim.items():
                true_memory_bank[name] = true_memory_bank[name][:trim]

        # Now we can start the process of generating one sentence using beam search

        # filter decoder's states to keep only valid examples
        fn = lambda state, dim: state.index_select(dim, valid_examples)
        self.model.decoder.map_state(fn)

        # re-fetch decoder states, now that they are filtered
        dec_states = self.model.decoder.state['input_feed'].squeeze(0)

        # Also computing grounding entities and elaborations to memory_bank
        with torch.no_grad():

            # First compute the two queries
            queries = self.model.context_predictor.states_to_queries(dec_states)

        # split 'em
        first_query, second_query = queries.chunk(2, dim=1)

        # Next, compute the attention scores over all entities
        candidates = true_memory_bank['high_level_repr'].permute(1, 2, 0)
        first_scr = torch.bmm(first_query.unsqueeze(1), candidates)
        second_scr = torch.bmm(second_query.unsqueeze(1), candidates)

        # Set proba of selecting a padding views to zero
        padding_views_mask = ~sequence_mask(batch.n_primaries, first_scr.size(2))
        first_scr = first_scr.masked_fill(padding_views_mask.unsqueeze(1), float('-inf'))
        second_scr = second_scr.masked_fill(padding_views_mask.unsqueeze(1), float('-inf'))

        first_entity = first_scr.topk(1, dim=2).indices.view(batch_size)
        second_entity = second_scr.topk(1, dim=2).indices.view(batch_size)

        # Concat the chosen entities into same array, and add padding for elaborations
        contexts = torch.stack([first_entity, second_entity], dim=1)
        contexts = torch.cat([contexts, torch.zeros(batch_size, 2, device=self.device)], dim=1)

        rerun_encoding = False
        for batch_idx in range(batch_size):
            elab = elaborations[batch_idx].item()

            if elab == self.vocabs['elab_vocab']['<primary>']:
                continue

            elif elab == self.vocabs['elab_vocab']['<none>']:
                contexts[batch_idx] = 0

            elif elab == self.vocabs['elab_vocab']['<event>']:
                for eidx, ent in enumerate(contexts[batch_idx, :2], 2):

                    if ent > 0:

                        if (ent, '<event>') not in batch.elaboration_view_idxs[batch_idx]:
                            rerun_encoding = True
                            self.add_elaboration_to_batch(
                                batch, batch_idx, ent, '<event>',
                                batch.elaborations_query_mapping[batch_idx],
                                batch.elaboration_view_idxs[batch_idx]
                            )

                        _idx = batch.elaboration_view_idxs[batch_idx][ent, '<event>']
                        contexts[batch_idx][eidx] = _idx

            elif elab == self.vocabs['elab_vocab']['<time>']:
                for eidx, ent in enumerate(contexts[batch_idx, :2], 2):
                    if ent > 0:

                        if (ent, '<time>') not in batch.elaboration_view_idxs[batch_idx]:
                            rerun_encoding = True
                            self.add_elaboration_to_batch(
                                batch, batch_idx, ent, '<time>',
                                batch.elaborations_query_mapping[batch_idx],
                                batch.elaboration_view_idxs[batch_idx]
                            )

                        _idx = batch.elaboration_view_idxs[batch_idx][ent, '<time>']
                        contexts[batch_idx][eidx] = _idx

            else:
                raise RuntimeError(f'Unexpected elaboration: {elab}')

        # Not the most efficient thing, but the easiest by far:
        # we recompute the whole encoding of the source, with the added entities
        # We also need to replace all keys of true_memory_bank,
        # without creating a new object (and lengths)

        if rerun_encoding:
            with torch.no_grad():
                true_memory_bank = self.model.encoder(*batch.src, batch.n_primaries)

        true_lengths[:] = batch.src[1]

        # Clone the memory bank to avoid messing anything up during beam search
        # (e.g. ordering, done beams, etc.)
        src_lengths = true_lengths.clone()
        memory_bank = {name: tensor.clone() for name, tensor in true_memory_bank.items()}

        memory_bank['contexts'] = contexts.unsqueeze(0).long() - 1  # Padding
        memory_bank['elaborations'] = elaborations.unsqueeze(0)

        return (
            current_sentences,
            current_plans,
            true_memory_bank,
            true_lengths,
            memory_bank,
            src_lengths
        )


class GuidedInference(BaseInference):

    def __init__(self, *args, **kwargs):
        super(GuidedInference, self).__init__(*args, **kwargs)
        self.is_guided_inference = True

    def count_nb_planned_sentences(self, batch):
        return batch.elaborations.size(0)

    def prep_sentence_generation(self, batch, sent_idx,
                                 current_sentences, current_plans,
                                 prev_states, true_lengths, true_memory_bank):

        # Create list of examples that are not done. We know an example is done
        # when its elaboration is 2 (<eod>) or 1 (<pad>).
        valid_examples = batch.elaborations[sent_idx].ge(3).nonzero().squeeze(1)
        if not len(valid_examples):
            return [None] * PREP_SENTENCE_GENERATION_N_OUPTUS

        # Keep track of current sentences & plans
        current_sentences = [current_sentences[idx] for idx in valid_examples]
        current_plans = [current_plans[idx] for idx in valid_examples]

        # Filter batch examples
        batch.index_select(valid_examples)

        # Also filter the true_memory_bank and true_src_lengths
        true_lengths = true_lengths.index_select(dim=0, index=valid_examples)
        true_memory_bank = {name: tensor.index_select(dim=1, index=valid_examples)
                            for name, tensor in true_memory_bank.items()}

        # Now we can start the process of generating one sentence using beam search
        # Clone the memory bank to avoid messing anything up during beam search
        # (e.g. ordering, done beams, etc.)
        src_lengths = true_lengths.clone()
        memory_bank = {name: tensor.clone() for name, tensor in true_memory_bank.items()}

        # Initialize decoder and also filter its states to keep only valid examples
        if not sent_idx:
            self.model.decoder.init_state(memory_bank['game_repr'],
                                          memory_bank['primary_mask'])
        else:
            self.model.decoder.set_state(prev_states)

        fn = lambda state, dim: state.index_select(dim, valid_examples)
        self.model.decoder.map_state(fn)

        # Also add contexts and elaborations to memory_bank
        memory_bank['contexts'] = batch.contexts[sent_idx:sent_idx + 1].transpose(1, 2).contiguous()
        memory_bank['elaborations'] = batch.elaborations[sent_idx:sent_idx + 1]

        return (
            current_sentences,
            current_plans,
            true_memory_bank,
            true_lengths,
            memory_bank,
            src_lengths
        )
