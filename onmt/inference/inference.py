from onmt.rotowire import RotowireGuidedInferenceDataset, build_dataset_iter
from onmt.modules.copy_generator import collapse_copy_scores
from onmt.inference import BeamSearch, GNMTGlobalScorer
from onmt.model_builder import load_test_model
from onmt.utils.misc import Container

import torch
import tqdm
import os


def build_inference(opts, logger=None):

    device = torch.device(f'cuda:{opts.gpu}') if opts.gpu >= 0 else None
    vocabs, model, model_opts = load_test_model(opts.model_path, device)

    inference_cls = GuidedInference if opts.guided_inference else Inference
    return inference_cls.from_opts(
        model,
        vocabs,
        opts,
        logger=logger
    )


class BaseInference:
    def __init__(self,
                 model,
                 vocabs,

                 seed,
                 gpu,

                 beam_size,
                 min_sent_length,
                 max_sent_length,

                 block_ngram_repeat,
                 ignore_when_blocking,

                 dest,
                 logger):

        self.model = model
        self.vocabs = vocabs

        self.seed = seed
        self.gpu = gpu

        self.beam_size = beam_size
        self.min_sent_length = min_sent_length
        self.max_sent_length = max_sent_length

        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking

        self.dest = dest
        self.logger = logger

    @property
    def device(self):
        return self.model.device

    @classmethod
    def from_opts(cls, model, vocabs, opts, logger):
        return cls(
            model=model,
            vocabs=vocabs,

            seed=opts.seed,
            gpu=opts.gpu,

            beam_size=opts.beam_size,
            min_sent_length=opts.min_sent_length,
            max_sent_length=opts.max_sent_length,

            block_ngram_repeat=opts.block_ngram_repeat,
            ignore_when_blocking=set(opts.ignore_when_blocking),

            dest=opts.dest,
            logger=logger,
        )

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

            tokens.extend(lexicalized_tok.split('_'))

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


class GuidedInference(BaseInference):

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

        dataset = RotowireGuidedInferenceDataset.build_from_raw_json(
            filename, config=self.model.config, vocabs=self.vocabs)

        opt = Container(batch_size=batch_size, num_threads=1)
        inference_iter = build_dataset_iter(dataset, opt, self.device)

        for batch in tqdm.tqdm(inference_iter, desc="Running inference"):
            batch_predicted_sentences = self.run_on_batch(batch)

            with open(self.dest, mode="a", encoding="utf8") as f:
                for predicted_sentences in batch_predicted_sentences:
                    pred = ' '.join(sent.strip('<s> ')
                                    for sent in predicted_sentences)
                    f.write(f"{pred}\n")

    def run_on_batch(self, batch):

        # Used in warnings
        warn_indices = batch.indices.clone()

        # Count number of planned sentences
        n_sentences = batch.elaborations.size(0)

        # prepare final outputs
        all_sentences = [list() for _ in range(batch.batch_size)]

        # tracking predictions that are still not done
        current_sentences = all_sentences

        with torch.no_grad():
            # 1. Run the encoder on the src.
            true_memory_bank = self.model.encoder(*batch.src, batch.n_primaries)
            true_lengths = batch.src[1]

        # Generate sentences one by one using beam search
        for sent_idx in range(n_sentences):

            # Create list of examples that are not done. We know an example is done when
            # it elaboration is 2 (<eod>) or 1 (<pad>).
            valid_examples = batch.elaborations[sent_idx].ge(3).nonzero().squeeze(1)
            if not len(valid_examples):
                if sent_idx != n_sentences - 1:
                    self.logger.warning('Stopping generation earlier than '
                                        f'expected, for batch={warn_indices}')
                break

            # Keep track of current sentences
            current_sentences = [current_sentences[idx] for idx in valid_examples]

            # Filter batch examples
            batch.index_select(valid_examples)

            # Also filter the true_memory_bank and true_src_lengths
            true_lengths = true_lengths.index_select(dim=0, index=valid_examples)
            true_memory_bank = {name: tensor.index_select(dim=1, index=valid_examples)
                                for name, tensor in true_memory_bank.items()}

            # Now we can start the process of generating one sentence using beam search

            # Clone the memory bank to avoid messing anything up during beam search
            # (e.g. orderding, done beams, etc.)
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

        return all_sentences

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


class Inference(BaseInference):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Not-Guided Inference is not implemented yet')
