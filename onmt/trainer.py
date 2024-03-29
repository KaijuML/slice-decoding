"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from onmt.utils.misc import check_object_for_nan
from onmt.utils.logging import logger

from typing import Union

import onmt.utils
import torch
import tqdm


def build_trainer(opt, model, vocabs, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        vocabs (dict): dict of vocabs (main/cols/elaborations)
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    sentence_loss, context_loss = onmt.utils.loss.build_loss_computes(model,
                                                                      vocabs,
                                                                      opt)

    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    average_decay = opt.average_decay
    average_every = opt.average_every

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    device_id = 0 if opt.use_gpu else -1
    report_manager = onmt.utils.build_report_manager(opt, device_id)
    trainer = Trainer(model, sentence_loss, context_loss,
                      optim, norm_method,
                      accum_count, accum_steps,
                      elab_loss_weight=opt.elab_loss_weight,
                      ents_loss_weight=opt.ents_loss_weight,
                      report_manager=report_manager,
                      model_saver=model_saver,
                      average_decay=average_decay,
                      average_every=average_every,
                      model_dtype=opt.model_dtype,
                      earlystopper=earlystopper)
    return trainer


class BatchError(Exception):
    base_msg = \
        '''
        There was an error at iteration {} (counting from 0).
        You may be able to reproduce the error by manually running the
        training loop on the following batches:
        '''

    def __init__(self, step, indices):
        msg = self.base_msg.format(step).strip() + '\n'
        for b_indices in indices:
            msg += f'\t{b_indices}\n'
        super().__init__(msg)


class BatchErrorHandler:
    """
    This handler raises informative errors to speed up dev work
    """

    def __init__(self, step, batches):
        self.indices = [b.indices.tolist() for b in batches]
        self.step = step

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise BatchError(self.step, self.indices)


class SentenceBatch:
    """
    This container reuses onmt's API, to compute the loss on sentences.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def batch_size(self):
        return len(self.indices)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            sentence_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,
                 model: onmt.models.NMTModel,
                 sentence_loss,
                 context_loss,
                 optim,
                 norm_method: str = "sents",
                 accum_count: Union[int] = [1],
                 accum_steps: Union[int] = [0],
                 elab_loss_weight: float = .15,
                 ents_loss_weight: float = .15,
                 report_manager: onmt.utils.ReportMgr = None,
                 model_saver=None,
                 average_decay: int = 0,
                 average_every: int = 1,
                 model_dtype: str = 'fp32',
                 earlystopper=None):

        # Basic attributes.
        self.model = model
        self.sentence_loss = sentence_loss
        self.context_loss = context_loss
        self.valid_loss = None  # for now
        self.optim = optim
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.elab_loss_weight = elab_loss_weight
        self.ents_loss_weight = ents_loss_weight

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0

        # Set model in training mode.
        self.model.train()

    @property
    def report_every(self):
        return self.report_manager.report_every

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.model.parameters()]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay,
                                1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(enumerate(self.moving_average),
                                     self.model.parameters()):
                self.moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        iterable = enumerate(self._accum_batches(train_iter), 1)
        with tqdm.tqdm(total=self.report_every) as progressbar:
            for i, (batches, normalization) in iterable:
                step = self.optim.training_step

                # Everything is done here!
                with BatchErrorHandler(i, batches):
                    self._gradient_accumulation(batches, total_stats, report_stats)

                if i % progressbar.total == 0:
                    progressbar.n = 0
                else:
                    progressbar.update(n=1)
                progressbar.refresh()

                if self.average_decay > 0 and i % self.average_every == 0:
                    self._update_average(step)

                report_stats = self._maybe_report_training(
                    step, train_steps,
                    self.optim.learning_rate(),
                    report_stats)

                if (self.model_saver is not None
                        and (save_checkpoint_steps != 0
                             and step % save_checkpoint_steps == 0)):
                    self.model_saver.save(step, moving_average=self.moving_average)

                if 0 < train_steps <= step:
                    break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    @property
    def device(self):
        return self.model.device

    def _gradient_accumulation(self, true_batches, total_stats, report_stats):

        # Zeroing gradient before iterating through batches
        self.optim.zero_grad()

        for k, batch in enumerate(true_batches):

            src, src_lengths = batch.src
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            with torch.cuda.amp.autocast(enabled=self.optim.amp):

                # 1. Encode the entire input table. Both primary entities
                #    and additional entities are encoded here.
                memory_bank = self.model.encoder(*batch.src, batch.n_primaries)

                # 2. Decode sentence by sentence
                # Note that from the decoder's POV, it's like decoding one long
                # sequence. For each example of the batch, sentences have been
                # concatenated to form one sequence.
                # We use the mapping of token to sentence index to also build
                # the representation of current slice for each sentence.

                # We initialize losses at zero
                losses = {'main': None, 'ents': None, 'elab': None}

                # 2.0 Initialize decoder's hidden state with the encoder out.
                decoder = self.model.decoder
                decoder.init_state(memory_bank['game_repr'],
                                   memory_bank['primary_mask'])

                # Extract sentences and token to sentence index mapping
                sentences, sentence_starts, sentence_indices = batch.sentences

                # 2.1 Decode sentences

                # 2.1.1 Compute slice representation for each sentence
                context_repr, contexts = decoder(
                    action="compute_context_representation",
                    memory_bank=memory_bank,
                    contexts=batch.contexts,
                    elaborations=batch.elaborations
                )

                # The sentence_indices are padded with zeros and first sent is
                # denoted by 1. To use torch.gather effectively, we then add
                # a fake context on contexts[0] which will be gathered on
                # pad index, and move everything by 1.
                n_sents, batch_size, repr_dim = context_repr.shape
                fake_ctx = torch.zeros(1, batch_size, repr_dim, device=self.device)
                context_repr = torch.cat([fake_ctx, context_repr], dim=0)
                index = sentence_indices.unsqueeze(2).expand(-1, -1, repr_dim)
                context_repr = context_repr.gather(dim=0, index=index)

                # We make the same operation on the contexts tensor. Because
                # the result will be used in indexing/masking operations, we
                # move everything to cpu, to save some CUDA memory
                n_sents, n_ents, batch_size = contexts.shape
                index = sentence_indices.unsqueeze(1).expand(-1, n_ents, -1)
                contexts = torch.cat([contexts[-1:], contexts], dim=0).to('cpu')
                contexts = contexts.gather(dim=0, index=index.to("cpu"))

                # 2.1.2 Actually decode sentences
                outputs, attns = decoder(action='decode_full',
                                         sentences=sentences,
                                         context_repr=context_repr,
                                         contexts=contexts,
                                         memory_bank=memory_bank)

                # 3. Compute losses based on decoder's hidden states.

                # This batch object is used to comply with onmt's API
                _batch = SentenceBatch(
                    src_map=batch.src_map,
                    src_ex_vocab=batch.src_ex_vocab,
                    tgt=sentences,
                    alignment=batch.alignments,
                    indices=batch.indices,
                )

                # 3.1 Compute the main loss from sentence generation
                loss, batch_stats = self.sentence_loss(
                    _batch,
                    outputs,
                    attns
                )

                losses['main'] = loss / _batch.batch_size

                # 2.2 Predict context at the start of each sentence
                ents_loss, elab_loss = self.context_loss(outputs,
                                                         memory_bank,
                                                         sentence_starts,
                                                         batch.n_primaries,
                                                         batch.elaborations,
                                                         batch.contexts)

                losses['ents'] = ents_loss / _batch.batch_size
                losses['elab'] = elab_loss / _batch.batch_size

                # Gather stats from each batch example
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

            # Compute backward pass for each batch, using the three losses
            self.optim.backward(self._merge_losses(losses))

        # Sanity check before updating weights
        check_object_for_nan(self.model)

        # After going through all batches, update parameters
        self.optim.step()

        # Sanity check after updating weigths
        check_object_for_nan(self.model)

    def _merge_losses(self, loss_dict):
        """
        TODO: compute an actual weighted average of the losses.
        """

        main_loss = loss_dict['main']
        ents_loss = loss_dict['ents']
        elab_loss = loss_dict['elab']

        w1 = self.elab_loss_weight
        w2 = self.ents_loss_weight
        w3 = 1 - w1 - w2

        final_loss = (w1 * elab_loss) + (w2 * ents_loss) + (w3 * main_loss)

        return final_loss

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=False)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)
