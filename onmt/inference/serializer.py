from onmt.rotowire.utils import MultiOpen
from copy import deepcopy
import os


class DestinationAlreadyExistsError(Exception):
    pass


class Serializer:

    def __init__(self,
                 vocabs,
                 logger,

                 desc_dest,
                 plan_dest,
                 if_file_exists='raise'):

        self.vocabs = vocabs
        self.logger = logger

        if if_file_exists not in {'raise', 'overwrite', 'append'}:
            raise ValueError(f'Unknown instruction: {if_file_exists=}')

        self.desc_dest = self.check_destination(desc_dest, if_file_exists)
        self.plan_dest = self.check_destination(plan_dest, if_file_exists)

        self.data = None

    def check_destination(self, dest, if_file_exists):
        if os.path.exists(dest):
            if if_file_exists == 'raise':
                raise DestinationAlreadyExistsError(dest)
            elif if_file_exists == 'overwrite':
                self.logger.info(f'Overwriting destination file: {dest}')
                with open(dest, mode="w", encoding='utf8') as f:
                    pass  # overwrites
            else:
                self.logger.info('Appending new generations to existing file: {dest}')
        return dest

    def prep_serialization(self, batch):
        self.data = deepcopy(batch.elaborations_query_mapping)

    def serialize(self, batch_predicted_descriptions):

        kwargs = {"mode": "a", "encoding": "utf8"}
        with MultiOpen(self.desc_dest, self.plan_dest, **kwargs) as files:
            desc_file, plan_file = files
            for bidx, (desc, plan) in enumerate(zip(*batch_predicted_descriptions)):
                assert len(desc) == len(plan)

                for sentence, (entities, elab) in zip(desc, plan):
                    desc_file.write(sentence.strip('<s> ').replace('_', ' ') + '\n')
                    plan_file.write(self._format_plan(bidx, entities, elab) + '\n')

                desc_file.write('\n')
                plan_file.write('\n')

        self.data = None

    def _format_plan(self, bidx, entities, elab):
        formatted_plan = self.vocabs["elab_vocab"].itos[elab]
        if formatted_plan == '<none>':
            return formatted_plan

        entities = [e for e in entities if 0 <= e < len(self.data[bidx])]
        if entities:
            formatted_plan += ' ' + ', '.join([
                self.data[bidx][e]['<primary>']['FULL_NAME']
                for e in entities
            ])

        return formatted_plan

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