"""
As everywhere else in this repo, this file has been heavily modified for our
use case. I have removed everything that is not relevant to training our model
on RotoWire.

It's not possible to build any other model than the one presented in our paper.
"""
import re
import torch
from torch.nn.init import xavier_uniform_

import onmt.modules

from onmt.encoders import HierarchicalTransformerEncoder
from onmt.decoders import HierarchicalRNNDecoder

from onmt.modules import TableEmbeddings, CopyGenerator, ContextPredictor
from onmt.utils.parse import ArgumentParser
from onmt.utils.logging import logger


def build_embeddings(opt, main_vocab, cols_vocab):
    """
    Builds embeddings for table values and cols, using both vocabs.
    Embeddings for values can be accesed via .value_embeddings
    Args:
        opt: the option in current environment.
        vocabs: main and cols vocabs.
    """
        
    # value field
    word_padding_idx = main_vocab['<pad>']
    word_vocab_size = len(main_vocab)

    # pos field
    feat_padding_idx = cols_vocab['<pad>']
    feat_vocab_size = len(cols_vocab)

    ent_idx = main_vocab['<ent>']

    return TableEmbeddings(
        word_vec_size=opt.main_embedding_size,
        word_vocab_size=word_vocab_size,
        word_padding_idx=word_padding_idx,
        feat_vec_size=opt.cols_embedding_size,
        feat_vocab_size=feat_vocab_size,
        feat_padding_idx=feat_padding_idx,
        merge=opt.cols_merge,
        merge_activation=opt.cols_merge_activation,
        dropout=opt.dropout,
        ent_idx=ent_idx
    )


def load_test_model(model_path, device=None):
    device = device if device is not None else torch.device('cpu')

    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)

    vocabs = checkpoint['vocabs']

    model = build_base_model(model_opt,
                             vocabs,
                             checkpoint['config'],
                             checkpoint,
                             -1).to(device)
    model.eval()
    model.generator.eval()
    return vocabs, model, model_opt


def build_base_model(model_opt, vocabs, config, checkpoint=None, device_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        vocabs: the main/cols vocabs built from the training set.
        config: rotowire config.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        device_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # Build embeddings.
    table_embeddings = build_embeddings(model_opt, 
                                        vocabs['main_vocab'],
                                        vocabs['cols_vocab'])

    # Build encoder.
    encoder = HierarchicalTransformerEncoder.from_opt(model_opt, 
                                                      table_embeddings, 
                                                      config)
    
    # Build decoder
    decoder = HierarchicalRNNDecoder.from_opt(model_opt, 
                                              table_embeddings,
                                              config)
    
    # Build NMTModel(= encoder + decoder).
    model = onmt.models.NMTModel(encoder, decoder, config)

    # Build Generator (it's always a CopyGenerator)
    vocab_size = len(vocabs['main_vocab'])
    pad_idx = vocabs['main_vocab']['<pad>']
    generator = CopyGenerator(decoder.hidden_size, vocab_size, pad_idx)
    if model_opt.share_decoder_embeddings:
        if not generator.linear.weight.shape == decoder.embeddings.weight.shape:
            msg = """
                You want to tie generator's weights with the decoder's
                embedding's weights. 
                However, they do not have the same shape... You are probably
                using "concat" as a way to merge values embeddings and cols
                embeddings, which leads to decoder's hidden state having the
                wrong size.
            """
            raise ValueError(msg.replace('\n', ' ').replace('    ', '').strip())
        generator.linear.weight = decoder.embeddings.weight

    # Build Context Predictor
    context_predictor = ContextPredictor(decoder.hidden_size,
                                         vocabs['elab_vocab'])

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:

        def _load(module, name, ckpt):
            state_dict = ckpt.get(name, None)
            if state_dict is None:
                logger.warn(f'State dict not found: {name}')
            else:
                res = module.load_state_dict(ckpt[name], strict=False)
                if len(res.missing_keys):
                    logger.warn('The following keys were missing from the '
                                f'{name} checkpoint: {res.missing_keys}')
                if len(res.unexpected_keys):
                    logger.warn('The following keys were missing from the '
                                f'{name} checkpoint: {res.unexpected_keys}')

        _load(encoder, 'encoder', checkpoint)
        _load(decoder, 'decoder', checkpoint)
        _load(generator, 'generator', checkpoint)
        _load(context_predictor, 'context_predictor', checkpoint)

    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in context_predictor.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in context_predictor.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

    model.generator = generator
    model.context_predictor = context_predictor
    
    # Cast to desired device (either cuda:0 or cpu)
    if isinstance(device_id, int) and device_id>=0:
        device = torch.device("cuda", device_id)
    else:
        device = torch.device("cpu")
    model.to(device)
    
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()
        
    return model


def build_model(model_opt, opt, vocabs, config, checkpoint, verbose=True):
    if verbose: logger.info('Building model...')
    device_id = 0 if opt.use_gpu else -1
    model = build_base_model(model_opt, vocabs, config, checkpoint, device_id)
    if verbose: logger.info(model)
    return model
