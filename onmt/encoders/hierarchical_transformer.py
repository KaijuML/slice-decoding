from onmt.modules.self_attention import MultiHeadSelfAttention
from onmt.encoders.encoder import EncoderBase
from onmt.utils.misc import sequence_mask
import torch


class ContainsNaN(Exception):
    pass


def _check_for_nan(tensor, msg=''):
    if (tensor!=tensor).any():
        raise ContainsNaN(msg)


class FeedForward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(input_size)
        
    def forward(self, src):
        ret = self.linear1(self.norm(src))
        ret = self.linear2(self.dropout(torch.nn.functional.relu(ret)))
        return src + self.dropout(ret)  # residual connetion
    
    def update_dropout(self, dropout):
        self.dropout.p = dropout
        

class TransformerEncoderLayer(torch.nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
        This standard encoder layer is based on the paper "Attention Is All You Need".
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, 
        Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in 
        Neural Information Processing Systems, pages 6000–6010.
        Users may modify or implement in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    """

    def __init__(self, input_size, heads, dim_feedforward=2048, glu_depth=-1, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(input_size, heads, 
                                                dropout=dropout,
                                                glu_depth=glu_depth)
        self.norm = torch.nn.LayerNorm(input_size, dim_feedforward, dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.feedforward = FeedForward(input_size, dim_feedforward, dropout)

    def forward(self, src, src_mask=None):
        """Pass the input through the layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
        """
        src = src + self.dropout(self.self_attn(self.norm(src), attn_mask=src_mask)[0])
        return self.feedforward(src)
                
    def update_dropout(self, dropout):
        self.feedforward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(torch.nn.Module):
    """TransformerEncoder is a stack of N transformer encoder layers
    It is heavily inspired by pytorch's.
    
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, hidden_size, heads=8, num_layers=6, glu_depth=-1,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
            
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(input_size=hidden_size,
                                                heads=heads, 
                                                dim_feedforward=dim_feedforward, 
                                                glu_depth=glu_depth,
                                                dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, src, mask=None):
        r"""Pass the input through the all layers in turn.
        Args:
            src: the sequence to encode (required).
            src_mask: the mask for the src sequence (optional).
        """
        for encoder_layer in self.layers: 
            src = encoder_layer(src, mask)
        return self.final_norm(src)
            
    def update_dropout(self, dropout):
        for layer in self.layers: layer.update_dropout(dropout)
            

def block_eye(n, size):
    """
    Create a block_diagonal matrix of n blocks, where each block
    is torch.eye(size)
    """
    m1 = torch.ones(n, size, 1, size)
    m2 = torch.eye(n).view(n, 1, n, 1)
    return (m1*m2).view(n*size, n*size).to(torch.uint8)


def build_pad_mask(source, ent_size, pad_idx):
    """
    [seq_len, n_ents, ent_size]
    To be used in attention mechanism in decoder
    """
    mask = source[:, :, 0]
    mask = (mask.transpose(0, 1)
                  .squeeze()
                  .contiguous()
                  .view(source.size(1), -1, ent_size)
                  .eq(pad_idx))
    mask[:, :, 0] = 1  # we also mask the <ent> token
    return mask


def build_chunk_mask(lengths, ent_size):
    """
    [bsz, n_ents, n_ents]
    Filled with -inf where self-attention shouldn't attend, a zeros elsewhere.
    """
    ones = lengths // ent_size
    ones = sequence_mask(ones).unsqueeze(1).repeat(1, ones.max(), 1).to(lengths.device)
    mask = torch.full(ones.shape, float('-inf')).to(lengths.device)
    mask.masked_fill_(ones, 0)
    return mask
    

class HierarchicalTransformerEncoder(EncoderBase):
    """
    Two encoders, one on the unit level and one on the chunk level
    """
    def __init__(self, embeddings, dataset_config,
                 units_layers=2, chunks_layers=2, 
                 units_heads=2, chunks_heads=2,
                 units_glu_depth=-1, chunks_glu_depth=-1,  
                 dim_feedforward=1000, dropout=.5):
        super().__init__()
        
        self.embeddings = embeddings
        
        self.ent_size = dataset_config.entity_size
        
        self.unit_encoder = TransformerEncoder(hidden_size=embeddings.embedding_size, 
                                               heads=units_heads, 
                                               num_layers=units_layers, 
                                               dim_feedforward=dim_feedforward, 
                                               glu_depth=units_glu_depth,
                                               dropout=dropout)
        self.chunk_encoder = TransformerEncoder(hidden_size=embeddings.embedding_size, 
                                                heads=chunks_heads,
                                                num_layers=chunks_layers, 
                                                dim_feedforward=dim_feedforward, 
                                                glu_depth=chunks_glu_depth,
                                                dropout=dropout)
    @classmethod
    def from_opt(cls, opt, embeddings, dataset_config):
        """Alternate constructor."""
        
        return cls(
            embeddings=embeddings,
            dataset_config=dataset_config,
            units_layers=opt.low_level_layers,
            chunks_layers=opt.high_level_layers,
            units_heads=opt.low_level_heads,
            chunks_heads=opt.high_level_heads,
            dim_feedforward=opt.transformer_ff,
            units_glu_depth=opt.low_level_glu_depth,
            chunks_glu_depth=opt.high_level_glu_depth,
            dropout=opt.dropout
        )
    
    def forward(self, src, lengths=None):
        """
        See :func:`EncoderBase.forward()`
        
        src (tensor) [seq_len, bs, 2]
        2 <-- (value, type)
        """
        self._check_args(src, lengths)
        
        seq_len, bsz, _ = src.shape
        n_ents = seq_len // self.ent_size
        
         # sanity check
        assert seq_len % n_ents == 0
        assert seq_len == lengths.max()
        
        # We build the masks for self attention and decoding
        eye = block_eye(n_ents, self.ent_size).to(src.device)
        self_attn_mask = torch.full((seq_len, seq_len), float('-inf')).to(src.device)
        self_attn_mask.masked_fill_(eye.to(src.device), 0)
        unit_mask = build_pad_mask(src, self.ent_size, self.embeddings.word_padding_idx).to(src.device)
        chunk_mask = build_chunk_mask(lengths, self.ent_size).to(src.device)
        
        # embs [seq_len, bs, hidden_size]
        embs, pos_embs = self.embeddings(src)
        _check_for_nan(embs, 'after embedding layer')
        _check_for_nan(pos_embs, 'after embedding layer')
        
        # units [seq_len, bs, hidden_size]
        units = self.unit_encoder(embs, mask=self_attn_mask)

        # chunks & units_tokens [n_units, bs, hidden_size]
        units_tokens = units[range(0, seq_len, self.ent_size), :, :]
        chunks = self.chunk_encoder(units_tokens, mask=chunk_mask)
        
        # memory bank every thing we want to pass to the decoder
        # all tensors should have dim(1) be the batch size
        memory_bank = (
            chunks, 
            units,
            pos_embs,
            unit_mask.transpose(0, 1),
            chunk_mask[:, 0, :].unsqueeze(0).eq(float('-inf'))
        )
        
        # We average the units representation to give a final encoding
        # and be inline with the onmt framework
        encoder_final = chunks.mean(dim=0).unsqueeze(0)
        
        return encoder_final, memory_bank, lengths
