from onmt.utils.misc import sequence_mask, block_eye, check_object_for_nan
from onmt.modules.self_attention import MultiHeadSelfAttention
import torch


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
    

class HierarchicalTransformerEncoder(torch.nn.Module):
    """
    Two encoders, one on the unit level and one on the chunk level
    """
    def __init__(self, embeddings, dataset_config,
                 low_level_layers=2, high_level_layers=2,
                 low_level_heads=2, high_level_heads=2,
                 units_glu_depth=-1, chunks_glu_depth=-1,
                 dim_feedforward=1000, dropout=.5):
        super().__init__()
        
        self.embeddings = embeddings
        self.hidden_size = hidden_size=embeddings.embedding_size
        
        self.ent_size = dataset_config.entity_size
        
        self.low_level_encoder = TransformerEncoder(hidden_size=self.hidden_size,
                                                    heads=low_level_heads,
                                                    num_layers=low_level_layers,
                                                    dim_feedforward=dim_feedforward,
                                                    glu_depth=units_glu_depth,
                                                    dropout=dropout)
        self.high_level_encoder = TransformerEncoder(hidden_size=self.hidden_size,
                                                     heads=high_level_heads,
                                                     num_layers=high_level_layers,
                                                     dim_feedforward=dim_feedforward,
                                                     glu_depth=chunks_glu_depth,
                                                     dropout=dropout)

        game_repr = torch.nn.Parameter(torch.Tensor(1, 1, self.hidden_size))
        self.register_parameter('game_repr', game_repr)

        self._init_parameters_manually()

        # once the module is initialized, check for NaNs
        check_object_for_nan(self)

    def _init_parameters_manually(self):
        torch.nn.init.uniform_(self.game_repr, -1, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def from_opt(cls, opt, embeddings, dataset_config):
        """Alternate constructor."""
        
        return cls(
            embeddings=embeddings,
            dataset_config=dataset_config,
            low_level_layers=opt.low_level_layers,
            high_level_layers=opt.high_level_layers,
            low_level_heads=opt.low_level_heads,
            high_level_heads=opt.high_level_heads,
            dim_feedforward=opt.transformer_ff,
            units_glu_depth=opt.low_level_glu_depth,
            chunks_glu_depth=opt.high_level_glu_depth,
            dropout=opt.dropout
        )

    @staticmethod
    def build_low_level_mask(source, ent_size, pad_idx):
        """
        Builds a mask for the hierarchical attention module (not this one).
        The mask is [batch_size, n_ents, ent_size], with True everywhere there
        is a padding / ent token.
        """
        mask = (source[:, :, 0].transpose(0, 1)
                .squeeze()
                .contiguous()
                .view(source.size(1), -1, ent_size)
                .eq(pad_idx))
        mask[:, :, 0] = 1  # we also mask the <ent> token
        return mask

    def build_high_level_mask(self, lengths, max_size):
        """
        Builds a mask for the hierarchical encoder module (this one!).
        The mask is [bsz, n_ents, n_ents], with 0 everywhere an entity can
        attend to another entity, and filled with -inf where self-attention
        shouldn't attend.
        """
        ones = sequence_mask(lengths, max_size).unsqueeze(1).expand(-1, max_size, -1)
        mask = torch.full(ones.shape, float('-inf'), device=self.device)
        mask.masked_fill_(ones, 0)
        return mask

    def forward(self, src, lengths, n_primaries):
        """
        The Hierarchical Encoder first encodes all entities independently
        and them computes a game representation given all entities repr.
            1) A low_level_encoder outputs low_level_repr
            2) A high_level_encoder outputs high_level_repr

        To compute the high_level_repr, we extract the first repr of each entity
                low_level_repr[range(0, n_ents * ent_size, ent_size)
        We also add a special game_repr token, to be used at two placed later
        on: to initialize the encoder hidden state, to be used as context repr
        for sentences that have no grounded entities.

        :param src: (torch.LongTensor) [seq_len, batch_size, 2]
                The source tokens that should be encoded. On dim(2), first row
                are the actual cell values, and second row are column names
        :param lengths: (torch.LongTensor) [batch_size]
                Total number of non-null entities for each batch example. It is
                used nowhere and will be removed in a future commit. Hi to you
                if you are actually reading all file of all commit!
        :param n_primaries: (torch.LongTensor) [batch_size]
                Number of primary entities for each batch example. Used to build
                the high_level_mask, so that the game_repr does not depend of
                elaborations that are supposed to be chosen later on in the
                decoding procedure

        :returns high_level_repr (torch.FloatTensor) [n_ents, batch_size, dim]
            High level representation of <ent> tokens, as computed by the high
            level encoder, after extracting the reprs computed by the low level
            encoder.
        :returns low_level_repr (torch.FloatTensor) [seq_len, batch_size, dim]
            Representation of all cell values, as computed by the low_level_encoder
            (reprs from ont entity are independent of all reprs from other entites.
        :returns pos_embs (torch.FloatTensor) [seq_len, batch_size, dim]
            Embeddings of column names. Will be used instead of low_level_repr
            in the attention module, if --use-pos is given as a training option
        :returns low_level_mask (torch.FloatTensor) [batch_size, n_ents, ent_size]
            Mask for the padding / ent tokens inside each entities
        :returns high_level_mask (torch.FloatTensor) [1 batch_size, n_ents]
            Dynamic context mask for the hierarchical attention. Constrains the
            decoder to attend to a select number of entities each step.
        :returns game_repr (torch.FloatTensor) [1 batch_size, dim]
            Game representation computed as an aggregation of all primary reprs

        """
        
        seq_len, bsz, _ = src.shape
        n_ents = seq_len // self.ent_size
        
        # sanity check
        assert seq_len % n_ents == 0
        assert seq_len == lengths.max()
        
        # We build the masks for self attention and decoding
        eye = block_eye(n_ents, self.ent_size, device=self.device, dtype=torch.bool)
        self_attn_mask = torch.full((seq_len, seq_len), float('-inf'), device=self.device)
        self_attn_mask.masked_fill_(eye, 0)
        low_level_mask = self.build_low_level_mask(src, self.ent_size,
                                              self.embeddings.word_padding_idx)

        # high_level_mask is based on primaries, because we don't want to
        # include elaboration at this step of the encoding (during inference,
        # this will not be available information).
        # We +1 the number of primaries, because we include a <game> token
        # (which is simply a learned parameter of the encoder)
        # that will serve as an aggregated repr for the whole match. This repr
        # will be used to ground sentences which have no grounded entities,
        # as well as initialize the decoder's hidden states.
        high_level_mask = self.build_high_level_mask(n_primaries + 1, n_ents + 1)
        
        # embs [seq_len, bs, hidden_size]
        embs, pos_embs = self.embeddings(src)
        check_object_for_nan(embs)
        check_object_for_nan(pos_embs)
        
        # low_level_repr [seq_len, bs, hidden_size]
        low_level_repr = self.low_level_encoder(embs, mask=self_attn_mask)

        # Extract the repr at the positions of <ent> tokens.
        # high_level_repr  [n_units, bs, hidden_size]
        high_level_repr = low_level_repr[range(0, seq_len, self.ent_size), :, :]

        # We also add a <game> repr at the beginning.
        # self.game_repr is [1, 1, dim] and is expanded to [1, batch_size, dim]
        game_repr = self.game_repr.expand(-1, high_level_repr.size(1), -1)
        high_level_repr = torch.cat([game_repr, high_level_repr], dim=0)

        # We high level encode the entities
        high_level_repr = self.high_level_encoder(high_level_repr, mask=high_level_mask)

        # We extract the <game> repr to initialize the decoder's hidden states
        # We also remove useless dim in the high_level_mask
        game_repr, high_level_repr = high_level_repr.split([1, n_ents], dim=0)
        high_level_mask = high_level_mask[:, 1, 1:]  # we removed game_repr

        # memory bank every thing we want to pass to the decoder
        # all tensors should have dim(1) be the batch size
        memory_bank = {
            'high_level_repr': high_level_repr,
            'low_level_repr': low_level_repr,
            'pos_embs': pos_embs,
            'low_level_mask': low_level_mask.transpose(0, 1),
            'high_level_mask': high_level_mask.unsqueeze(0).eq(float('-inf')),
            'game_repr': game_repr
        }

        return memory_bank
