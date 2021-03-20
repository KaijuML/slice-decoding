import torch


class ContextPredictor(torch.nn.Module):
    """
    Predicts a context from a decoder state.
    """
    def __init__(self, decoder_hidden_size, elaboration_vocab):
        super().__init__()

        self.elaboration_predictor = torch.nn.Linear(decoder_hidden_size,
                                                     len(elaboration_vocab))

    def forward(self, ):
        pass
