import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_scalar
from gatr.layers import EquiLinear, GeometricBilinear, ScalarGatedNonlinearity


def encode_tokens(type_token, global_token, token_size, isgatr, batchsize, device):
    """Compute embedded type_token and global_token to be used within Transformers

    Parameters
    type_token: iterable of int
        list with type_tokens for each particle in the event
    global_token: int
    isgatr: bool
        whether the encoded tokens will be used within L-GATr or within the baseline Transformer
        This affects how many zeroes have to be padded to the global_token (4 more for the baseline Transformer)
    batchsize: int
    device: torch.device


    Returns:
    type_token: torch.Tensor with shape (batchsize, num_particles, type_token_max)
        one-hot-encoded type tokens, to be appended to each encoded 4-momenta in case of the
        baseline transformer / make up the full scalar channel for L-GATr
    global_token: torch.Tensor with shape (batchsize, 1, type_token_max+4)
        ont-hot-encoded dataset token, this will be the global_token and appended to the individual particles
    """

    type_token_raw = torch.tensor(type_token, device=device).flatten()

    type_token = nn.functional.one_hot(type_token_raw, num_classes=token_size)
    type_token = type_token.expand(batchsize, *type_token.shape).float()

    global_token = torch.tensor(global_token, device=device)

    global_token = nn.functional.one_hot(
        global_token, num_classes=token_size + (0 if isgatr else 4)
    )

    global_token = (
        global_token.unsqueeze(0)
        .expand(batchsize, *global_token.shape)
        .float()
        .unsqueeze(1)
    )

    return type_token, global_token


class AmplitudeMLPWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token, global_token):
        # ignore type_token and global_token (architecture is not permutation invariant)
        out = self.net(inputs[:, (len(type_token) * 4) :])
        return out


class AmplitudeGATrWrapper(nn.Module):
    def __init__(self, net, token_size, reinsert_type_token=False):
        super().__init__()
        self.net = net
        # reinsert_type_token is processed in the experiment class
        self.token_size = token_size

    def forward(self, inputs: torch.Tensor, type_token, global_token):
        batchsize, _, _ = inputs.shape

        multivector, scalars = self.embed_into_ga(inputs, type_token, global_token)

        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars)
        amplitude = self.extract_from_ga(multivector_outputs, scalar_outputs)

        return amplitude

    def embed_into_ga(self, inputs, type_token, global_token):
        batchsize, num_objects, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        multivector = multivector.unsqueeze(2)

        type_token, global_token = encode_tokens(
            type_token,
            global_token,
            self.token_size,
            isgatr=True,
            batchsize=batchsize,
            device=inputs.device,
        )

        # encode type_token in scalars
        scalars = type_token

        # global token
        global_token_mv = torch.zeros(
            (batchsize, 1, multivector.shape[2], multivector.shape[3]),
            dtype=multivector.dtype,
            device=multivector.device,
        )
        global_token_s = global_token
        multivector = torch.cat((global_token_mv, multivector), dim=1)

        scalars = torch.cat((global_token_s, scalars), dim=1)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Extract scalars from GA representation
        lorentz_scalars = extract_scalar(multivector)[..., 0]

        amplitude = lorentz_scalars[:, 0, :]

        return amplitude
