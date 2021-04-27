# -*- coding: utf-8 -*-

from itertools import tee, zip_longest

import inspect
import random
import torch
import os


class Container:
    """
    Dummy class that can be instantiated with arbitrary key-word arguments
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, item):
        """Only called when item is not known to the container"""
        raise RuntimeError(f'{item} is not a known attribute of this Container.'
                           f'This code uses Containers at places where I need '
                           f'an objects that behaves like another object (e.g. '
                           f'a batch, a namespace, etc.). Find where this '
                           f'container is used in the code and fix this issue!')


def format_device(device_or_device_id):
    if isinstance(device_or_device_id, torch.device):
        return device_or_device_id
    if isinstance(device_or_device_id, str):
        return torch.device(device_or_device_id)
    if isinstance(device_or_device_id, int):
        device_id = device_or_device_id
        return torch.device(device_id if device_id >= 0 else 'cpu')
    raise RuntimeError(f'Unknown device: {device_or_device_id}')


class ContainsNaN(Exception):
    pass


def check_object_for_nan(obj):
    if isinstance(obj, torch.nn.Module):
        for name, tensor in obj.named_parameters():
            if (tensor != tensor).any():
                raise ContainsNaN(name)
    elif isinstance(obj, torch.Tensor):
        if (obj != obj).any():
            raise ContainsNaN()
    elif isinstance(obj, (list, tuple)):
        for _obj in obj:
            check_object_for_nan(_obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            try:
                check_object_for_nan(value)
            except ContainsNaN:
                raise ContainsNaN(key)


def nwise(iterable, n=2):
    iterables = tee(iterable, n)
    [next(iterables[i]) for i in range(n) for j in range(i)]
    return zip(*iterables)


def grouped(iterable, n):
    return zip_longest(*[iter(iterable)]*n)


def block_eye(n, size, dtype=torch.uint8, device=None):
    """
    Create a block_diagonal matrix of n blocks, where each block
    is torch.ones(size, size)
    """
    if device is None:
        device = torch.device('cpu')

    m1 = torch.ones(n, size, 1, size, dtype=dtype, device=device)
    m2 = torch.eye(n, dtype=dtype, device=device).view(n, 1, n, 1)
    return (m1*m2).view(n*size, n*size)


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def fn_args(fun):
    """Returns the list of function arguments name."""
    return inspect.getfullargspec(fun).args


def report_matrix(row_label, column_label, matrix):
    header_format = "{:>10.10} " + "{:>10.7} " * len(row_label)
    row_format = "{:>10.10} " + "{:>10.7f} " * len(row_label)
    output = header_format.format("", *row_label) + '\n'
    for word, row in zip(column_label, matrix):
        max_index = row.index(max(row))
        row_format = row_format.replace(
            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
        row_format = row_format.replace(
            "{:*>10.7f} ", "{:>10.7f} ", max_index)
        output += row_format.format(word, *row) + '\n'
        row_format = "{:>10.10} " + "{:>10.7f} " * len(row_label)
    return output


def check_model_config(model_config, root):
    # we need to check the model path + any tokenizer path
    for model in model_config["models"]:
        model_path = os.path.join(root, model)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "{} from model {} does not exist".format(
                    model_path, model_config["id"]))
    if "tokenizer" in model_config.keys():
        if "params" in model_config["tokenizer"].keys():
            for k, v in model_config["tokenizer"]["params"].items():
                if k.endswith("path"):
                    tok_path = os.path.join(root, v)
                    if not os.path.exists(tok_path):
                        raise FileNotFoundError(
                            "{} from model {} does not exist".format(
                                tok_path, model_config["id"]))
