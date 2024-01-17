
import torch
import pdb
import numpy as np

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)



def smooth_ln_fcs_temporary(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = True
    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1*shifts)/ scales

    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.temp_bias = fc.bias + fc.weight@shifts
        else:
            fc.temp_bias = fc.weight@shifts
        fc.temp_weight = fc.weight * scales.view(1,-1)


def smooth_fc_fc_temporary(fc1, fc2, scales,shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True

    assert len(scales) == len(shifts), "Lengths of scales and shifts must be equal"

    chunk_size = 256
    num_chunks = len(scales) // chunk_size

    shifts_2d = np.array(shifts).reshape(num_chunks, chunk_size)
    average_shifts = np.max(shifts_2d, axis=0)
    trimmed_shifts = average_shifts.tolist()
    complete_shifts = trimmed_shifts * (len(shifts) // len(trimmed_shifts))
    scales_2d = np.array(scales).reshape(num_chunks, chunk_size)
    average_scales = np.max(scales_2d, axis=0)
    trimmed_scales = average_scales.tolist()
    complete_scales = trimmed_scales * (len(scales) // len(trimmed_scales))

    if hasattr(fc1, 'temp_weight'):
        fc1.temp_bias = fc1.temp_bias - trimmed_shifts
        fc1.temp_bias = fc1.temp_bias/trimmed_scales.view(-1)
        fc1.temp_weight = fc1.temp_weight/trimmed_scales.view(-1,1)
    else:
        fc1.temp_bias = fc1.bias/trimmed_scales.view(-1)
        fc1.temp_weight = fc1.weight/trimmed_scales.view(-1,1)
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.temp_bias = fc2.bias + fc2.weight@complete_shifts
    else:
        fc2.temp_bias = fc2.weight@complete_shifts
    fc2.temp_weight = fc2.weight * complete_scales.view(1,-1)


def smooth_q_k_temporary(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True

    chunk_size = 256
    num_chunks = len(scales) // chunk_size

    scales_2d = np.array(scales).reshape(num_chunks, chunk_size)
    average_scales = np.max(scales_2d, axis=0)
    trimmed_scales = average_scales.tolist()
    complete_scales = trimmed_scales * (len(scales) // len(trimmed_scales))

    q_proj.temp_weight = q_proj.temp_weight/complete_scales.view(-1,1)
    q_proj.temp_bias = q_proj.temp_bias/complete_scales.view(-1)
    k_proj.temp_weight = k_proj.temp_weight*trimmed_scales.view(-1,1)
    k_proj.temp_bias = k_proj.temp_bias*trimmed_scales.view(-1)

def smooth_ln_fcs_inplace(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        del ln.bias
        ln.register_buffer('bias',(-1*shifts)/scales)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.use_temporary_parameter = False
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.bias.add_(fc.weight@shifts)
        else:
            del fc.bias
            fc.register_buffer('bias',fc.weight@shifts)
        fc.weight.mul_(scales.view(1,-1))


def smooth_fc_fc_inplace(fc1, fc2, scales,shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False

    assert len(scales) == len(shifts), "Lengths of scales and shifts must be equal"

    chunk_size = 256
    num_chunks = len(scales) // chunk_size

    shifts_2d = np.array(shifts).reshape(num_chunks, chunk_size)
    average_shifts = np.max(shifts_2d, axis=0)
    trimmed_shifts = average_shifts.tolist()
    complete_shifts = trimmed_shifts * (len(shifts) // len(trimmed_shifts))
    scales_2d = np.array(scales).reshape(num_chunks, chunk_size)
    average_scales = np.max(scales_2d, axis=0)
    trimmed_scales = average_scales.tolist()
    complete_scales = trimmed_scales * (len(scales) // len(trimmed_scales))

    fc1.bias.sub_(trimmed_shifts)
    fc1.bias.div_(trimmed_scales.view(-1))
    fc1.weight.div_(trimmed_scales.view(-1,1))
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.bias.add_(fc2.weight@complete_shifts)
    else:
        del fc2.bias
        fc2.register_buffer('bias',fc2.weight@complete_shifts)
    fc2.weight.mul_(complete_scales.view(1,-1))

def smooth_q_k_inplace(q_proj, k_proj, scales,):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False

    chunk_size = 256
    num_chunks = len(scales) // chunk_size

    scales_2d = np.array(scales).reshape(num_chunks, chunk_size)
    average_scales = np.max(scales_2d, axis=0)
    trimmed_scales = average_scales.tolist()
    complete_scales = trimmed_scales * (len(scales) // len(trimmed_scales))

    q_proj.weight.div_(complete_scales.view(-1,1))
    q_proj.bias.div_(complete_scales.view(-1))
    k_proj.weight.mul_(trimmed_scales.view(-1,1))
    k_proj.bias.mul_(trimmed_scales.view(-1))