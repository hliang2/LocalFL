import collections
import logging
import math
import sys
import copy
import numpy as np
import torch
# import torch.distributed as dist
import functools

def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def communicate(tensors, communication_op, attention=False):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    flat_tensor = flatten_tensors(tensors)
    communication_op(tensor=flat_tensor)
    if attention:
        return tensors/flat_tensor
    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        t.set_(f)

def SyncEAvg(model, anchor_model, rank, size, group, alpha):
    '''
    Inputs:
        model: (x^i) local neural net model at i-th worker node
        anchor_model: (z^1=z^2=...=z^m=z) local copy of auxiliary variable
        rank: (i) worker index
        size: (m) total number of workers
        group: worker group
        alpha: (a) elasticity parameter
    Output:
        return void, change in-place
    Formula:
        x_new = (1-a)*x^i + a*z
        z_new = z + a*(sum_i x^i - m*z) 
    '''

    for param1, param2 in zip(anchor_model.parameters(), model.parameters()):
        diff = (param2.data - param1.data)
        param2.data = (1-alpha)*param2.data + alpha*param1.data
        param1.data = param1.data/float(size) + alpha*diff
    
    for param in anchor_model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group)


def AsyncEAvg(model, anchor_model, rank, size, group, req, alpha):
    '''
    Inputs:
        model: (x^i) local neural net model at i-th worker node
        anchor_model: (z^1=z^2=...=z^m=z) local copy of auxiliary variable
        rank: (i) worker index
        size: (m) total number of workers
        group: worker group
        alpha: (a) elasticity parameter
        req: handle of last iteration's communication
    Output:
        return a handle of asynchronous fuction
    Formula:
        x_new = (1-a)*x^i + a*z
        z_new = z + a*(sum_i x^i - m*z)
        * the computation of z_new isn't finished when the function returns
    '''
    if req:
        for param1, param2 in zip(anchor_model.parameters(), model.parameters()):
            req[param1].wait() # wait the last iteration's update of z to finish

            diff = (param2.data - param1.data)
            param2.data = (1-alpha)*param2.data + alpha*param1.data
            param1.data = param1.data/float(size) + alpha*diff
    else:
        for param1, param2 in zip(anchor_model.parameters(), model.parameters()):
            diff = (param2.data - param1.data)
            param2.data = (1-alpha)*param2.data + alpha*param1.data
            param1.data = param1.data/float(size) + alpha*diff
    
    for param in anchor_model.parameters():
        req[param] = dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group, async_op=True)

    return req


def AsyncEAvgHierarchical(new_anchor_model, anchor_model):
    for param1, param2 in zip(anchor_model.parameters(), new_anchor_model.parameters()):
        param1.data = (param1.data + param2.data) / 2
    return new_anchor_model

def VRLSGDAllreduce(model, deviation, lr, cp, ratio):
    communication_op = functools.partial(dist.all_reduce)
    params_list = []
    tmp_model = [torch.clone(param.data).detach().cuda() for param in model.parameters()]

    for param in model.parameters():
        param.data.mul_(ratio)
        params_list.append(param.data)

    communicate(params_list, communication_op)

    for param1, param2, param3 in zip(tmp_model, deviation, model.parameters()):
        param2.data.add_(1/(lr * cp), param3.data - param1.data)
    #     params_list.append(param.data)
    # deviation += 1/(lr * cp) * (model.parameters().data - tmp_model)

    return deviation

def NormalSGDALLreduce(models, anchor_models, localCps, globalCp, ratios):
    # communication_op = functools.partial(dist.all_reduce)
    params_list = []
    for model, anchor_model, localCp, ratio in zip(models, anchor_models, localCps, ratios):
        sub_params_list = []
        for param1, param2 in zip(model.parameters(), anchor_model.parameters()):
            variance = param1.data - param2.data
            variance.mul_(ratio).div_(localCp).mul_(globalCp)
            sub_params_list.append(variance)
        params_list.append(sub_params_list)

    new_list = []
    for i in range(len(params_list[0])):
        x = [j[i] for j in params_list]
        y = sum(x)
        new_list.append(y)

    for model, anchor_model in zip(models, anchor_models):
        for param1, param2, param3 in zip(model.parameters(), anchor_model.parameters(), new_list):
            param2.data.add_(param3.data)
            param1.data = param2.data.clone().detach()

def SyncAllreduce(model, rank, size):
    '''
    Inputs:
        model: (x^i) local neural net model at i-th worker node
        anchor_model: (z^1=z^2=...=z^m=z) local copy of auxiliary variable
        rank: (i) worker index
        size: (m) total number of workers
        group: worker group
    Output:
        return void, change in-place
    Formula:
        x_new = sum_i x_i / size
    '''
    communication_op = functools.partial(dist.all_reduce)
    params_list = []
    for param in model.parameters():
        param.data.div_(float(size))
        params_list.append(param.data)

    communicate(params_list, communication_op)

def unbalanced_SyncAllreduce(models, ratios):
    params_list = []
    for model, ratio in zip(models, ratios):
        sub_params_list = []
        for param in model.parameters():
            param.data.mul_(ratio)
            sub_params_list.append(param.data)
        params_list.append(sub_params_list)

    new_list = []
    for i in range(len(params_list[0])):
        x = [j[i] for j in params_list]
        y = sum(x)
        new_list.append(y)

    for model in models:
        for param1, param2 in zip(model.parameters(), new_list):
            param1.data = param2.data.clone().detach()

def FedProx_SyncAllreduce(models, ratios, anchor_models = None):
    params_list = []
    for model, ratio in zip(models, ratios):
        sub_params_list = []
        for param in model.parameters():
            param.data.mul_(ratio)
            sub_params_list.append(param.data)
        params_list.append(sub_params_list)

    new_list = []
    for i in range(len(params_list[0])):
        x = [j[i] for j in params_list]
        y = sum(x)
        new_list.append(y)

    for model, anchor_model in zip(models, anchor_models):
        for param1, param2, param3 in zip(model.parameters(), new_list, anchor_model.parameters()):
            param1.data = param2.data.clone().detach()
            param3.data = param2.data.clone().detach()

def unbalanced_SyncAllreduce_layerwise(model, anchor_model, ratio, size, theta):
    communication_op = functools.partial(dist.all_reduce)
    ratios = []
    for param1, param2 in zip(anchor_model.parameters(), model.parameters()):
        diff = (param2.data - param1.data).view(-1)
    # diff = torch.cat(diff, dim=0)
        diff_l2 = torch.norm(diff, 2).view(1, -1).cuda()
        exp_l2 = torch.exp(diff_l2)
        ratio_att = communicate(exp_l2, communication_op, attention=True)
        ratio_att = ratio_att.view(-1)
        ratios.append(ratio_att)
    # communication_op2 = functools.partial(dist.all_reduce)
    # params_list = []
    for param, ratio1 in zip(model.parameters(), ratios):
        param.data.mul_(ratio1 * theta + ratio * (1 - theta))
        params_list = [param.data]
        communicate(params_list, communication_op)
        # params_list = []
    # anchor_model.load_state_dict(copy.deepcopy(model.state_dict()))
    # params_list2 = []
    # for param in model.buffers():
    #     param.data.mul_(ratio)
    #     params_list2.append(param.data.float())
    #     # print(param.data)
    # communicate(params_list2, communication_op)