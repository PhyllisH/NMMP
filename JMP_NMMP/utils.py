import numpy as np
import os
import sys
import ipdb
import json
import torch
import copy
import random


def load_json(json_path):
    samples = None
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            samples = json.load(f)
    return samples

def dis(actor_centers, moving_actors, dis_thre):
    '''
    actor_centers: [N_actors, 2]
    output: [N_actors, N_actors]
    '''
    old_moving_actors = [-1] + copy.deepcopy(moving_actors)
    moving_actors = []
    for i in old_moving_actors:
        if i < (actor_centers.shape[0]-1):
            moving_actors.append(i+1)
    dis_mat = np.zeros([actor_centers.shape[0], actor_centers.shape[0]]) + dis_thre + 1
    for i in moving_actors:
        for j in range(actor_centers.shape[0]):
            # if j == i:
            #     continue
            diff = actor_centers[i] - actor_centers[j]
            dis_mat[i,j] = np.sqrt(np.sum(diff ** 2))
    for i in range(actor_centers.shape[0]):
        for j in moving_actors:
            # if j == i:
            #     continue
            diff = actor_centers[i] - actor_centers[j]
            dis_mat[i,j] = np.sqrt(np.sum(diff ** 2))
    bin_dis_mat = (dis_mat < dis_thre)*1
    return bin_dis_mat


def compute_loss(loss_fn, pred, target, moving_mask=None):
    '''
    Function: Compute loss
    param loss_fn: loss function, mse/ce
    param pred: [batch_size, pred_steps*2]
    param target: [batch_size, pred_steps*2]
    output: scalar
    '''
    flat_loss = loss_fn(pred, target)    #[b, pred_steps*2]
    mask = torch.where(target>0, torch.ones(target.shape).cuda(), torch.zeros(target.shape).cuda())    #[b, pred_steps*2]
    if moving_mask is not None:
        mask = mask * moving_mask.expand(moving_mask.shape[0], mask.shape[-1])
    # treat all the points equally
    if torch.sum(mask) <= 0:
        loss = torch.sum(flat_loss*mask)
    else:
        loss = torch.sum(flat_loss*mask)/torch.sum(mask)

    # add more weight to the badly predicted samples
    # weight = 2 * torch.sigmoid(flat_loss) - 1
    # loss = torch.sum(flat_loss*mask*weight)/torch.sum(mask)
    return loss

def encode_onehot(labels, N_classes):
    classes = set(labels)
    classes_dict = {c: np.identity(N_classes)[i, :] for i, c in
                        enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                        dtype=np.int32)
    return labels_onehot

def rel_mat(N_actors):
    off_diag = np.ones([N_actors, N_actors]) - np.eye(N_actors)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    return rel_rec, rel_send

def displacement_error(pred, target, moving_masks=None, size_average=True):
    '''
    Function: d=((x-x')^2+(y-y')^2)^(1/2)
    Calculate the Euclidean distance between prediction and GT.
    param: pred [N_sample, pred_steps, 2] or [N_sample, N_actors, pred_steps, 2]
           moving_masks [N_sample, N_actors]
    '''
    assert pred.shape == target.shape
    # pred = np.reshape(pred, [-1, pred.shape[-2], pred.shape[-1]])
    # target = np.reshape(target, [-1, target.shape[-2], target.shape[-1]])
    mask = (target >= 0)*1
    diff = (pred - target) * mask
    diff = diff**2
    dis = np.sqrt(np.sum(diff, -1))    #[N_sample, pred_steps]
    mask = np.sum(mask, -1)/2    #[N_sample, pred_steps]
    if size_average:
        if moving_masks is None:
            return np.sum(dis)/np.sum(mask)
        else:
            dis = np.sum(dis, -1)
            mask = np.sum(mask, -1)
            mask = moving_masks * mask
            return np.sum((mask>0)*1.0*dis)/np.sum(mask)
    elif len(dis.shape) <= 2:
        loss_mask = np.sum(mask, -1) #[N_sample]
        loss_mask = (loss_mask > 0)*1
        return (np.sum(dis, -1)/(np.sum(mask, -1) + 1e-6))*loss_mask
    else:
        dis = np.reshape(dis, [dis.shape[0], -1])    #[N_sample*N_actors, pred_step]
        mask = np.reshape(mask, [mask.shape[0], -1])
        loss_mask = (np.sum(mask, -1) > 0)*1
        return (np.sum(dis, -1)/(np.sum(mask, -1) + 1e-6))*loss_mask        

def final_displacement_error(pred, target, moving_masks=None, size_average=True):
    '''
    Function: d=((x-x')^2+(y-y')^2)^(1/2)
    Calculate the Euclidean distance between prediction and GT.
    param: pred [N_sample, pred_steps, 2] or [N_sample, N_actors, pred_steps, 2]
           moving_masks [N_sample, N_actors]
    '''
    assert pred.shape == target.shape
    # pred = np.reshape(pred, [-1, pred.shape[-2], pred.shape[-1]])
    # target = np.reshape(target, [-1, target.shape[-2], target.shape[-1]])
    mask = (target >= 0)*1
    diff = (pred - target) * mask
    diff = diff**2
    dis = np.sqrt(np.sum(diff, -1))[:,:,-1]    #[N_sample, 1]
    mask = np.sum(mask, -1)/2   #[N_sample, pred_steps]
    mask = mask[:, :, -1]
    if size_average:
        if moving_masks is None:
            return np.sum(dis)/np.sum(mask)
        else:
            mask = moving_masks * mask
            return np.sum((mask>0)*1.0*dis)/np.sum(mask)
    elif len(dis.shape) <= 2:
        loss_mask = np.sum(mask, -1) #[N_sample]
        loss_mask = (loss_mask > 0)*1
        return (np.sum(dis, -1)/(np.sum(mask, -1) + 1e-6))*loss_mask
    else:
        dis = np.reshape(dis, [dis.shape[0], -1])    #[N_sample*N_actors, pred_step]
        mask = np.reshape(mask, [mask.shape[0], -1])
        loss_mask = (np.sum(mask, -1) > 0)*1
        return (np.sum(dis, -1)/(np.sum(mask, -1) + 1e-6))*loss_mask

def along_track_error(pred, target):
    pass

def cross_track_error(pred, target):
    pass

def bce_loss(input, target, mask):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    loss = loss * mask
    return loss.sum()/mask.sum()

def gan_d_loss(scores_real, scores_fake, mask):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples
    - mask: Tensor of shape (N, ) whether the sample counts

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real, mask)
    loss_fake = bce_loss(scores_fake, y_fake, mask)
    return loss_real + loss_fake

