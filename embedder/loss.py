from collections import UserDict
from typing import Iterable, Dict, List

import torch
from torch import nn, Tensor

from embedder.grad_cache import GradCache


def cos_sim(a: Tensor, b: Tensor, pair_wise=False):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)
        
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    if pair_wise:
        result = torch.sum(a_norm * b_norm, dim=1)
    else:
        result = torch.mm(a_norm, b_norm.transpose(0, 1))

    return result

def simcse(
    single_reps: List[torch.Tensor],
    multiple_reps_pos: List[torch.Tensor],
    multiple_reps_neg: List[torch.Tensor],
    similarity_fct=cos_sim,
    scale=50.0
) -> torch.Tensor:
    positive_scores = []

    for embedding_a, embedding_c in zip(single_reps, multiple_reps_pos):
        positive_scores.append((similarity_fct(embedding_a, embedding_c)).squeeze() * scale)
    
    negative_scores = []
    for embedding_a, embedding_c in zip(single_reps, multiple_reps_neg):
        negative_scores.append((similarity_fct(embedding_a, embedding_c)).squeeze() * scale)

    loss = 0
    for positive_score, negative_score in zip(positive_scores, negative_scores):
        positive_score = torch.exp(positive_score)
        negative_score = torch.exp(negative_score)
        prob = torch.sum(positive_score) / (torch.sum(positive_score) + torch.sum(negative_score))
        loss += -torch.log(prob)
    loss /= len(positive_scores)
    
    return loss
    
class SimCSE(nn.Module):
    def __init__(self, model, scale: float = 50.0, similarity_fct = cos_sim):
        super(SimCSE, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct

    def forward(self, sentence_features):
        single_features = sentence_features[0]
        multiple_features_pos = sentence_features[1]
        multiple_features_neg = sentence_features[2]

        single_reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in single_features]

        multiple_reps_pos = []
        for multiple_sentence_feature in multiple_features_pos:
            features_list = []
            for feature in multiple_sentence_feature[1]['text']:
                features_list.append(('document', {'text': feature}))
            multiple_reps_pos.append(self.model(features_list)['sentence_embedding'])
            
        multiple_reps_neg = []
        for multiple_sentence_feature in multiple_features_neg:
            features_list = []
            for feature in multiple_sentence_feature[1]['text']:
                features_list.append(('document', {'text': feature}))
            multiple_reps_neg.append(self.model(features_list)['sentence_embedding'])

        loss = simcse(single_reps, multiple_reps_pos, multiple_reps_neg, self.similarity_fct, self.scale)
        return loss

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}


def mismatched_sizes_all_gather(tensor: Tensor, group=None, async_op=False):
    # all_gather doesn't support tensor lists where the first dimension is mismatched. This does.
    assert torch.distributed.is_initialized(), "torch.distributed not initialized"
    world_size = torch.distributed.get_world_size()
    # let's get the sizes for everyone
    dim_0_size = torch.tensor([tensor.shape[0]], dtype=torch.int64, device="cuda")
    sizes = [torch.zeros_like(dim_0_size) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, dim_0_size, group=group, async_op=async_op)
    sizes = torch.cat(sizes).cpu().tolist()
    # now pad to the max dim-0 size
    max_size = max(sizes)
    padded = torch.zeros((max_size, *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype)
    padded[:tensor.shape[0], :] = tensor
    # gather the padded tensors
    tensor_list = [torch.zeros(padded.shape, device=padded.device, dtype=padded.dtype) for _ in range(world_size)]
    all_gather_with_grad(tensor_list, padded, group, async_op)
    # trim off the padding
    for rank in range(world_size):
        assert not tensor_list[rank][sizes[rank]:, :].count_nonzero().is_nonzero(), \
            "This would remove non-padding information"
        tensor_list[rank] = tensor_list[rank][:sizes[rank], :]
    return tensor_list


# from https://github.com/vlkit/vlkit/blob/master/vlkit/ops/distributed.py
class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """
    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        torch.distributed.all_gather(tensor_list, tensor, group=group, async_op=async_op)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = torch.distributed.get_rank()

        dist_ops = [
            torch.distributed.reduce(grad_list[i], i, async_op=True) for i in range(torch.distributed.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None


all_gather_with_grad = AllGather.apply
