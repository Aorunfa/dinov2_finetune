# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.distributed as dist


logger = logging.getLogger("dinov2")


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())                       # 计算相似程度
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)             # 自己与自己计算的相似度填充为-1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741     # I: index
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):     # 打开全精度
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)  # student_output.pow(2).sum(-1) --> 1
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741           # 给student_output每一个向量匹配一个除自己外最相似的向量
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()                  # 使两个global输出分布尽可能相似
        return loss
