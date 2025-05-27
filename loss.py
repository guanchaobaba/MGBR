#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class _Loss(nn.Module):
    def __init__(self, reduction='sum'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. 
        `none`: no reduction will be applied, 
        `mean`: the sum of the output will be divided by the number of elements in the output, 
        `sum`: the output will be summed. 

        Note: size_average and reduce are in the process of being deprecated, 
        and in the meantime,  specifying either of those two args will override reduction. 
        Default: `sum`
        '''
        super().__init__()
        assert(reduction == 'mean' or reduction ==
               'sum' or reduction == 'none')
        self.reduction = reduction

class BPRLoss(_Loss):
    def __init__(self, reduction='mean'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        # ensure reduction in (mean，sum, none)
        super().__init__(reduction)

    def forward(self, model_output, **kwargs):
        '''
        `model_output` (tensor) - column 0 must be the scores of positive bundles/items, column 1 must be the negative.
        '''
        pred1, pred2, L2_loss1, L2_loss2, loss3, loss4, loss5, loss6, loss7, loss8 = model_output
        # BPR loss
        loss1 = -torch.log(torch.sigmoid(pred1[:, 0] - pred1[:, 1]))
        loss2 = -torch.log(torch.sigmoid(pred2[:, 0] - pred2[:, 1]))
        # reduction
        if self.reduction == 'mean':
            loss1 = torch.mean(loss1)
            loss2 = torch.mean(loss2)
        elif self.reduction == 'sum':
            loss1 = torch.sum(loss1)
            loss2 = torch.sum(loss2)
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        
        loss1 += L2_loss1 / kwargs['batch_size'] if 'batch_size' in kwargs else 0
        loss2 += L2_loss2 / kwargs['batch_size'] if 'batch_size' in kwargs else 0

        loss = 7*loss1 + 3*loss2 + 0.5*loss3 + 0.6*loss4 + 0.4*loss5 + 0.01*loss6 + 0.01*loss7 + 0.01*loss8

        return loss

