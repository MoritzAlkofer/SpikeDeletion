import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedKLDivWithLogitsLoss(nn.KLDivLoss):
    def __init__(self, weight):
        super(WeightedKLDivWithLogitsLoss, self).__init__(size_average=None, reduce=None, reduction='none')
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        # TODO: For KLDivLoss: input should 'log-probability' and target should be 'probability'
        # TODO: input for this method is logits, and target is probabilities
        batch_size = input.size(0)
        log_prob = F.log_softmax(input, 1)
        element_loss = super(WeightedKLDivWithLogitsLoss, self).forward(log_prob, target)

        sample_loss = torch.sum(element_loss, dim=1)
        sample_weight = torch.sum(target * self.weight, dim=1)

        weighted_loss = sample_loss * sample_weight
        # Average over mini-batch, not element-wise
        avg_loss = torch.sum(weighted_loss) / batch_size

        return avg_loss
    # alpha=.25, gamma=2 SOTA , modify the alpha
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=.25, gamma=2):
            super(WeightedFocalLoss, self).__init__()        
            self.alpha = torch.tensor([alpha, 1-alpha]).cuda()        
            self.gamma = gamma
            
    def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')   
            #BCE_loss = F.cross_entropy(inputs, targets, reduction='none')      
            targets = targets.type(torch.long)        
            at = self.alpha.gather(0, targets.data.view(-1))        
            pt = torch.exp(-BCE_loss)        
            F_loss = at*(1-pt)**self.gamma * BCE_loss        
            return F_loss.mean()


class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=.75):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        # split to n bins
        return torch.floor(g * (self._bins - 0.0001)).long()


    def forward(self, x, target):
        # compute value g
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            # 计算落入bins的梯度模长数量
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd  # 计算好样本的gd值

        # 借由binary_cross_entropy_with_logits,gd值当作参数传入
        return F.binary_cross_entropy_with_logits(x, target, weight=beta[bin_idx])