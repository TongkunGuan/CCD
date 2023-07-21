from fastai.vision import *


class MultiCELosses(nn.Module):
    def __init__(self, reduction="batchmean", kl_div=False):
        super().__init__()
        self.ce = SoftCrossEntropyLoss(reduction=reduction, kl_div=kl_div)

    @property
    def last_losses(self):
        return self.losses

    @staticmethod
    def _flatten(sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    @staticmethod
    def _merge_list(all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res

        def merge(items):
            if isinstance(items[0], torch.Tensor):
                return torch.cat(items, dim=0)
            else:
                return items[0]

        res = dict()
        for key in all_res[0].keys():
            items = [r[key] for r in all_res]
            res[key] = merge(items)
        return res

    def _ce_loss(self, output, gt_labels, gt_lengths, record=True, mask=None):
        loss_name = output.get('name')
        pt_logits = output['logits']

        assert pt_logits.shape[0] % gt_labels.shape[0] == 0
        iter_size = pt_logits.shape[0] // gt_labels.shape[0]
        if iter_size > 1:
            gt_labels = gt_labels.repeat(3, 1, 1)
            gt_lengths = gt_lengths.repeat(3)
        flat_gt_labels = self._flatten(gt_labels, gt_lengths)
        flat_pt_logits = self._flatten(pt_logits, gt_lengths)

        if mask is not None:
            if iter_size > 1:
                mask = mask.repeat(3, 1)
            mask = self._flatten(mask, gt_lengths)

        loss = self.ce(flat_pt_logits, flat_gt_labels, mask=mask)
        if record and loss_name is not None:
            self.losses[f'{loss_name}_loss'] = loss

        return loss

    def forward(self, outputs, gt_labels, gt_lengths, record=False, mask=None):
        self.losses = {}
        if isinstance(outputs, (tuple, list)):
            outputs = [self._merge_list(o) for o in outputs]
            return sum([self._ce_loss(o, gt_labels, gt_lengths, mask=mask) for o in outputs if o['loss_weight'] > 0.])
        else:
            return self._ce_loss(outputs, gt_labels, gt_lengths, record=record, mask=mask)


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="batchmean", apply_softmax=True, kl_div=False):
        super().__init__()
        self.reduction = reduction
        self.apply_softmax = apply_softmax
        self.kl_div = kl_div

    def forward(self, input, target, mask=None, eps=1e-12):
        if self.apply_softmax:
            log_prob = F.log_softmax(input, dim=-1)
        else:
            log_prob = torch.log(input)
        if not self.kl_div:  # cross entropy loss
            loss = - target * log_prob
        else:  # KL divergence: F.kl_div(log_prob, target, reduction=self.reduction)
            loss = target * torch.log(target + eps) - target * log_prob
        loss = loss.sum(dim=-1)
        if mask is not None:
            loss = mask * loss
        if self.reduction == "batchmean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise NotImplementedError(f'reduction={self.reduction} is not implemented for CE loss')
        return loss
