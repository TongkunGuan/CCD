import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

class SegLoss(nn.Module):
    def __init__(self, loss_seg=False):
        super(SegLoss, self).__init__()
        self.num_classes = 1
        self.loss_seg = loss_seg
        self.gts_loss = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        self.m = nn.Sigmoid()

    def cross_entropy(self, global_text_segs, gts_masks, bool_):
        if global_text_segs.shape[-1] == gts_masks.shape[-1]:
            global_mask = gts_masks
        else:
            global_mask = nn.functional.interpolate(gts_masks.float().unsqueeze(1),
                                                    size=(global_text_segs.shape[2], global_text_segs.shape[3]),
                                                    scale_factor=None, mode='bilinear', align_corners=None)
            global_mask = (global_mask >= 0.5)
        input_global_masks = global_mask.view([-1]).long()
        pred_masks = global_text_segs.permute(0, 2, 3, 1).contiguous().view([-1, 2])
        loss = F.cross_entropy(pred_masks, input_global_masks, reduce=bool_)
        return loss

    def forward(self, seg_mask, gts_masks, bool_):
        if isinstance(seg_mask, list):
            cel_loss = self.cross_entropy(seg_mask[0], gts_masks) + self.cross_entropy(seg_mask[1], gts_masks)
        else:
            cel_loss = self.cross_entropy(seg_mask, gts_masks, bool_)
        return cel_loss

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # self.register_buffer("global_center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.seg_loss = SegLoss()
        self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)

    @property
    def last_losses(self):
        return self.losses

    def forward(self, student_output, teacher_output, epoch):
        self.losses = {}
        total_loss = 0.
        predmask_loss = 0.
        mask = torch.cat(student_output['gt'])
        learn_mask_0 = student_output['mask']
        backfore_softmax = F.softmax(learn_mask_0, dim=1)
        predmask_loss += self.seg_loss(backfore_softmax, mask.contiguous(), True)
        self.losses[f'mask_loss'] = predmask_loss
        total_loss += predmask_loss

        # index = student_output['index']
        # B = index.shape[0]
        # length = torch.clamp(index.sum(1), 3, 26).unsqueeze(-1)[:B//2]
        # grid = torch.arange(0, 26, device=length.device).unsqueeze(0)
        # new_index = grid <= length
        # student_l_output = torch.cat([student_output['instances_view'][:B//2][new_index], student_output['instances_view'][B//2:][new_index]])
        # teacher_l_output = torch.cat([teacher_output['instances_view'][:B//2][new_index], teacher_output['instances_view'][B//2:][new_index]])

        """
            Character: Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_l_output = student_output['instances_view']
        teacher_l_output = teacher_output['instances_view']
        student_out = student_l_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        # teacher_out = self.sinkhorn_knopp_teacher(teacher_l_output, temp)
        teacher_out = F.softmax((teacher_l_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        dino_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                dino_loss += loss.mean()
                n_loss_terms += 1
        dino_loss /= n_loss_terms
        self.update_center(teacher_l_output)
        self.losses[f'Dino_loss'] = dino_loss
        total_loss += dino_loss

        """
            Global: Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # student_g_out = student_output['view'].squeeze(1)
        # teacher_g_out = teacher_output['view'].squeeze(1)
        # student_out = student_g_out / self.student_temp
        # student_out = student_out.chunk(self.ncrops)
        # teacher_out = F.softmax((teacher_g_out - self.global_center) / temp, dim=-1)
        # teacher_out = teacher_out.detach().chunk(2)
        # dino_global_loss = 0
        # n_loss_terms = 0
        # for iq, q in enumerate(teacher_out):
        #     for v in range(len(student_out)):
        #         if v == iq:
        #             # we skip cases where student and teacher operate on the same view
        #             continue
        #         loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        #         dino_global_loss += loss.mean()
        #         n_loss_terms += 1
        # dino_global_loss /= n_loss_terms
        # self.update_global_center(teacher_g_out)
        # self.losses[f'Dino_global_loss'] = dino_global_loss
        # total_loss += dino_global_loss
        return total_loss


    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def update_global_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.global_center = self.global_center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()