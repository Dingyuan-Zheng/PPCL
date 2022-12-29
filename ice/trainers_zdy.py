from __future__ import print_function, absolute_import
import time
import numpy as np
import collections
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from ice.loss import CrossEntropyLabelSmooth, ViewContrastiveLoss, Weighted_CrossEntropyLabelSmooth
from .utils.meters import AverageMeter
from .evaluation_metrics import accuracy


class ImageTrainer(object):
    def __init__(self, model_1, model_1_ema, num_cluster=500, alpha=0.999, num_instance=4, tau_c=0.5, tau_v=0.09,
                 scale_kl=2.0):
        super(ImageTrainer, self).__init__()
        self.model_1 = model_1
        self.model_1_ema = model_1_ema
        self.alpha = alpha

        self.tau_c = tau_c
        self.tau_v = tau_v
        self.scale_kl = scale_kl

        self.ccloss = CrossEntropyLabelSmooth(num_cluster)
        self.vcloss = ViewContrastiveLoss(num_instance=num_instance, T=tau_v)  # ori's contri loss 1
        self.kl = nn.KLDivLoss(reduction='batchmean')                          # ori's contri loss 2
        self.wcloss = Weighted_CrossEntropyLabelSmooth(num_cluster)
        self.crosscam_epoch = 0
        self.beta = 0.07
        self.bg_knn = 50

        self.mse = nn.MSELoss(reduction='sum')

    def train(self, epoch, data_loader_target,
              optimizer, print_freq=1, train_iters=200, centers=None, intra_id_labels=None, intra_id_features=None,
              cams=None, all_pseudo_label=None):
        self.model_1.train()
        self.model_1_ema.train()
        centers = centers.cuda()
        # outliers = outliers.cuda()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ccl = AverageMeter()
        losses_vcl = AverageMeter()  # ori's loss 1
        losses_cam = AverageMeter()
        losses_kl = AverageMeter()   # ori's loss 2
        # ours
        losses_vp = AverageMeter() # virtual positive
        precisions = AverageMeter()

        self.all_img_cams = torch.tensor(cams).cuda()
        self.unique_cams = torch.unique(self.all_img_cams)
        # print(self.unique_cams)

        self.all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        self.init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        self.percam_memory = []
        self.percam_memo2update = []
        self.memory_class_mapper = []
        self.concate_intra_class = []
        self.concate_intra_cam = []
        for cc in self.unique_cams:
            percam_ind = torch.nonzero(self.all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(self.all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            self.concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            self.memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(self.init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = self.init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                self.percam_memory.append(proto_memory.detach())
                self.percam_memo2update.append(proto_memory.detach().clone()) ## zdy
                self.concate_intra_cam.append(torch.ones(proto_memory.shape[0]).long() * int(cc))
        self.concate_intra_class = torch.cat(self.concate_intra_class)
        self.concate_intra_cam = torch.cat(self.concate_intra_cam).cuda()

        if epoch >= self.crosscam_epoch:
            percam_tempV = []
            for ii in self.unique_cams:
                percam_tempV.append(self.percam_memory[ii].detach().clone())
            percam_tempV = torch.cat(percam_tempV, dim=0).cuda()

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)
            # process inputs
            inputs_1, inputs_weak, targets, inputs_2, cids = self._parse_data(target_inputs)
            b, c, h, w = inputs_1.size()

            # ids for ShuffleBN
            shuffle_ids, reverse_ids = self.get_shuffle_ids(b)

            f_out_t1 = self.model_1(inputs_1)
            f_out_t1_ema = self.model_1_ema(inputs_1)
            p_out_t1 = torch.matmul(f_out_t1, centers.transpose(1, 0)) / self.tau_c   # for instance-center similarity

            # clone f_out_t1_ema for update
            f_out_t1_ema_ = f_out_t1_ema.detach().clone()
            f_out_t1_ = f_out_t1.detach().clone()
            # our kd loss
            with torch.no_grad():
                #kd_targets = self.knowledge_ensemble(f_out_t1.detach(), p_out_t1.detach())
                kd_targets = self.knowledge_ensemble(f_out_t1_ema.detach(), p_out_t1.detach())

            kdloss = KDLoss(1.0)
            loss_kd = kdloss(p_out_t1, kd_targets)

            f_out_t2 = self.model_1(inputs_2)

            loss_cam = torch.tensor([0.]).cuda()
            loss_vp = torch.tensor([0.]).cuda()
            for cc in torch.unique(cids):
                # print(cc)
                inds = torch.nonzero(cids == cc).squeeze(-1)
                percam_targets = targets[inds]
                # print(percam_targets)
                percam_feat = f_out_t1[inds]

                # # intra-camera loss
                # mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
                # mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
                # # percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
                # percam_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(self.percam_memory[cc].t()))
                # percam_inputs /= self.beta  # similarity score before softmax
                # loss_cam += F.cross_entropy(percam_inputs, mapped_targets)

                # cross-camera loss
                if epoch >= self.crosscam_epoch:
                    associate_loss = 0
                    virtual_loss = 0
                    virtual_loss2 = 0
                    # target_inputs = percam_feat.mm(percam_tempV.t().clone())
                    target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
                    temp_sims = target_inputs.detach().clone()
                    temp_sims_multi = target_inputs.detach().clone()
                    target_inputs /= self.beta   # per-cam feat X all cam-feat  i.e. 9 (feats for cam N) X 238

                    for k in range(len(percam_feat)):
                        ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_targets[k]).squeeze(-1)
                        temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive

                        sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn-2:-2]
                        concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0) # i.e. P+50
                        concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                            torch.device('cuda'))

                        concated_target[0:len(ori_asso_ind)] = 1.0 / (len(ori_asso_ind) + 2)

                        associate_loss += -1 * (
                                F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                            0)).sum()

                        # fixed/adaptive mining strategy
                        temp_sims_score = F.softmax(temp_sims_multi[k] / 0.5, dim=0)
                        sim_score_true = temp_sims_score[ori_asso_ind]
                        ref_max = sim_score_true.max()

                        ref_min = sim_score_true.min()
                        num_fix = math.ceil((temp_sims_score >= ref_min).sum() * 0.001)

                        temp_sims_score[ori_asso_ind] = -10000.0
                        sim_cam_ind = torch.nonzero(self.concate_intra_cam == cc).squeeze(-1)
                        temp_sims_score[sim_cam_ind] = -10000.0

                        virtual_weight = torch.sort(temp_sims_score)[0][-num_fix:] / ref_max
                        virtual_weight[torch.where(virtual_weight > 1)] = 1.0
                        virtual_neg_ind = torch.sort(temp_sims[k])[1][-self.bg_knn - num_fix:-num_fix]

                        for j in range(num_fix):
                            virtual_pos_ind = torch.sort(temp_sims_score)[1][-(j + 1)]
                            concated_input_virtual = torch.cat((target_inputs[k, virtual_pos_ind].unsqueeze(-1), \
                                                                target_inputs[k, virtual_neg_ind]), dim=0)
                            concated_target_virtual = torch.zeros((len(concated_input_virtual)),
                                                                  dtype=concated_input_virtual.dtype).to(
                                torch.device('cuda'))
                            concated_target_virtual[0] = 1.0 / (len(ori_asso_ind) + num_fix)
                            virtual_loss += -1 * (
                                    F.log_softmax(concated_input_virtual.unsqueeze(0), dim=1) * virtual_weight[
                                -(j + 1)] * concated_target_virtual.unsqueeze(0)).sum()


                    # original cross-camera loss
                    loss_cam += 0.5 * associate_loss / len(percam_feat) # base loss 1 - cross-cam inter-instance loss
                    # our virtual pos loss
                    loss_vp += 0.5 * (virtual_loss + virtual_loss2) / (num_fix * len(percam_feat))


            with torch.no_grad():
                inputs_1 = inputs_1[shuffle_ids]
                #f_out_t1_ema = self.model_1_ema(inputs_1)
                #f_out_t1_ema = f_out_t1_ema[reverse_ids]

                inputs_2 = inputs_2[shuffle_ids]
                f_out_t2_ema = self.model_1_ema(inputs_2)
                f_out_t2_ema = f_out_t2_ema[reverse_ids]

                inputs_weak = inputs_weak[shuffle_ids]
                f_out_weak_ema = self.model_1_ema(inputs_weak)
                f_out_weak_ema = f_out_weak_ema[reverse_ids]

            loss_ccl = self.ccloss(p_out_t1, targets)    # base loss 2 - instance-center contrast

            # ori's other losses
            loss_vcl = self.vcloss(F.normalize(f_out_t1), F.normalize(f_out_t2_ema), targets)

            loss_kl = self.kl(F.softmax(
                torch.matmul(F.normalize(f_out_t1), F.normalize(f_out_t2_ema).transpose(1, 0)) / self.scale_kl,
                dim=1).log(),
                              F.softmax(torch.matmul(F.normalize(f_out_weak_ema),
                                                     F.normalize(f_out_weak_ema).transpose(1, 0)) / self.scale_kl,
                                        dim=1)) * 10

            loss = loss_ccl + loss_cam + 0.9 * loss_vp + 1.0 * loss_vcl + 1.0 * loss_kl
            #loss = loss_ccl + loss_cam # base loss: instance - center + cross - camera

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch * len(data_loader_target) + i)
            ###
            self._update_cluster_center(centers, f_out_t1_, targets, epoch)
            for cc in torch.unique(cids):
                # print(cc)
                inds = torch.nonzero(cids == cc).squeeze(-1)
                percam_targets = targets[inds]
                self._update_percam_memory(self.memory_class_mapper, self.percam_memo2update, f_out_t1_[inds], percam_targets, cids[inds], epoch)
            percam_tempV_ = []
            for ii in self.unique_cams:
                percam_tempV_.append(self.percam_memo2update[ii])
            percam_tempV = torch.cat(percam_tempV_, dim=0).cuda()
            ####

            prec_1, = accuracy(p_out_t1.data, targets.data)

            losses_ccl.update(loss_ccl.item())
            losses_cam.update(loss_cam.item())
            losses_vcl.update(loss_vcl.item())
            losses_kl.update(loss_kl.item())

            losses_vp.update(loss_vp.item())

            precisions.update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ccl {:.3f}\t'
                      'Loss_vcl {:.3f}\t'
                      'Loss_cam {:.3f}\t'
                      'Loss_kl {:.3f}\t'
                      'Loss_vp {:.3f}\t'
                      'Prec {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ccl.avg,
                              losses_vcl.avg,
                              losses_cam.avg,
                              losses_kl.avg,
                              losses_vp.avg,
                              precisions.avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def _update_cluster_center(self, centers, inputs, targets, epoch):
        for x, y in zip(inputs, targets):
            centers[y] = (self.alpha) * centers[y] + (1 - self.alpha) * x
            #centers[y] = (0.99) * centers[y] + (0.01) * x
            centers[y] /= centers[y].norm()

    def _update_percam_memory(self, mapper, percam_memory, inputs, targets, cids, epoch):
        for i in range(inputs.shape[0]):
            percam_memory[cids[i]][mapper[cids[i]][int(targets[i])]] = (self.alpha) * percam_memory[cids[i]][mapper[cids[i]][int(targets[i])]] + \
                                                                (1-self.alpha) * inputs[i]
            percam_memory[cids[i]][mapper[cids[i]][int(targets[i])]] /= (percam_memory[cids[i]][mapper[cids[i]][int(targets[i])]]).norm()

    def _parse_data(self, inputs):
        imgs_1, imgs_2, img_mutual, pids, cids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        inputs_mutual = img_mutual.cuda()
        targets = pids.cuda()
        cids = cids.cuda()
        return inputs_1, inputs_2, targets, inputs_mutual, cids

    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for ShuffleBN"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    # kd_targets = self.knowledge_ensemble(f_out_t1.detach(), p_out_t1.detach())
    def knowledge_ensemble(self, feats, logits, temp=0.5, omega=0.5):
        batch_size = logits.size(0)
        masks = torch.eye(batch_size)
        masks = masks.cuda()
        feats = nn.functional.normalize(feats, p=2, dim=1)
        #logits = nn.functional.softmax(logits / temp, dim=1)
        W = torch.matmul(feats, feats.permute(1, 0)) - masks * 1e9
        W = F.softmax(W, dim=1)
        W = (1 - omega) * torch.inverse(masks - omega * W)
        logits = nn.functional.softmax(logits / temp, dim=1)
        return torch.matmul(W, logits)

#kdloss = KDLoss(1.0)
#loss_kd = kdloss(p_out_t1, kd_targets)

class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")
        #self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, target)*0.1*(self.temp_factor**2)/input.size(0)
        return loss