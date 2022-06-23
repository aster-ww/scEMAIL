import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Beta
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scEMAIL_utils import ZINBLoss, MeanAct, DispAct,dip, calculate_bimodality_coefficient
import numpy as np
from time import time
import scanpy as sc
import csv
import os

np.set_printoptions(threshold=np.inf)

def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        net.append(nn.BatchNorm1d(layers[i], affine=True))
    return nn.Sequential(*net)


def feat_classifier(class_num, bottleneck_dim=32):
    net = []
    net.append(nn.Linear(bottleneck_dim, class_num))
    net.append(nn.BatchNorm1d(class_num, affine=True))
    return nn.Sequential(*net)


def mixup(x_batch):
    m = Beta(torch.tensor([1.0]), torch.tensor([1.0]))
    x_batch_expand = x_batch.unsqueeze(1)
    weight = m.sample([x_batch_expand.size()[0], x_batch_expand.size()[1]])
    x_batch_mix = weight * x_batch + (1-weight) * x_batch_expand
    x_batch_mix = x_batch_mix.view([-1, x_batch_mix.size()[-1]])
    index = torch.arange(0, x_batch_mix.size()[0])
    x_batch_mix = x_batch_mix[index % (x_batch.size()[0] + 1) != 0]
    return x_batch_mix

def ensemble_score(logit_tensor,logit_tensor_1, logit_tensor_2):
    label_pred_tensor = nn.Softmax(-1)(logit_tensor)
    label_pred_tensor_1 = nn.Softmax(-1)(logit_tensor_1)
    label_pred_tensor_2 = nn.Softmax(-1)(logit_tensor_2)
    consistency =  (label_pred_tensor_1 * label_pred_tensor_2).sum(-1)
    softmax, _ = torch.max(label_pred_tensor, 1)
    entropy = -(label_pred_tensor * torch.log(label_pred_tensor)).sum(-1)
    entropy = 1-entropy/np.log(label_pred_tensor.size()[1])
    ensemble = (entropy+consistency+softmax)/3
    return ensemble

def cal_pseudo_label(score_bank,fea_bank,output_f_):
    class_center = torch.mm(score_bank.T.cpu(), fea_bank) / (
         score_bank.T.sum(-1).view(-1, 1).cpu())
    dist = fea_bank @ class_center.T
    _, pseudo_label_total = torch.max(dist, -1)
    pseudo_label_total_onehot = torch.zeros(score_bank.size()).scatter_(1, pseudo_label_total.unsqueeze(1).cpu(), 1)
    class_center = torch.mm(pseudo_label_total_onehot.T, fea_bank) / (
            pseudo_label_total_onehot.T.sum(-1).view(-1, 1) + 1e-5)
    dist = output_f_ @ class_center.T
    _, pseudo_label_batch = torch.max(dist, dim=1)
    return pseudo_label_batch.cuda(),class_center

class target_model(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters,encodeLayer=[], decodeLayer=[],
                 activation="relu", sigma=1.):
        super(target_model, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.encoder = buildNetwork([input_dim] + encodeLayer,activation=activation)
        self.decoder = buildNetwork([z_dim] + decodeLayer,  activation=activation)
        self.classifier = feat_classifier(n_clusters)
        self.add_classifier_1 = feat_classifier(n_clusters)
        self.add_classifier_2 = feat_classifier(n_clusters)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        self.zinb_loss = ZINBLoss().cuda()
        for name, param in self.named_parameters():
            if 'classifier' in  name:
                param.requires_grad = False
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)
        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        label = self.classifier(z0)
        label_1 = self.add_classifier_1(z0)
        label_2 = self.add_classifier_2(z0)

        return z0, label, label_1, label_2, _mean, _disp, _pi

    def fit(self, x,annotation, X_raw, size_factor, pretrain_epoch=10,midtrain_epoch=20, K=5, KK=5, alpha=0.1, batch_size=256, lr=0.001, epochs=500,
            error=0.001,pseudo_option=True):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)

        num_sample = len(x)
        print("number of samples:", num_sample)
        print("number of class:", self.n_clusters)
        fea_bank = torch.randn(num_sample, self.z_dim)
        score_bank = torch.randn(num_sample, self.n_clusters).cuda()
        ensemble_total = torch.randn(num_sample)
        _,pseudo_bank = score_bank.max(dim=-1)

        with torch.no_grad():
            iter_test = iter(dataloader)
            for i in range(len(dataloader)):
                indx = iter_test._next_index()
                data = dataset[indx]
                x_batch = data[0]
                x_tensor = Variable(x_batch).cuda()
                latent_tensor, logit_tensor,logit_tensor_1, logit_tensor_2, _, _, _ = self.forward(x_tensor)
                latent_tensor_norm = F.normalize(latent_tensor)
                label_pred_tensor = nn.Softmax(-1)(logit_tensor)
                ensemble_batch = ensemble_score(logit_tensor.cpu(), logit_tensor_1.cpu(),
                                                   logit_tensor_2.cpu())
                fea_bank[indx] = latent_tensor_norm.detach().clone().cpu()
                score_bank[indx] = label_pred_tensor.detach().clone()
                ensemble_total[indx] = ensemble_batch.detach().clone().cpu()
            _, last_pred = torch.max(score_bank, 1)
            y_pred = last_pred
            X_total_tensor = torch.Tensor(x)
            X_total_tensor = Variable(X_total_tensor).cuda()
            latent_total_tensor, logit_total_tensor,logit_total_tensor_1, logit_total_tensor_2, _, _, _ = self.forward(X_total_tensor)
            dip_test = dip(ensemble_total.cpu().detach().numpy())
            dip_p_value=dip_test[-1]
            dip_test=dip_test[-2]
            ensemble_total_scale = (ensemble_total).sqrt().cpu().detach().numpy()
            BC = calculate_bimodality_coefficient(ensemble_total_scale)
            print("bimodality of dip test:",dip_p_value, not dip_test)
            print("bimodality coefficient:(>0.555 indicates bimodality)", BC,BC>0.555)
            bimodality = (not dip_test) or (BC > 0.555)
            print("ood sample exists:",bimodality)
            if bimodality:
                if len(x) > 500:
                    idx_rand = torch.randperm(len(x))[:500]
                    latent_total_tensor = latent_total_tensor[idx_rand]
                latent_total_tensor_mix = mixup(latent_total_tensor.cpu())
                latent_total_tensor_mix = latent_total_tensor_mix.cuda()
                logit_total_tensor_mix = self.classifier(latent_total_tensor_mix)
                logit_total_tensor_1_mix = self.add_classifier_1(latent_total_tensor_mix)
                logit_total_tensor_2_mix = self.add_classifier_2(latent_total_tensor_mix)
                ensemble_threshold = ensemble_score(logit_total_tensor_mix.cpu(),
                                                       logit_total_tensor_1_mix.cpu(),
                                                       logit_total_tensor_2_mix.cpu())
                ensemble_threshold = ensemble_threshold.mean()

        for epoch in range(epochs):
            batch_idx = 0
            iter_test = iter(dataloader)
            for i in range(len(dataloader)):
                indx = iter_test._next_index()
                data = dataset[indx]
                x_batch = data[0]
                x_raw_batch = data[1]
                sf_batch = data[2]
                x_tensor = Variable(x_batch).cuda()
                x_raw_tensor = Variable(x_raw_batch).cuda()
                sf_tensor = Variable(sf_batch).cuda()
                latent_batch_tensor, logit_batch_tensor, logit_batch_tensor_1, logit_batch_tensor_2, \
                mean_tensor, disp_tensor, pi_tensor = self.forward(x_tensor)
                label_pred_tensor = nn.Softmax(-1)(logit_batch_tensor)
                ensemble_batch = ensemble_score(logit_batch_tensor.cpu(),
                                                   logit_batch_tensor_1.cpu(),
                                                   logit_batch_tensor_2.cpu())
                ensemble_batch = ensemble_batch.cuda()
                if bimodality:
                    idx_rand = torch.randperm(len(x_batch))
                    latent_batch_tensor_sub = latent_batch_tensor[idx_rand]
                    latent_batch_tensor_mix = mixup(latent_batch_tensor_sub.cpu())
                    latent_batch_tensor_mix = latent_batch_tensor_mix.cuda()
                    logit_batch_tensor_mix = self.classifier(latent_batch_tensor_mix)
                    logit_batch_tensor_1_mix = self.add_classifier_1(latent_batch_tensor_mix)
                    logit_batch_tensor_2_mix = self.add_classifier_2(latent_batch_tensor_mix)
                    ensemble_threshold = ensemble_score(logit_batch_tensor_mix.cpu(),
                                                           logit_batch_tensor_1_mix.cpu(),
                                                           logit_batch_tensor_2_mix.cpu())
                    ensemble_threshold = ensemble_threshold.mean()
                with torch.no_grad():
                    output_f_norm = F.normalize(latent_batch_tensor)
                    output_f_ = output_f_norm.cpu().detach().clone()
                    fea_bank[indx] = output_f_.detach().clone().cpu()
                    score_bank[indx] = label_pred_tensor.detach().clone()
                    ensemble_total[indx] = ensemble_batch.detach().clone().cpu()
                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,dim=-1, largest=True,k=K + 1)
                    idx_near = idx_near[:, 1:]
                    score_near = score_bank[idx_near]
                    fea_near = fea_bank[idx_near]
                    fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1,-1)
                    distance_ = torch.bmm(fea_near,fea_bank_re.permute(0, 2, 1))
                    _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                                  k=KK + 1)
                    idx_near_near = idx_near_near[:, :, 1:]
                    indx_Tensor = torch.Tensor(indx)
                    indx_ = indx_Tensor.unsqueeze(-1).unsqueeze(-1)
                    match = (idx_near_near == indx_).sum(-1).float()
                    weight = torch.where(
                        match > 0., match,
                        torch.ones_like(match).fill_(
                             alpha) )
                    weight_kk = weight.detach().clone().unsqueeze(-1).expand(-1, -1,KK)
                    weight_kk = weight_kk.fill_(alpha)
                    score_near_kk = score_bank[idx_near_near]
                    score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1, self.n_clusters)
                    score_self = score_bank[indx]
                    pseudo_label_batch,class_center= cal_pseudo_label(score_bank, fea_bank,output_f_)
                    pseudo_bank[indx]=pseudo_label_batch
                    pseudo_label_onehot = torch.zeros(
                        [latent_batch_tensor.size()[0], score_bank.size()[1]]).scatter_(1,
                        pseudo_label_batch.unsqueeze(1).cpu(), 1).cuda()
                    if bimodality:
                        ensemble_near = ensemble_total[idx_near]
                        weight = torch.where(ensemble_near > ensemble_threshold.cpu(), weight,torch.ones_like(weight).fill_(0.0))
                        ensemble_near_near = ensemble_total[idx_near_near]
                        weight_kk = torch.where(ensemble_near_near > ensemble_threshold.cpu(), weight_kk,
                                             torch.ones_like(weight_kk).fill_(0.0))
                        weight_kk = weight_kk.contiguous().view(weight_kk.shape[0], -1)
                    else:
                        weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],-1)

                zinb_loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                           scale_factor=sf_tensor)
                softmax_out_un = label_pred_tensor.unsqueeze(1).expand(-1, K, -1)
                output_re = label_pred_tensor.unsqueeze(1).expand(-1, K * KK, -1)
                if bimodality:
                    neighbor_loss = -torch.mean(((softmax_out_un[ensemble_batch > ensemble_threshold] *
                                                  score_near[ensemble_batch > ensemble_threshold]).sum(-1) *
                                                 weight[ensemble_batch > ensemble_threshold].cuda()).sum(1))
                    neighbor_loss2 = -torch.mean(((output_re[ensemble_batch > ensemble_threshold] *
                                                   score_near_kk[ensemble_batch > ensemble_threshold]).sum(-1) *
                                                  weight_kk[ensemble_batch > ensemble_threshold].cuda()).sum(1))

                    self_loss = -torch.mean(((label_pred_tensor[ensemble_batch > ensemble_threshold] *
                                              score_self[ensemble_batch > ensemble_threshold]).sum(-1)))
                    if epoch < pretrain_epoch:
                        optimizer.zero_grad()
                        (zinb_loss).backward()
                        optimizer.step()
                        print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'
                              .format(batch_idx + 1, epoch + 1, zinb_loss.item()))

                    elif epoch < midtrain_epoch:
                        mid_loss = 1 * zinb_loss + 1 * neighbor_loss + 1 * neighbor_loss2 + 1 * self_loss
                        optimizer.zero_grad()
                        mid_loss.backward()
                        optimizer.step()
                        print('Midtrain epoch [{}/{}], ZINB loss:{:.4f}, '
                            ' neighbor loss 1:{:.4f}, expanded neighbor loss 1:{:.4f}, self loss:{:.4f}'
                                .format(batch_idx + 1, epoch + 1, zinb_loss.item()
                                        , neighbor_loss.item(), neighbor_loss2.item(), self_loss.item()))
                    else:
                        if pseudo_option == True:
                            pseudo_loss = -torch.mean(
                                ((pseudo_label_onehot[(ensemble_batch > ensemble_threshold)].cuda() *
                                  torch.log(label_pred_tensor[(ensemble_batch > ensemble_threshold)])).sum(-1)))

                            loss = 1 * zinb_loss + 1 * neighbor_loss \
                                   + 1 * self_loss + 1 * neighbor_loss2 + 1 * pseudo_loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            print('Funetrain epoch [{}/{}], ZINB loss:{:.4f}, '
                                ' neighbor loss:{:.4f}, expanded neighbor loss:{:.4f}, self loss:{:.4f},'
                                ' pseudo loss:{:.4f}'
                                    .format(batch_idx + 1, epoch + 1, zinb_loss.item()
                                            , neighbor_loss.item(), neighbor_loss2.item(), self_loss.item(),
                                            pseudo_loss.item()))
                        else:
                            center_loss = -torch.mean((output_f_ * class_center[pseudo_label_batch]).sum(1))
                            loss = 1 * zinb_loss + 1 * neighbor_loss + 1 * neighbor_loss2 + 1 * self_loss + 1 * center_loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            print('Funetrain epoch [{}/{}], ZINB loss:{:.4f}, '
                                ' neighbor loss:{:.4f}, expanded neighbor loss:{:.4f}, self loss:{:.4f},'
                                'center loss:{:.4f}'
                                    .format(batch_idx + 1, epoch + 1, zinb_loss.item()
                                            , neighbor_loss.item(), neighbor_loss2.item(), self_loss.item(),
                                            center_loss.item()))
                else:
                    neighbor_loss = -torch.mean(((softmax_out_un *
                                                  score_near).sum(-1) *
                                                 weight.cuda()).sum(1))
                    neighbor_loss2 = -torch.mean(((output_re *
                                                   score_near_kk).sum(-1) *
                                                  weight_kk.cuda()).sum(1))
                    self_loss = -torch.mean((label_pred_tensor *
                                             score_self).sum(-1))
                    if epoch < pretrain_epoch:
                        optimizer.zero_grad()
                        (zinb_loss).backward()
                        optimizer.step()
                        print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'
                              .format(batch_idx + 1, epoch + 1, zinb_loss.item()))
                    elif epoch < midtrain_epoch:
                        mid_loss = zinb_loss + 1 * neighbor_loss + 1 * neighbor_loss2 + 1 * self_loss
                        optimizer.zero_grad()
                        mid_loss.backward()
                        optimizer.step()
                        print('Midtrain epoch [{}/{}], ZINB loss:{:.4f}, '
                            ' neighbor loss 1:{:.4f}, expanded neighbor loss 1:{:.4f}, self loss:{:.4f}'
                                .format(batch_idx + 1, epoch + 1, zinb_loss.item()
                                        , neighbor_loss.item(), neighbor_loss2.item(), self_loss.item()))
                    else:

                        if pseudo_option == True:
                            pseudo_loss = -torch.mean(torch.sum(pseudo_label_onehot.cuda() *
                                                                torch.log(label_pred_tensor), dim=-1))
                            loss = zinb_loss + 1 * neighbor_loss + 1 * neighbor_loss2 + 1 * self_loss + 1 * pseudo_loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            print('Funetrain epoch [{}/{}], ZINB loss:{:.4f}, '
                                ' neighbor loss:{:.4f}, expanded neighbor loss:{:.4f}, self loss:{:.4f},'
                                ' pseudo loss:{:.4f}'
                                    .format(batch_idx + 1, epoch + 1, zinb_loss.item()
                                            , neighbor_loss.item(), neighbor_loss2.item(), self_loss.item(),
                                            pseudo_loss.item()))
                        else:
                            center_loss = -torch.mean((output_f_ * class_center[pseudo_label_batch]).sum(1))
                            loss = zinb_loss + 1 * neighbor_loss + 1 * neighbor_loss2 + 1 * self_loss + 1 * center_loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            print('Funetrain epoch [{}/{}], ZINB loss:{:.4f}, '
                                ' neighbor loss:{:.4f}, expanded neighbor loss:{:.4f}, self loss:{:.4f},'
                                'center loss:{:.4f}'
                                    .format(batch_idx + 1, epoch + 1, zinb_loss.item()
                                            , neighbor_loss.item(), neighbor_loss2.item(), self_loss.item(),
                                            center_loss.item()))
                batch_idx += 1

            with torch.no_grad():
                _, y_pred = torch.max(score_bank, 1)
                if bimodality:
                    X_total_tensor = torch.Tensor(x)
                    X_total_tensor = Variable(X_total_tensor).cuda()
                    latent_total_tensor, logit_total_tensor, logit_total_tensor_1, logit_total_tensor_2, _, _, _ = self.forward(
                        X_total_tensor)
                    if len(x) > 500:
                        idx_rand = torch.randperm(len(x))[:500]
                        latent_total_tensor = latent_total_tensor[idx_rand]
                    latent_total_tensor_mix = mixup(latent_total_tensor.cpu())
                    latent_total_tensor_mix = latent_total_tensor_mix.cuda()
                    logit_total_tensor_mix = self.classifier(latent_total_tensor_mix)
                    logit_total_tensor_1_mix = self.add_classifier_1(latent_total_tensor_mix)
                    logit_total_tensor_2_mix = self.add_classifier_2(latent_total_tensor_mix)
                    ensemble_threshold = ensemble_score(logit_total_tensor_mix.cpu(),
                                                        logit_total_tensor_1_mix.cpu(),
                                                        logit_total_tensor_2_mix.cpu())
                    ensemble_threshold = ensemble_threshold.mean()
                    y_pred_known = y_pred[ensemble_total > ensemble_threshold]
                    current_error = (((y_pred != last_pred)[ensemble_total > ensemble_threshold]).sum()) / len(
                        y_pred_known)
                else:
                    current_error = (y_pred != last_pred).sum() / len(last_pred)
                if epoch >= pretrain_epoch:
                    print("current error:", current_error)
                    if current_error < error:
                        break
                    else:
                        last_pred = y_pred
        if bimodality:
           pred_celltype=[]
           for idx, i in enumerate(y_pred):
               anno = annotation[i]
               if ensemble_total[idx] <= ensemble_threshold:
                   anno='Unknown'
               pred_celltype.append(anno)
        else:
            pred_celltype = [annotation[i] for i in y_pred]
        return bimodality,pred_celltype