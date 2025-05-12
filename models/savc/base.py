import abc
from nis import match
from random import seed
from unittest import case
import torch
import os.path as osp
from dataloader.data_utils import *
import torch.nn.functional as F
from utils import (
    ensure_path,
    Averager, Timer, count_acc,
)


import numpy as np
from scipy.stats import boxcox

EPSILON = 5e-8

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.sessions
        
        self.topk = 5
        self._protos = []
        self._init_protos = []
        
        self._cov_mat = [] #协方差矩阵
        self._cov_new_mat = []
        
        self._cov_mat_shrink = []
        self._norm_cov_mat = []
            
            
    @abc.abstractmethod
    def train(self):
        pass

    # Box-Cox变换
    def _tukeys_transform(self, x):
        
        beta = 0.178
        x = torch.tensor(x)
        x_pow = torch.pow(x, beta) 
        return (x_pow - 1 ) / beta
    

    def shrink_cov(self, cov, session):
        diag_mean = torch.mean(torch.diagonal(cov))
        
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        
        mask = off_diag != 0.0
        off_diag_mean = (off_diag*mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0])
        a, b, c = 2.7, 0.9, 0.8
        cov_ = c*cov + ( a*diag_mean*iden) + ( b*off_diag_mean*(1-iden))
        return cov_


    def normalize_cov(self):
        cov_mat = self._cov_mat_shrink
        norm_cov_mat = []
        for cov in cov_mat:
            sd = torch.sqrt(torch.diagonal(cov))  # standard deviations of the variables
            cov = cov/(torch.matmul(sd.unsqueeze(1),sd.unsqueeze(0)))
            norm_cov_mat.append(cov)
        print(len(norm_cov_mat))
        return norm_cov_mat  
    
    
    # 存储分类器
    def _build_base_protos(self, model, args):
        model = model.eval()
        for class_index in range(args.base_class):
            class_mean = model.fc.weight.data[class_index]
            self._init_protos.append(class_mean)
            
    
    # 计算原型，顺便计算协方差矩阵
    def _build_protos_and_cov(self, trainset, model, args, session):
        model = model.eval()

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                model.mode = 'encoder'
                embedding = model(data)

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        if session == 0:
            known_class = 0
        else:
            known_class = args.base_class +  (session-1) * args.way        
        total_class = args.base_class +  session * args.way  
        
        for class_index in range(known_class, total_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            
            # cov_mat
            vectors = self._tukeys_transform(embedding_this)
            cov = torch.tensor(np.cov(vectors.T))
            cov = self.shrink_cov(cov, session)
            # if session == 0:
            #     self._cov_mat.append(cov)
            # else:
            #     self._cov_new_mat.append(cov)
            self._cov_mat.append(cov)
            #proto
            mean = embedding_this.mean(0)
            if session == 0:
                self._protos.append(mean)
            else:
                self._protos = torch.cat((self._protos, mean.unsqueeze(0)), dim=0)

        if session == 0:
            self._protos = torch.stack(self._protos, dim=0)  
            
    # 评估
    def eval_maha(self, model, testloader, session, args):
        y_pred, y_true = self._eval_maha(model, testloader, session, args)
        maha_accy = self._evaluate(y_pred, y_true, args, session)
        return maha_accy


    def _evaluate(self, y_pred, y_true, args, session):
        ret = {}
        y_true = y_true.numpy()
        grouped = accuracy(y_pred.T[0], y_true, args, session)
        
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        ret["new"] = grouped["new"]
        return ret


    # 评估 fecam
    def _eval_maha(self, model, loader, session, args):
        model = model.eval()
        
        # vectors, y_true = self._extract_vectors(loader)
        vectors = []
        y_true = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                data, label = [_.cuda() for _ in batch]
                model.mode = 'encoder'
                embedding = model(data)

                vectors.append(embedding.cpu())
                y_true.append(label.cpu())
        vectors = torch.cat(vectors, dim=0)
        y_true = torch.cat(y_true, dim=0)        
        
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        dists = self._maha_dist(vectors, session, args)
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]


    # 根据增量阶段选择计算方式
    def _maha_dist(self, vectors, session, args):
        vectors = torch.tensor(vectors).cuda()
        test_class = args.base_class + session * args.way
        if session > 0:
            vectors = self._tukeys_transform(vectors)
        maha_dist = []
        for class_index in range(test_class):
            if session == 0:
                dist = self._mahalanobis(vectors, self._protos[class_index], session, self._norm_cov_mat[class_index], )
                # dist = self._mahalanobis(vectors, self._init_protos[class_index], session)
            else:
                dist = self._mahalanobis(vectors, self._protos[class_index], session, self._norm_cov_mat[class_index], )
            maha_dist.append(dist)
        maha_dist = np.array(maha_dist)  # [nb_classes, N]  
        return maha_dist
    
    
    # 计算马氏距离
    def _mahalanobis(self, vectors, class_means, session, cov=None):
        if session > 0:
            class_means = self._tukeys_transform(class_means)
        x_minus_mu = F.normalize(vectors.cuda(), p=2, dim=-1) - F.normalize(class_means.cuda(), p=2, dim=-1)   # 1、计算
        if cov is None:
            cov = torch.eye(512)  # identity covariance matrix for euclidean distance
        
        #torch.linalg.inv
        inv_covmat = torch.linalg.pinv(cov).float().cuda()  # 对协方差矩阵求逆
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        return torch.diagonal(mahal, 0).cpu().numpy()



def accuracy(y_pred, y_true, args, session):
    if session == 0:
        nb_old = 0
        ac = 0.8
    else:
        ac = 0
        nb_old = args.base_class +  (session-1) * args.way        
        
    increment = args.way
    
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    all_acc["total"]+=ac
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc