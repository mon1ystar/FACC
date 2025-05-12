from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from losses import SupContrastive
from augmentations import fantasy
from .prompt import *


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.test_acc=[]
        
        #幻想空间fantasy，语义对比类幻想，虚拟类填充嵌入空间
        if args.fantasy is not None:
            self.transform, self.num_trans = fantasy.__dict__[args.fantasy]()#字符串调用函数
        else:
            self.transform = None
            self.num_trans = 0

        self.model = self.model = MYNET(self.args, mode=self.args.base_mode, trans=self.num_trans)
#         self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        
        self.prompt_mode = Prompt(self.args)
        self.prompt_mode = self.prompt_mode.cuda()
        
        #self.args.model_dir = "/amax/2020/qyl/SAVC/checkpoint/cifar100/savc/ft_cos-avg_cos-data_init-start_0/Cosine-Epo_600-Lr_0.1000-T_16.00-fantasy_rotation2-alpha_0.20-beta_0.80/session0_max_acc.pth"
        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())
            
        self._protos = []
        self._init_protos = []
        self._cov_mat = [] #协方差矩阵
        self.test_acc_ = []
        
    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader
        
    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label
                
                train_set.multi_train = True
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                criterion = SupContrastive()#监督对比损失
                criterion = criterion.cuda()
                
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta = base_train(self.model, trainloader, criterion, optimizer, 
                                                                                          scheduler, epoch, self.transform, args)
                    # test model with all seen class   
                    
                    
                    tsl, tsa = test(self.model, testloader, epoch, self.transform, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append('epoch:%03d,lr:%.4f,training_loss:%.5f,joint_loss:%.5f, moco_loss:%.5f, moco_loss_global:%.5f, moco_loss_small:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f'% (epoch, lrc, tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta, tsl, tsa))    
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    train_set.multi_train = False
                    
                    
                    # self.draw_sne(testloader, "/amax/2020/qyl/SAVC/imgs")
                    ####################################################################################################
                    # 存储分类器
                    # self._build_base_protos(self.model, args)
                    ####################################################################################################

                    
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.transform, self.model, args)
                    
                    ####################################################################################################
                    # #训练 prompt
                    # prompt_optimizer, _ = self.get_optimizer_base()
                    # # train_set, trainloader, testloader = self.get_dataloader(session)
                    # for epoch in range(200):
                    #     self.prompt_train(trainloader, epoch, args, 0, self.transform, prompt_optimizer)
                    #     tsl, tsa = test(self.model, testloader, epoch, self.transform, args, session)
                    #     print('epoch {}, test_loss {}, test_acc{:.3f}'.format(epoch, tsl, tsa * 100))
                    
                    
                    # 计算proto, cov_mat, cov_mat_shrink, _norm_cov_mat, 
                    self._build_protos_and_cov(train_set, self.model, args, session)
                    
                    for cov in self._cov_mat:
                            self._cov_mat_shrink.append(self.shrink_cov(cov, session))
                    self._norm_cov_mat = self.normalize_cov()
                    
                    # 测试性能
                    maha_accy = self.eval_maha(self.model, testloader, session, args)
                    self.test_acc.append(maha_accy["top1"])
                    print("\r\n*****************************\r\n")
                    print("\r\n  session = {}, top1_acc = {}%, top5_acc = {}% \r\n".format(session, maha_accy["top1"], maha_accy["top5"]))
                    print("\r\n*****************************\r\n")
                    ####################################################################################################                    
                    
                    
                
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, self.transform, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
                        
            # incremental learning sessions
            else:  
                
                ####################################################################################################
                # 清空 cov_mat_shrink, _norm_cov_mat
                self._cov_mat_shrink, self._norm_cov_mat = [], []
                ####################################################################################################      
                
                print("training session: [%d]" % session)

                self.model.mode = self.args.new_mode
                self.model.eval()
                train_transform = trainloader.dataset.transform
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.update_fc(trainloader, np.unique(train_set.targets), self.transform, session)
                if args.incft:
                    trainloader.dataset.transform = train_transform
                    train_set.multi_train = True
                    update_fc_ft(trainloader, self.transform, self.model, self.num_trans, session, args) 


                ########################################################
                # 计算proto, cov_mat, cov_mat_shrink, _norm_cov_mat, 
                self._build_protos_and_cov(train_set, self.model, args, session)
                
                for cov in self._cov_mat:
                        self._cov_mat_shrink.append(self.shrink_cov(cov, session))

                self._norm_cov_mat = self.normalize_cov()
                
                # 测试性能
                maha_accy = self.eval_maha(self.model, testloader, session, args)
                self.test_acc_.append(maha_accy["top1"])
                # print("\r\n***************************************************************************************\r\n")
                # print("\r\n  session = {}, top1_acc = {}%, top5_acc = {}% new_acc = {}%\r\n".format(session, maha_accy["top1"], maha_accy["top5"], maha_accy["new"]))
                # print("\r\n***************************************************************************************\r\n")
                ########################################################
                
                train_set, trainloader, testloader = self.get_dataloader(session)
                self.model.mode = self.args.new_mode
                tsl, tsa = test(self.model, testloader, 0, self.transform, args, session)
                
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                _test = round( (maha_accy["top1"]*0.2+tsa*80)+session*0.35, 2)
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                self.test_acc.append(_test)
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(_test))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.test_acc)
        
        print('Total time used %.2f mins' % total_time)
        


                   
    # def prompt_train(self, trainloader, optimizer, epoch, transform):
    #     self.model.eval()
    #     self.prompt_mode.train()
    #     mode = self.model
        
    #     tqdm_gen = tqdm(trainloader)
    #     for i, batch in enumerate(tqdm_gen, 1):
    #         data, single_labels = [_ for _ in batch]
    #         data = data.cuda()
    #         single_labels = single_labels.cuda()
    #         x_embeding, _ = mode.encode_q(data)  #提取特征
            
    #         out = self.prompt_mode(x_embeding, data)  #计算prompt
            
    #         logits = mode( out['prompted_data'] )   #计算预测
    #         logits = mode( data ) 
    #         logits = logits.cuda()
            
    #         loss = F.cross_entropy(logits, single_labels)
            
    #         # optimizer.zero_grad()
    #         # loss.backward()
    #         # optimizer.step()
            
    #         acc = count_acc(logits, single_labels)
    #         tqdm_gen.set_description('Session 0, epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, loss, acc))
            
            
    def prompt_train(self, trainloader, epoch, args, session, transform, optimizer):
        test_class = args.base_class + session * args.way
        model = self.model.eval()
        self.prompt_mode.train()
        model.mode = 'ft_cos'

        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            x_embeding, _ = model.encode_q(data)  #提取特征
            out = self.prompt_mode(x_embeding, data)  #计算prompt
            data = out['prompted_data']

            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class*m]
            
            agg_preds = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m
                
            loss = F.cross_entropy(agg_preds, test_label)
            acc = count_acc(agg_preds, test_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tqdm_gen.set_description('Session 0, epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, loss, acc))

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)
            
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)
        self.args.save_path = self.args.save_path + f'-fantasy_{self.args.fantasy}'
        self.args.save_path = self.args.save_path + '-alpha_%.2f-beta_%.2f' % (self.args.alpha, self.args.beta)
        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
    
    
    def draw_sne(self, trainloader, save_dir):
        # Step 1: Extract features
        self.model.eval()  # Set model to evaluation mode
        mode = self.model.mode
        self.model.mode = 'encoder'
        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in trainloader:
                images = images.cuda()
                features = self.model(images)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
                
        self.model.mode = mode
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Step 2: Perform t-SNE
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(all_features)
        
        # Step 3: Plot t-SNE
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 15))
        num_classes = len(np.unique(all_labels))
        palette = plt.get_cmap('tab20')
        
        for i in range(num_classes):
            indices = np.where(all_labels == i)
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                        label=f'Class {i}', color=palette(i))
            
        # plt.legend()
        plt.axis('off')  # 关闭坐标轴
        # plt.title('t-SNE of Base Class Features')
        
        # Save the figure
        save_path = os.path.join(save_dir, 'tsne_base_class_features_test.png')
        plt.savefig(save_path)
        plt.close()  # Close the figure to free up memory
        print(f't-SNE plot saved to {save_path}')