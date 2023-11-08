from __future__ import print_function
import argparse
import os
import random

import torch
import torch.optim as optim
import utils
import basenet
import torch.nn.functional as F
import random
import numpy as np
import warnings
from datapre import  all_data, train_test_preclass,load_data03
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics

from collections import Counter

from clean import train_test_preclass_for_pseudo_label
from LPA import label_propagation_torch
import csv
import losses





warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='CLDA HSI Classification')
parser.add_argument('--batch-size', type=int, default=36, metavar='N',
                    help='input batch size for training (default: 36)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=16, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--num_samples', type=int, default=180)
parser.add_argument('--flag', type=bool, default=False)
parser.add_argument('--lr_flag', type=bool, default=True)

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',  
                    help='learning rate (default: 0.01)')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--HalfWidth', type=int, default=4)
parser.add_argument('--warmup',type=int, default=20)
parser.add_argument('--nDataSet', type=int, default=1)
parser.add_argument('--tao', type=float, default=0.5)



parser.add_argument('--kernel_mul', type=float, default=2.0)
parser.add_argument('--kernel_num', type=int, default=5)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.gpu)
use_gpu = torch.cuda.is_available()
print(args)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

checkpiont_path = 'results/houston'
if not os.path.exists(checkpiont_path):
    os.makedirs(checkpiont_path)


tao = args.tao
alpha = args.alpha
beta = args.beta
warmup = args.warmup
HalfWidth = args.HalfWidth
lr = args.lr
patch_size = 2 * HalfWidth + 1

num_epoch = args.epochs
num_k = args.num_k
BATCH_SIZE = args.batch_size

n_outputs = 128
nBand = 48
CLASS_NUM = 7
num_samples = args.num_samples

#load data
data_path_s = './datasets/Houston/Houston13.mat'
label_path_s = './datasets/Houston/Houston13_7gt.mat'
data_path_t = './datasets/Houston/Houston18.mat'
label_path_t = './datasets/Houston/Houston18_7gt.mat'

source_data,source_label = load_data03(data_path_s,label_path_s)
target_data,target_label = load_data03(data_path_t,label_path_t)
print(source_data.shape,source_label.shape)
print(target_data.shape,target_label.shape)

nDataSet = args.nDataSet    #sample times

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])

seeds = [1511, 1514, 1447, 1417, 1386, 1327, 1574, 1539, 1274, 1533]

best_predict_all = 0
best_test_acc = 0
best_G,best_RandPerm,best_Row,best_Column = None,None,None,None
# alpha = args.alpha

def cosine_similarity_matrix(a, b):
    """
    计算矩阵a和b中每行之间的余弦相似度.
    """
    # 求出a和b的范数
    a_norm = a.norm(dim=1, keepdim=True)
    my_list = list(b.values())
    b = torch.stack(my_list).to(a.device)
    b_norm = b.norm(dim=1, keepdim=True)

    # 计算点积数量
    dot_product = torch.matmul(a, b.transpose(0, 1))

    # 计算余弦相似度矩阵
    cosine_similarities = dot_product / torch.matmul(a_norm, b_norm.transpose(0, 1))
    return cosine_similarities

def train(ep, data_loader, data_loader_t,train_epoch, flag=False,U=None):

    criterion_s = nn.CrossEntropyLoss().cuda()
    criterion_t = nn.CrossEntropyLoss().cuda()
    mmd_loss = losses.MMD(kernel_mul=args.kernel_mul, kernel_num=args.kernel_num)
    # mmd_loss = losses.MMDLoss(kernel_mul=args.kernel_mul, kernel_num=args.kernel_num)
    mmd_weight = 1

    gamma = 0.01

    for batch_idx, data in enumerate(zip(data_loader, data_loader_t)):
        G.train()
        F1.train()
        F2.train()
        start_test = True

        if ep >= train_epoch and flag:
            (data_s, label_s), (data_t, fake_label_t) = data
            fake_label_t = Variable(fake_label_t).cuda()
        else:
            (data_s, label_s), (data_t, _) = data
        if args.cuda:
            data_s, label_s = data_s.cuda(), label_s.cuda()
            data_t = data_t.cuda()

        data_all = Variable(torch.cat((data_s, data_t), 0))
        label_s = Variable(label_s)
        bs = len(label_s)

        """source domain discriminative"""
        # Step A train all networks to minimize loss on source
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        output = G(data_all)

        if start_test:
            train_data = output[:bs, :].detach().cpu().float()
            train_labels = label_s.cpu()
            start_test = False
        else:
            train_data = torch.cat((train_data, output[:bs, :].detach().cpu().float()), 0)  # (53200,128)
            train_labels = torch.cat((train_labels, label_s.cpu()), 0)  # 53200


        # train classifiers
        output1 = F1(output)
        output2 = F2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]
        output_t2 = output2[bs:, :]
        output_t1_prob = F.softmax(output_t1)
        output_t2_prob = F.softmax(output_t2)

        entropy_loss = 0


        loss1 = criterion_s(output_s1, label_s)
        loss2 = criterion_s(output_s2, label_s)
        if ep >= train_epoch and flag:
            target_loss = criterion_t(output_t1, fake_label_t) + criterion_t(output_t2, fake_label_t)
            entropy_loss = -torch.mean(torch.sum(output_t1_prob * torch.log(output_t1_prob / (output_t2_prob + 1e-6)), dim=1))
            entropy_loss -= torch.mean(torch.sum(output_t2_prob * torch.log(output_t2_prob / (output_t1_prob + 1e-6)), dim=1))
        else:
            target_loss = 0
        entroy_target_loss = 0
        all_loss = loss1 + loss2 + 0.01 * entropy_loss +  alpha *  target_loss

        all_loss.backward()
        optimizer_g.step()
        optimizer_f.step()


        """target domain discriminative"""
        # Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        output = G(data_all)

        output1 = F1(output)
        output2 = F2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]
        output_t2 = output2[bs:, :]
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)

        loss1 = criterion_s(output_s1, label_s)
        loss2 = criterion_s(output_s2, label_s)

        loss_dis = utils.cdd(output_t1, output_t2)
        F_loss = loss1 + loss2 - gamma * loss_dis
        F_loss.backward()
        optimizer_f.step()

        # Step C train genrator to minimize discrepancy
        for i in range(num_k):
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            output = G(data_all)

            output1 = F1(output)
            output2 = F2(output)

            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]

            output_t1_prob = F.softmax(output_t1)
            output_t2_prob = F.softmax(output_t2)

            loss_dis = utils.cdd(output_t1_prob, output_t2_prob)

            D_loss = gamma * loss_dis

            D_loss.backward()
            optimizer_g.step()



         # Step E train genrator to minimize discrepancy MMD
        for i in range(1):
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data_all)
            output1 = F1(output)
            output2 = F2(output)

            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            if ep>=train_epoch:
                if U:
                    cosine_similarities = cosine_similarity_matrix(output[bs:], U)
                    temp_label_xu = torch.argmax(cosine_similarities, dim=1)
                    unique_labels_u = torch.unique(temp_label_xu)   # 获得唯一标签
                    class_xu_1 = {} # 划分集合 目标域的特征
                    class_xu_2 = {} # 划分集合 目标域的特征
                    # Loop through unique labels
                    for label in unique_labels_u:
                        # Get indices of samples with current label
                        indices = torch.where(temp_label_xu == label)[0]
                        # Get samples with current label
                        samples_1 = output1[bs:,:][indices] # 可以再考虑需不需要使用概率输出
                        samples_2 = output2[bs:,:][indices]
                        class_xu_1[label.item()] = samples_1
                        class_xu_2[label.item()] = samples_2

                    unique_labels = torch.unique(label_s)   # 获得唯一标签
                    
                    class_x_1 = {} # 划分集合 源域的特征
                    class_x_2 = {} # 划分集合 源域的特征
                    for label in unique_labels:
                        # Get indices of samples with current label
                        indices = torch.where(label_s == label)[0]
                        # Get samples with current label
                        samples_1 = output1[:bs,:][indices]
                        samples_2 = output2[:bs,:][indices]
                        class_x_1[label.item()] = samples_1
                        class_x_2[label.item()] = samples_2
        
                    # 逐类 计算 mmd 损失
                    m_loss = 0
                    c = 0
                    for i in unique_labels_u:
                        if i in unique_labels:
                            c += 1
                            m_loss = mmd_loss(class_x_1[i.item()] + class_x_2[i.item()], class_xu_1[i.item()] + class_xu_2[i.item()])
                    M_loss = mmd_weight * ((m_loss / c) + mmd_loss(output1[:bs, :]+output2[:bs, :], output1[bs:, :]+output2[bs:, :]))
                else:
                    M_loss = mmd_weight*mmd_loss(output1[:bs, :]+output2[:bs, :], output1[bs:, :]+output2[bs:, :])
                M_loss.backward()
                optimizer_g.step()
                optimizer_f.step()

    if ep>=train_epoch:
        print(
            'Train Ep: {} \ttrian_target_dataset:{}\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} M_loss: {:.6f} '.format(
                ep, len(data_loader_t.dataset),
                loss1.item(), loss2.item(), loss_dis.item(), M_loss.item()))
    else:
        print(
            'Train Ep: {} \ttrian_target_dataset:{}\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} M_loss: {:.6f} '.format(
                ep, len(data_loader_t.dataset),
                loss1.item(), loss2.item(), loss_dis.item(), 0.0))
    

    unique_labels = torch.unique(train_labels)
    class_means = {}
    # Loop through unique labels
    for label in unique_labels:
        # Get indices of samples with current label
        indices = torch.where(train_labels == label)[0]
        # Get samples with current label
        samples = train_data[indices]
        # Calculate mean of samples with current label
        mean = torch.mean(samples, dim=0)
        # Add mean to dictionary with current label as key
        class_means[label.item()] = mean
    return class_means
    
def get_clean_label(tarin_loader, test_loader, G, F1, F2, tao=0.85, num_classes=7, num=36):
    '''
    随机采样
    获得所有源域训练样本和目标域样本的编码特征
    G: 编码器
    F1: 分类头
    F2: 投影头
    '''
    start_test = True
    G.eval()
    F1.eval()
    F2.eval()
    
    # 先获得所有源域训练样本特征
    with torch.no_grad():
        iter_test = iter(tarin_loader)
        for i in range(len(tarin_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = G(inputs)
            projects = F2(feas)
            if start_test:
                all_projects_source = projects
                all_label_source = labels
                start_test = False
            else:
                all_projects_source = torch.cat((all_projects_source, projects), 0)  # (53200,7)
                all_label_source = torch.cat((all_label_source, labels), 0)  # 53200

        start_test = True
        iter_test = iter(test_loader)

        true_inputs = []
        clean_ps_labels = []
        true_labels = []
        for i in range(len(test_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            feas = G(inputs)
            probilities = F1(feas)
            probilities = nn.Softmax(dim=1)(probilities)
            projects = F2(feas)
            indices = torch.randperm(all_projects_source.size()[0], device='cuda')[:len(labels)]
            projects_source = all_projects_source[indices]
            label_source = all_label_source[indices]
            lpa_prob = label_propagation_torch(projects_source, label_source, projects, alpha=0.99, num_class=7)
            target_lpa_prob = lpa_prob[len(indices):,]

            confident_prob = target_lpa_prob * probilities
            label_mask = confident_prob.max(1).values > tao
            _, ps_labels = torch.max(confident_prob, 1)
            # 现在这些都是cuda
            filtered_inputs = inputs[label_mask]
            filtered_ps_labels = ps_labels[label_mask]
            filtered_labels = labels[label_mask]

            if start_test:
                true_inputs = filtered_inputs.detach().cpu().numpy()
                clean_ps_labels = filtered_ps_labels.detach().cpu().numpy()
                true_labels = filtered_labels.detach().cpu().numpy()
                start_test = False
            else:
                true_inputs = np.concatenate((true_inputs, filtered_inputs.detach().cpu().numpy()), axis=0)  
                clean_ps_labels = np.concatenate((clean_ps_labels, filtered_ps_labels.detach().cpu().numpy()), axis=0) 
                true_labels = np.concatenate((true_labels, filtered_labels.detach().cpu().numpy()), axis=0)  

        # 统计准确率
        C = metrics.confusion_matrix(clean_ps_labels, true_labels)
        A = np.diag(C) / np.sum(C, 1, dtype=np.float32)
        
        class_sample_count = np.zeros(num_classes).astype(np.int64)
        label_counts = Counter(clean_ps_labels)
        for label, count in label_counts.items():
            class_sample_count[label] = count
        print(f'伪标记个数：{class_sample_count}')
        if 0 in class_sample_count:
            flag_print = False
            print("There are elements equal to 0 in predicted_labels")
        else:
            flag_print = True
            print("There are no elements equal to 0 in predicted_labels")
        
        accuracy = metrics.accuracy_score(clean_ps_labels, true_labels)
        print(f'Pseudo Label Accuracy_score: {accuracy}')
        if flag_print:
            for i in range(num_classes):
                print("Class " + str(i) + ": " + "{:.2f}".format(100 * A[i]))

        
    return true_inputs, clean_ps_labels, accuracy, flag_print

def test(data_loader):

    test_pred_all = []
    test_all = []
    predict = np.array([], dtype=np.int64)

    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct_add = 0
    size = 0

    for batch_idx, data in enumerate(data_loader):
        img, label = data
        img, label = img.cuda(), label.cuda()
        img, label = Variable(img, volatile=True), Variable(label)
        output = G(img)
        output1 = F1(output)
        output2 = F2(output)

        output_add = output1 + output2
        pred = output_add.data.max(1)[1]
        test_loss += F.nll_loss(F.log_softmax(output1, dim=1), label, size_average=False).item()
        correct_add += pred.eq(label.data).cpu().sum()
        size += label.data.size()[0]
        test_all = np.concatenate([test_all, label.data.cpu().numpy()])
        test_pred_all = np.concatenate([test_pred_all, pred.cpu().numpy()])
        predict = np.append(predict, pred.cpu().numpy())
    test_accuracy = 100. * float(correct_add) / size
    test_loss /= len(data_loader.dataset)  # loss function already averages over batch size
    print('Epoch: {:d} Test set:test loss:{:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        ep, test_loss, correct_add, size, 100. * float(correct_add) / size))
    
    
    #这里应该用最优的
    if test_accuracy>acc[iDataSet]:
        acc[iDataSet] = 100. * float(correct_add) / size
        OA = acc
        C = metrics.confusion_matrix(test_all, test_pred_all)
        A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

        k[iDataSet] = metrics.cohen_kappa_score(test_all, test_pred_all)
        torch.save({'netG':G.state_dict(),'F1':F1.state_dict(),'F2':F2.state_dict()}, os.path.join(checkpiont_path, str(iDataSet)+'.pt'))



    return test_accuracy, predict

for iDataSet in range(nDataSet):
    
    print('#######################idataset######################## ', iDataSet)

    set_seed(seeds[iDataSet])
    train_xs, train_ys, test_xs, test_ys = train_test_preclass(source_data, source_label, HalfWidth, num_samples)
    testX, testY, G_test, RandPerm, Row, Column = all_data(target_data, target_label, HalfWidth)  # (7826,5,5,72)

    train_dataset = TensorDataset(torch.tensor(train_xs), torch.tensor(train_ys))
    train_t_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_t = DataLoader(train_t_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(train_t_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model
    G = basenet.EmbeddingNetHyperX(nBand, n_outputs=n_outputs, patch_size=patch_size, n_classes=CLASS_NUM).cuda()
    F1 = basenet.ResClassifier(num_classes=CLASS_NUM, num_unit=G.output_num(), middle=64)
    F2 = basenet.ResClassifier(num_classes=CLASS_NUM, num_unit=G.output_num(), middle=64)


    if args.cuda:
        G.cuda()
        F1.cuda()
        F2.cuda()

    optimizer_g = optim.SGD(list(G.parameters()), lr=lr / 10 , weight_decay=0.0005)

    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=lr / 10,
                            weight_decay=0.0005)

    train_num = warmup
    start_flag = False
    best_ps_label_acc = 0.0
    U = None

    for ep in range(1,num_epoch + 1):

        if (ep >= train_num and ep < num_epoch) and ep % train_num == 0:
            true_inputs, clean_ps_labels, ps_label_acc, flag_print = get_clean_label(train_loader_s, test_loader, G, F1, F2, tao, CLASS_NUM, num_samples)
            if flag_print:
                start_flag = True
            train_x_fake, train_y_fake, test_x_fake, test_y_fake = train_test_preclass_for_pseudo_label(true_inputs, clean_ps_labels, num_samples)
            target_datasets_fake = TensorDataset(torch.tensor(train_x_fake), torch.tensor(train_y_fake))
            train_loader_fake = DataLoader(target_datasets_fake, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,drop_last=True)
            print(len(target_datasets_fake))
        if start_flag: 
            U = train(ep, train_loader_s, train_loader_fake, train_num, start_flag, U)
        else:
            U = train(ep, train_loader_s, train_loader_t, train_num, start_flag, U)
        # if(ep+1)%10==0:
        test_accuracy, predict = test(test_loader)

    print('-' * 100, '\nTesting')


print(acc)
AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)

print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))
