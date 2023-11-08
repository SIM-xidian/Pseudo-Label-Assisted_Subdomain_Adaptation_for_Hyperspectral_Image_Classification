import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from LPA import label_propagation_torch
from sklearn import metrics
from collections import Counter
import math

from sklearn.cluster import KMeans


def get_cluster_label(data, k=7):
    # 定义聚类数量

    # 使用Scikit-learn进行K-means聚类
    kmeans = KMeans(n_clusters=k, algorithm='full')
    kmeans.fit(data.detach().cpu().numpy())

    # 获取聚类结果
    labels = kmeans.labels_

    return torch.from_numpy(labels)


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
            # true_label
            inputs = inputs.cuda()
            labels = labels.cuda()
            feas = G(inputs)
            projects = feas
            if start_test:
                all_projects_source = projects
                all_label_source = labels
                start_test = False
            else:
                all_projects_source = torch.cat((all_projects_source, projects), 0)  # (53200,7)
                all_label_source = torch.cat((all_label_source, labels), 0)  # 53200
        
        unique_labels = torch.unique(all_label_source)
        class_indices = [torch.where(all_label_source == label)[0] for label in unique_labels]

        start_test = True
        iter_test = iter(test_loader)

        true_inputs = []
        clean_ps_labels = []
        all_cluster_label = []
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
           
            # 获取聚类结果
            cluster_label = get_cluster_label(feas, k=num_classes)
            cluster_label = cluster_label.cuda()

            all_indices = []
            for i in unique_labels:
                indices = torch.randperm(class_indices[i].shape[0], device='cuda')[:9]
                all_indices.append(indices)
            all_indices = torch.cat(all_indices, dim=0)
            
            projects_source = all_projects_source[all_indices].to(labels.device)
                
            projects = feas

            label_source = all_label_source[all_indices]

            projects = nn.Softmax(dim=1)(projects)
            projects_source = nn.Softmax(dim=1)(projects_source)


            lpa_prob = label_propagation_torch(projects_source, label_source, projects, alpha=0.99, num_class=num_classes)
            target_lpa_prob = lpa_prob[len(all_indices):,]
            confident_prob = target_lpa_prob * probilities
            
            _, ps_labels = torch.max(confident_prob, 1)
            _, pred_probilities = torch.max(probilities, 1)

            label_mask = confident_prob.max(1).values > tao
            
            cluster_label_mask = ps_labels==cluster_label
            mask = label_mask

            filtered_inputs = inputs[mask]
            filtered_ps_labels = ps_labels[mask]
            filtered_labels = labels[mask]
            filtered_cluster_labels = cluster_label[mask]

            if start_test:
                true_inputs = filtered_inputs.detach().cpu().numpy()
                clean_ps_labels = filtered_ps_labels.detach().cpu().numpy()
                true_labels = filtered_labels.detach().cpu().numpy()
                all_cluster_label = filtered_cluster_labels.detach().cpu().numpy()
                start_test = False
            else:
                true_inputs = np.concatenate((true_inputs, filtered_inputs.detach().cpu().numpy()), axis=0)  
                clean_ps_labels = np.concatenate((clean_ps_labels, filtered_ps_labels.detach().cpu().numpy()), axis=0) 
                true_labels = np.concatenate((true_labels, filtered_labels.detach().cpu().numpy()), axis=0)  
                all_cluster_label = np.concatenate((all_cluster_label, filtered_cluster_labels.detach().cpu().numpy()), axis=0)  

        # 统计准确率
        C = metrics.confusion_matrix(clean_ps_labels, true_labels)
        A = np.diag(C) / np.sum(C, 1, dtype=np.float32)
        CC = metrics.confusion_matrix(all_cluster_label, true_labels)
        AA = np.diag(CC) / np.sum(CC, 1, dtype=np.float32)
        
        class_sample_count = np.zeros(num_classes).astype(np.int64)
        label_counts = Counter(clean_ps_labels)
        for label, count in label_counts.items():
            class_sample_count[label] = count
        print(f'伪标记个数_LPA：{class_sample_count}')

        if 0 in class_sample_count:
            flag_print = False
            print("There are elements equal to 0 in predicted_labels")
        else:
            flag_print = True
            print("There are no elements equal to 0 in predicted_labels")
        
        accuracy = metrics.accuracy_score(clean_ps_labels, true_labels)
        print(f'Pseudo Label Accuracy_score LPA: {accuracy}')
        if flag_print:
            for i in range(num_classes):
                print("Class " + str(i) + ": " + "{:.2f}".format(100 * A[i]))

    return true_inputs, clean_ps_labels, accuracy, flag_print

def train_test_preclass_for_pseudo_label(data, labels, train_num=180, test_num=None):
    # 获取每个类别的索引
    unique_labels = np.unique(labels)
    class_indices = [np.where(labels == label)[0] for label in unique_labels]

    # 随机打乱每个类别的索引
    for indices in class_indices:
        np.random.shuffle(indices)

    # 按照比例划分每个类别的索引为训练集和测试集
    train_indices = []
    test_indices = []
    for indices in class_indices:
        if len(indices)<train_num:
            l = math.ceil(train_num/len(indices))
            indices_temp = indices
            for item in range(l):
                indices = np.concatenate((indices,indices_temp), axis=0)
        train_idx = indices[:train_num]
        if test_num:
            test_idx = indices[train_num:train_num+test_num]
        else:
            test_idx = indices[train_num:]
        
        train_indices.append(train_idx)
        test_indices.append(test_idx)

    # 合并训练集和测试集的索引
    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)

    # 根据索引提取训练集和测试集数据和标签
    X_train = data[train_indices]
    y_train = labels[train_indices]
    X_test = data[test_indices]
    y_test = labels[test_indices]

    print('the number of the train samples:', len(train_indices))  # 520
    print('the number of the test samples:', len(test_indices))  # 520

    return X_train, y_train, X_test, y_test