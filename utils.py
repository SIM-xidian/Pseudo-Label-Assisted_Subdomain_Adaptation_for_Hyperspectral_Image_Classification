import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def cdd(output_t1,output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss


def get_mean_feature(loader, netE, netC1, netC2):
    """
    计算所有训练样本的类别均值，
    """
    start_test = True
    netE.eval()
    netC1.eval()
    netC2.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netE(inputs)
            
            if start_test:
                train_data = feas.float()
                train_labels = labels
                start_test = False
            else:
                train_data = torch.cat((train_data, feas.float()), 0)  # (53200,128)
                train_labels = torch.cat((train_labels, labels), 0)  # 53200


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