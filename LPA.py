import numpy as np
import torch
import torch.nn.functional as nnF
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances

def similarity_matrix(X, sigma):
    D = euclidean_distances(X)
    S = np.exp(-D**2 / (2*sigma**2))
    return S

def smooth_one_hot(true_labels, classes, smoothing=0.0):
    # 将true_labels转化为one-hot编码
    one_hot = nnF.one_hot(true_labels, classes).float()
    # 计算平滑的标签向量
    smooth_labels = one_hot * (1 - smoothing) + smoothing / classes
    return smooth_labels


def label_propagation_torch(X, Y, X_u, alpha=0.99, num_class=7, max_iter=100, tol=1e-3):
    """
    Standard Label Propagation algorithm with symmetric normalization.
    参数：
    X: torch.Tensor
    形状为(n_labeled_samples, n_features)的张量，包含有标记样本的特征。
    Y: torch.Tensor
    形状为(n_labeled_samples, n_classes)的张量，包含有标记样本的独热编码标签。
    X_u: torch.Tensor
    形状为(n_unlabeled_samples, n_features)的张量，包含无标记样本的特征。
    alpha: float
    阻尼系数，必须在0和1之间。
    max_iter: int，可选（默认为100）
    迭代的最大次数。
    tol: float，可选（默认为1e-3）
    收敛的容差。

    返回值：
    F: torch.Tensor
    形状为(n_labeled_samples + n_unlabeled_samples, n_classes)的张量，包含所有样本的标签概率。
    """
    device = X.device
    n_labeled_samples, n_features = X.shape
    # 获取无标签数据的样本数
    n_unlabeled_samples = X_u.shape[0]

    # 获取总样本数
    n_samples = n_labeled_samples + n_unlabeled_samples

    # 构建相似度矩阵W
    # 这部分应该构造成对角线为0的对称邻接矩阵
    X_all = torch.cat([X, X_u], dim=0)  # 将有标签数据和无标签数据合并成一个大的张量
    S = torch.from_numpy(cosine_similarity(X_all.cpu().detach().numpy())).float().to(X_all.device)
    
    # 对于每个节点，最多保留K个相邻节点
    K = int(num_class // 2)
    if K < 2:
        K = 2

    # 选择阈值
    thresholds = torch.topk(S, k=K, dim=-1)[0][:, -1]

    # 构造对称邻接矩阵
    A = (S >= thresholds.view(-1, 1)).to(Y.device) # broadcast thresholds along rows

    # 构造邻接矩阵
    S.fill_diagonal_(0)  # 将对角线上的元素设为0

    A = A.float() * (1 - torch.eye(S.size(0)).to(Y.device)) # set diagonal to 0
    A = (A + A.t()) / 2.0 # make the matrix symmetric
    A.fill_diagonal_(0)  # 将对角线上的元素设为0
    S = A

    # 构建度矩阵D
    degrees = torch.sum(S, dim=1)  # 对相似度矩阵S按行求和得到每个节点的度
    D = torch.diag(degrees)  # 构建对角线上元素为节点度数的矩阵
    # 构建对称归一化矩阵
    D_sqrt = torch.sqrt(D)  # 对度矩阵D的每个元素求平方根
    D_sqrt_inv = torch.inverse(D_sqrt)  # 求平方根的逆矩阵


    W = torch.matmul(torch.matmul(D_sqrt_inv, S), D_sqrt_inv)

    # 构造标签list
    label_list = np.array([i for i in range(num_class)])
    unlabel_label = np.random.choice(label_list, size=n_unlabeled_samples)
    unlabel_label = torch.from_numpy(unlabel_label).long().to(Y.device)
    one_hot_labels =  smooth_one_hot(Y, num_class, smoothing=0.1)
    one_hot_unlabels = smooth_one_hot(unlabel_label, num_class, smoothing=0.01)

    one_hot_unlabels = torch.zeros((n_unlabeled_samples, one_hot_unlabels.shape[1]))
    one_hot_unlabels -= 1


    # one_hot_labels = nnF.one_hot(Y, num_classes=num_class)
    # one_hot_unlabels = nnF.one_hot(unlabel_label, num_classes=num_class)

    Y_all = torch.zeros((n_samples, one_hot_labels.shape[1]))
    # Y_all -= 1
    Y_all[:n_labeled_samples, :] = torch.as_tensor(one_hot_labels, device=Y.device)
    Y_all[one_hot_labels.shape[0]:,:] = torch.as_tensor(one_hot_unlabels, device=Y.device)

    F = torch.as_tensor(Y_all, device=Y_all.device)  # 将有标签数据的标签赋值给标签概率矩阵的前n_labeled_samples行
    F = F.type(torch.FloatTensor)
    Y_all = Y_all.type(torch.FloatTensor)

    F, W, Y_all = F.to(device), W.to(device), Y_all.to(device)
    
    # 直接计算闭集解
    # I = torch.eye(n_samples).to(device)
    # F = torch.mm(torch.inverse(I - alpha * W), Y_all)

    
    # I = torch.eye(n_unlabeled_samples).to(device)
    # F = torch.mm(torch.inverse(I - alpha * W), Y_all[:n_labeled_samples,:])



    # # 迭代直到收敛
    for i in range(max_iter):
        F_old = F.clone()  # 备份上一轮的标签概率矩阵
        F = alpha * torch.mm(W, F) + (1 - alpha) * Y_all  # 根据公式更新标签概率矩阵
        if torch.norm(F - F_old) < tol:  # 如果标签概率矩阵变化的范数小于收敛阈值，则停止迭代
            print(f'now is dowm')
            break
        F[:n_labeled_samples, :] = torch.as_tensor(one_hot_labels, device=Y.device).type(torch.FloatTensor)


    return F  # 返回标签概率矩阵

def label_propagation_close_torch(X, Y, X_u, alpha, max_iter=100, tol=1e-3):
    n_labeled_samples, n_features = X.shape
    n_unlabeled_samples = X_u.shape[0]
    n_samples = n_labeled_samples + n_unlabeled_samples
    
    # 构造相似性矩阵 S
    X_all = torch.cat((X, X_u), dim=0)
    S = torch.matmul(X_all, X_all.T)  # 计算矩阵X_all和它的转置的乘积得到相似度矩阵
    S.diagonal().fill_(0)  # 将对角线上的元素设为0
    S = torch.triu(S) + torch.triu(S, diagonal=1).T  # 将相似度矩阵的对称元素赋值给下三角矩阵

    # 构造度矩阵 D
    degrees = torch.sum(S, dim=1)
    D = torch.diag(degrees)

    # 构造对称归一化矩阵
    D_sqrt = torch.sqrt(D)
    D_sqrt_inv = torch.inverse(D_sqrt)
    W = torch.matmul(D_sqrt_inv, torch.matmul(S, D_sqrt_inv))

    # 计算矩阵 Omega
    Omega = torch.eye(n_samples) - alpha * W

    # 计算矩阵 Phi
    Phi = torch.inverse(Omega)
    q_s = torch.zeros((n_samples, Y.shape[1]))
    q_s[:n_labeled_samples, :] = torch.tensor(Y)
    F = torch.matmul(Phi, q_s)

    return F


# if __name__=='__main__':
#     from sklearn.semi_supervised import LabelPropagation
#     torch.manual_seed(0)
#     X = torch.rand(10, 4)
#     X_u = torch.rand(5, 4)
#     y = torch.tensor([0, 1, 0, 2, 2, 1, 0, 0, 2, 1])
#     y_u = torch.zeros(1, X_u.shape[0])
#     # F = label_propagation_torch(X, y, X_u, alpha=0.99, num_class=3, max_iter=100, tol=1e-3)
#     lp_model = LabelPropagation(kernel='knn', n_neighbors=3)
#     lp_model.fit(X, y)
#     F = lp_model.predict_proba(X_u)
#     print(F)