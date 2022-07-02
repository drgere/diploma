import math
import torch
import numpy as np
import torch.nn as nn


class loss_cka(object):
    def __init__(self):
        pass 
    
    def center(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def r(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel(self, X, Y, sigma):
        return np.sum(self.center(self.r(X, sigma)) * self.center(self.r(Y, sigma)))

    def linear(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T        
        return np.sum(self.center(L_X) * self.center(L_Y))

    def linear_cka(self, X, Y):
        hsic = self.linear(X, Y)
        var1 = np.sqrt(self.linear(X, X))
        var2 = np.sqrt(self.linear(Y, Y))

        return hsic / (var1 * var2)

    def kernel_cka(self, X, Y, sigma=None):
        hsic = self.kernel(X, Y, sigma)
        var1 = np.sqrt(self.kernel(X, X, sigma))
        var2 = np.sqrt(self.kernel(Y, Y, sigma))

        return hsic / (var1 * var2)

    
class loss_cka_torch(object):
    def __init__(self, device):
        self.device = device
    
    def center(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def r(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel(self, X, Y, sigma):
        return torch.sum(self.center(self.r(X, sigma)) * self.center(self.r(Y, sigma)))

    def linear(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.center(L_X) * self.center(L_Y))

    def linear_cka(self, X, Y):
        X = torch.transpose(X, 0, 1)
        Y = torch.transpose(Y, 0, 1)
        hsic = self.linear(X, Y)
        var1 = torch.sqrt(self.linear(X, X))
        var2 = torch.sqrt(self.linear(Y, Y))

        return 1 - hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        X = torch.transpose(X, 0, 1)
        Y = torch.transpose(Y, 0, 1)
        hsic = self.kernel(X, Y, sigma)
        var1 = torch.sqrt(self.kernel(X, X, sigma))
        var2 = torch.sqrt(self.kernel(Y, Y, sigma))
        return 1 - hsic / (var1 * var2)

    
def mse_loss(pred, gt):
    loss=nn.MSELoss()
    return loss(pred, gt)


def cos_loss(pred, gt):
    device = device='cuda:{}'.format(pred.get_device())
    loss = nn.CosineEmbeddingLoss(
        margin=0.0, size_average=None, reduce=None, reduction='mean')
    return loss(pred, gt, torch.ones(gt.shape[0],device=device))


def cka_loss(pred, gt):
    device='cuda:{}'.format(pred.get_device())
    loss = loss_cka_torch(device).linear_cka
    return loss(pred, gt)


def loss(pred, gt, ignore_index=None):
    if ignore_index==None:
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss(pred, gt)


def loss_smooth(pred, gt, ignore_index=None, label_smoothing=0, **kwargs):
    if label_smoothing == 0:
        if ignore_index==None:
            loss = nn.CrossEntropyLoss()
        else:
            loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        if ignore_index==None:
            loss = nn.LabelSmoothingCrossEntropy(epsilon=label_smoothing)
        else:
            loss = nn.LabelSmoothingCrossEntropy(ignore_index=ignore_index, epsilon=label_smoothing)
    return loss(pred, gt)


def gloss(pred, gt, ignore_index=None):
    tokens = gt != ignore_index

    acc = (
        ((pred.argmax(-1) == gt) & tokens).float().sum() / tokens.sum()
    )
    return acc


LOSSES = {
    'mse': mse_loss,
    'cos': cos_loss,
    'cka': cka_loss,
    'xen': loss,
    'xens': loss_smooth,
}