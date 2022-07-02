import math
import torch
import numpy as np
import torch.nn.functional as F


class sim(object):
    def __init__(self):
        pass 
    
    def cent(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kern(self, X, Y, sigma):
        return np.sum(self.cent(self.rbf(X, sigma)) * self.cent(self.rbf(Y, sigma)))

    def linear(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T        
        return np.sum(self.cent(L_X) * self.cent(L_Y))

    def linear_cka(self, X, Y):
        hsic = self.linear(X, Y)
        var1 = np.sqrt(self.linear(X, X))
        var2 = np.sqrt(self.linear(Y, Y))

        return hsic / (var1 * var2)

    def kernel_cka(self, X, Y, sigma=None):
        hsic = self.kern(X, Y, sigma)
        var1 = np.sqrt(self.kern(X, X, sigma))
        var2 = np.sqrt(self.kern(Y, Y, sigma))

        return hsic / (var1 * var2)

    
class cka_sim(object):
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

    def linear_cka_sim(self, X, Y):
        X = torch.transpose(X, 0, 1)
        Y = torch.transpose(Y, 0, 1)
        hsic = self.linear(X, Y)
        var1 = torch.sqrt(self.linear(X, X))
        var2 = torch.sqrt(self.linear(Y, Y))

        return hsic / (var1 * var2)

    def kernel_cka_sim(self, X, Y, sigma=None):
        X = torch.transpose(X, 0, 1)
        Y = torch.transpose(Y, 0, 1)
        hsic = self.kernel(X, Y, sigma)
        var1 = torch.sqrt(self.kernel(X, X, sigma))
        var2 = torch.sqrt(self.kernel(Y, Y, sigma))
        return hsic / (var1 * var2)


def cos_sim(x1, x2):
    return F.cosine_similarity(x1, x2).mean()


def cka_sim(x1, x2):
    device = 'cuda:{}'.format(x1.get_device())
    return cka_sim(device).linear_cka_sim(x1, x2)


def gloss(pred, gt, ignore_index=None):
    
    tokens = gt != ignore_index
    acc = (
        ((pred.argmax(-1) == gt) & tokens).float().sum() / tokens.sum()
    )
    return acc


accur = {
    'cos': cos_sim,
    'cka': cka_sim,
    'accgls': gloss
}
