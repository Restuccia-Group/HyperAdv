import torch
import torch.nn as nn
import numpy as np

class PGD():
    def __init__(self,loss_fn=nn.CrossEntropyLoss(),eps=0.05,n_iter=10,norm='linf',lr=0.01,clip_min=None,clip_max=None,advtrain=False) -> None:
        
        self.loss_fn = loss_fn
        self.eps = eps
        self.n_iter = n_iter
        self.norm = norm
        self.lr = lr
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.advtrain = advtrain

    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)

    def clip(self,xadv,x):
        if self.norm == 'linf':
            if self.clip_min is not None:
                lb = torch.clamp(x-self.eps,min=self.clip_min) # lower bound
            else:
                lb = x-self.eps
            xadv = torch.max(xadv,lb)
            if self.clip_max is not None:
                ub = torch.clamp(x+self.eps,max=self.clip_max) # upper bound
            else:
                ub = x+self.eps
            xadv = torch.min(xadv,ub)
        else: # projection 
            d = np.prod([*x.shape[1:]])
            delta = xadv - x
            batchsize = delta.size(0)
            deltanorm = torch.norm(delta.view(batchsize,-1),p=2,dim=1)
            scale = np.sqrt(d)*self.eps/deltanorm
            scale[deltanorm<=(np.sqrt(d*self.eps))] = 1
            delta = (delta.transpose(0,-1)*scale).transpose(0,-1).contiguous()
            xadv = x + delta
            if self.clip_min is not None and self.clip_max is not None:
                xadv = torch.clamp(xadv,self.clip_min,self.clip_max)

        return xadv.detach()
    
    def normalize(self,x):
        if self.norm == 'linf':
            x = x.sign()
        else:
            batch_size = x.size(0)
            norm = torch.norm(x.view(batch_size, -1), 2, 1)
            x = (x.transpose(0,-1)/norm).transpose(0,-1).contiguous()
        return x

    def adv_gen(self,forward_fn,x,y,hyper=False,ensemble=False):
        
        xadv = x.clone().detach() + 0.001 * torch.randn_like(x)
        xadv = self.clip(xadv,x).detach()
        forward_fn.eval()
        for i in range(self.n_iter):
            if hyper:
                output = forward_fn(xadv,False,ensemble)
            else:
                output = forward_fn(xadv)
                
            xadv.requires_grad = True
            if hyper:
                if self.advtrain:
                    output = forward_fn(xadv,True)
                    loss = 0
                    for j in range(len(output)):
                        loss += self.loss_fn(output[j],y)
                    loss /= len(output)
                else:
                    output = forward_fn(xadv,False,ensemble)
                    loss = self.loss_fn(output,y)
            else:
                output = forward_fn(xadv)            
                loss = self.loss_fn(output,y)
                
            forward_fn.zero_grad()
            loss.backward()

            g = self.normalize(xadv.grad.data)
            xadv = xadv + self.lr*g
            xadv = self.clip(xadv,x).detach()

        return xadv