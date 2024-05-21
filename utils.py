import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from pgd import PGD
from trades import trades_loss
from torch.utils.data import DataLoader
    
class TrainValHandler():
    def __init__(self, model, device, trainset, testset, batchsize, lr, epochs, patience, path, mode, hyper) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.trainloader = DataLoader(trainset,batch_size=batchsize,shuffle=True)
        self.testloader = DataLoader(testset,batch_size=batchsize,shuffle=True)
        self.loss = CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.patience = patience
        self.min_loss = torch.inf
        self.path = path
        self.mode = mode
        if self.mode == "at":
            self.pgd = PGD(advtrain=True)
        self.hyper = hyper
        
    def saveModel(self):
        torch.save(self.model.state_dict(),self.path)

    def train_one_epoch(self):
        
        train_loss = 0
        self.model.train()
        for i, (x,y) in enumerate(self.trainloader):
            x = x.to(self.device,dtype=torch.float)
            y = y.to(self.device,dtype=torch.float)
            if self.mode == "trades":
                loss = trades_loss(self.model,x,y,hypercnn=self.hyper)
            elif self.mode == "at":
                self.model.eval()
                x = self.pgd(self.model,x,y,hyper=self.hyper)
                self.model.train()
                if self.hyper:
                    preds = self.model(x,True)
                    loss = torch.zeros(len(preds)).to(self.device)
                    for j in range(len(preds)):
                        loss[j] += self.loss(preds[j],y)
                    loss = loss.mean()
                else:
                    pred = self.model(x)
                    loss = self.loss(pred,y)
            else:
                if self.hyper:
                    preds = self.model(x,True)
                    loss = torch.zeros(len(preds)).to(self.device)
                    for j in range(len(preds)):
                        loss[j] += self.loss(preds[j],y)
                    loss = loss.mean()
                else:
                    pred = self.model(x)
                    loss = self.loss(pred,y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            train_loss += loss.item()
            print('batch %d/%d, training loss %.6f' % (i+1,len(self.trainloader),train_loss/(i+1)),end='\r')
        loss = 0
        return train_loss

    def val_one_epoch(self):

        self.model.eval()
        val_loss = 0
        
        for i, (x,y) in enumerate(self.testloader):
            x = x.to(self.device,dtype=torch.float)
            y = y.to(self.device,dtype=torch.float)
            if self.mode == "trades":
                loss = trades_loss(self.model,x,y,mode='val',hypercnn=self.hyper)
            elif self.mode == "at":
                if self.hyper:
                    x = self.pgd(self.model,x,y,True)
                    with torch.no_grad():
                        preds = self.model(x,True)
                    loss = torch.zeros(len(preds)).to(self.device)
                    for j in range(len(preds)):
                        loss[j] = self.loss(preds[j],y)
                    loss = loss.mean()
                else:
                    x = self.pgd(self.model,x,y)
                    with torch.no_grad():
                        pred = self.model(x)
                    loss = self.loss(pred,y)
            else:
                if self.hyper:
                    with torch.no_grad():
                        preds = self.model(x,True)
                    loss = torch.zeros(len(preds)).to(self.device)
                    for j in range(len(preds)):
                        loss[j] = self.loss(preds[j],y)
                    loss = loss.mean()
                else:
                    with torch.no_grad():
                        pred = self.model(x)
                        loss = self.loss(pred,y)
            val_loss += loss.item()
        loss = 0
        return val_loss

    def train(self):
        self.model.to(self.device)
        patience = 0
        history = {
            "training loss":[],
            "validation loss":[]
        }
        for epoch in np.arange(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.val_one_epoch()
            # self.scheduler.step()
            history["training loss"].append(train_loss/len(self.trainloader))
            history["validation loss"].append(val_loss/len(self.testloader))
            print('epoch %d / %d, training loss: %.6f, validation loss: %.6f' % 
                  (epoch+1, self.epochs, train_loss/len(self.trainloader), val_loss/len(self.testloader)))
            if val_loss < self.min_loss:
                self.min_loss = val_loss
                patience = 0
                self.saveModel()
                print('save best model at epoch %d' % (epoch+1))
            else:
                patience += 1

            if patience == self.patience:
                print('no improvement from last %d epoch, stop training' % patience)
                break
        return history
    
class EvalHandler():
    def __init__(self, model, device, testset, batchsize, hyper) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.testset = testset
        self.batchsize = batchsize
        self.test_lodaer = DataLoader(testset,batch_size=batchsize,shuffle=True)
        self.hyper = hyper
        
    def test_one_step(self,x,y):

        self.model.eval()
        x = x.to(self.device,dtype=torch.float)
        y = y.to(self.device,dtype=torch.float)
        if self.hyper:
            with torch.no_grad():
                pred = self.model(x,False,False)
        else:
            with torch.no_grad():
                pred = self.model(x)
        correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        return correct
    
    def test(self):
        correct = 0
        for i, (x,y) in enumerate(self.test_lodaer):
            correct += self.test_one_step(x,y)
            print('batch %d/%d, test acc %.6f' % (i+1,len(self.testset)/self.batchsize,correct/(i*self.batchsize+self.batchsize)),end='\r')
        return correct
    
    def attack(self,eps=0.05,n_iter=10,nsamples=5000):
        atk = PGD(eps=eps,n_iter=n_iter)
        samples = 0
        correct = 0
        for x,y in self.testset:
            x = torch.Tensor(x)[None].to(self.device,dtype=torch.float)
            y = torch.Tensor(y)[None].to(self.device,dtype=torch.float)
            xadv = atk(self.model,x,y,self.hyper)
            if self.hyper:
                with torch.no_grad():
                    pred = self.model(xadv,False,False)
            else:
                with torch.no_grad():
                    pred = self.model(xadv)
            samples += 1
            correct += pred.argmax(1) == y.argmax(1)
            print('samples %d/%d, robust acc %.6f' % (samples,nsamples,correct/samples),end='\r')
            if samples == nsamples:
                break
        return correct