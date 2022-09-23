'''
This script is used for running Deopen classification model.
Usage:
    THEANO_FLAGS='device=gpu,floatX=float32' python Deopen_classification.py -in <inputfile> -out <outputfile>

    inputfile.hkl -- preprocessed file containing different features (hkl format)
    outputfile -- trained model to be saved (hkl format)
'''

import hickle as hkl
import argparse
from loguru import logger
from sklearn import metrics
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, DataLoader
import numpy as np
class dataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        assert self.y[index]==0 or self.y[index]==1
        return [self.x[index], self.y[index]]

#split the data into training set, testing set
def data_split(inputfile):
    data = hkl.load(inputfile)
    X = data['mat']
    X_kspec = data['kmer']
    y = data['y'].astype(int)

    X_kspec = X_kspec.reshape((X_kspec.shape[0],1024,4))
    X = np.concatenate((X,X_kspec), axis = 1)
    X = X[:,np.newaxis]
    X = X.transpose((0,1,3,2)).astype("float32")

    return [X, y]

#define the network architecture
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = 1000
        pool_size = 5
        test_size1 = 13
        test_size2 = 7
        test_size3 = 5
        kernel1 = 128
        kernel2 = 128
        kernel3 = 128
        self.net_21=nn.Sequential(
            nn.Conv2d(1, kernel1, (4, test_size1)),
            nn.ReLU(),
            nn.Conv2d(kernel1, kernel1, (1, test_size1)),
            nn.ReLU(),
            nn.Conv2d(kernel1, kernel1, (1, test_size1)),
            nn.ReLU(),
            nn.MaxPool2d((1, pool_size)),
            nn.Conv2d(kernel1, kernel2, (1, test_size2)),
            nn.ReLU(),
            nn.Conv2d(kernel2, kernel2, (1, test_size2)),
            nn.ReLU(),
            nn.Conv2d(kernel2, kernel2, (1, test_size2)),
            nn.ReLU(),
            nn.MaxPool2d((1, pool_size)),
            nn.Conv2d(kernel2, kernel3, (1, test_size3)),
            nn.ReLU(),
            nn.Conv2d(kernel3, kernel3,(1, test_size3)),
            nn.ReLU(),
            nn.Conv2d(kernel3, kernel3,(1, test_size3)),
            nn.ReLU(),
            nn.MaxPool2d((1, pool_size)),
            )

        self.net_v2=nn.Linear(4*1024, 128)
        self.linear_v1=nn.Linear(kernel3*4, 256)
        self.outnet=nn.Sequential(nn.Dropout(0.5), nn.Linear(128+256, 256),nn.ReLU(), nn.Linear(256, 2), nn.Softmax(-1))
        self.accscore=[]

    def forward(self, x):
        assert x.shape[-1]==self.l+1024
        inp1=self.linear_v1(self.net_21(x[:, :, :, :self.l]).squeeze(2).reshape(-1, 128*4))
        inp2=self.net_v2(x[:, :, :, self.l:].reshape(-1, 1024*4))
        return self.outnet(torch.cat([inp1, inp2], 1))

def crossentropy(p, y):
    assert len(y.shape)==1
    assert len(p.shape)==2
    y=y.long()
    y=nn.functional.one_hot(y, 2).float()
    ret=-(y*torch.log(p+1e-10)).sum(-1).mean()
    return ret



class trainframe():
    def __init__(self) -> None:
        self.net=Model().cuda()
        self.optimizer=torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.loss=crossentropy
        self.ACC=-float("inf")
        self.bestmodel=None

    def eval(self, data, model):
        model.eval()
        GT, Pred=[], []
        for d in data:
            x, y=d
            with torch.no_grad():
                p=model.forward(x.cuda()).detach()
            assert len(y.shape)==1
            GT.extend(y.cpu().numpy().tolist())
            Pred.extend(p.cpu().numpy()[:, 1].tolist())
        return GT, Pred, sum([int(a==int(b>0.5)) for a,b in zip(GT, Pred)])/len(GT)

    def train_epochs(self, train, val, num_epochs):
        for epoch in range(num_epochs):
            for idx, d in enumerate(train):
                self.optimizer.zero_grad()
                self.net.train()
                x, y=d
                p=self.net(x.cuda())
                loss=self.loss(p, y.cuda())
                loss.backward()
                self.optimizer.step()
                if idx%100==0:
                    logger.info("Loss is {}".format(loss.item()))
            _, _, acc=self.eval(val, self.net)
            logger.info("ACC in epoch {} is {}".format(epoch, acc))

            if acc>self.ACC:
                self.ACC=acc
                self.bestmodel=copy.deepcopy(self.net)
                logger.info("save best model in epoch {} with acc {}".format(epoch, acc))
        return self.bestmodel



if  __name__ == "__main__" :
    import os
    parser = argparse.ArgumentParser(description='Deopen classication model')
    parser.add_argument('-in', dest='input', type=str, help='inputfile')
    parser.add_argument('-val', dest='val', type=str, help='validfile')
    parser.add_argument('-out', dest='output', type=str, help='outputfile')
    parser.add_argument('-test',nargs='*', dest='testfile', help='testfile')
    parser.add_argument('-testout',nargs='*', dest='testoutfile',  help='testfile')
    args = parser.parse_args()
    print(args.testfile)
    print(args.input)
    assert os.path.exists(args.input), "train file not exists!"
    X_train, y_train = data_split(args.input)
    x_val, y_val=data_split(args.val)
    model=trainframe()
    numsteps=2000//(len(X_train)//32)
    bestmodel=model.train_epochs(DataLoader(dataset(X_train, y_train), batch_size=32, shuffle=True), DataLoader(dataset(x_val, y_val), batch_size=64), max(55, numsteps))

    if args.testfile!='':
        for i,j in zip(args.testfile, args.testoutfile):
            X_test, y_test = data_split(i)
            GT, PRED, _=model.eval(DataLoader(dataset(X_test, y_test), batch_size=64), bestmodel)
            logger.info("AUC for {} is {}".format(args.testoutfile, metrics.roc_auc_score(GT, PRED)))
            with open(j, "w") as f:
                for a,b in zip(GT, PRED):
                    f.writelines("{}\t{}\n".format(a,b))



