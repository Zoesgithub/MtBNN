from model import model
import argparse 
import json
import random
import os
import tensorflow as tf
from sklearn.metrics import *

parser=argparse.ArgumentParser()
parser.add_argument("-t",'--test', nargs='*', help="The path to test file, with the same order of train file. Multiple files can be given.", default=None)
parser.add_argument("-s", "--save", help="The path to save models.", default=None)
parser.add_argument("-r","--train", nargs='*', help='The path to train file. Multiple files can be given.', default=None)
parser.add_argument('-n','--snp', help='Whether to calculate snp. Default: False.', type=bool, default=False)
parser.add_argument("-a", "--tasknum", help="The index of task in snp mode.", type=int)
parser.add_argument("-k", "--n_task", help="The number of tasks. This value must be specified in the test and snp mode", type=int, default=21)
parser.add_argument("-p", "--loadpath", help="The path to load model", default=None)
parser.add_argument("-e", "--step", help="Number of training steps", default=6500*10, type=int)
parser.add_argument("-b", "--batchsize", help="Batchsize", type=int, default=400)

args=parser.parse_args()
def parse_data(data):
    diction={'a':0, 'c':1, 'g':2, 't':3}
    data=data.lower()
    try:
        return [diction[i] for i in data]
    except:
        return None

def loaddata(FILELIST,size=1000, snp=False):
    res=[]
    for FILE in FILELIST:
        with open(FILE, 'r') as f:
            content=json.load(f)

        if snp:
            content=[[item[0].lower(), item[1].lower(),item[2], item[-1]] for item in content]
            res.append([[parse_data(item[0]), parse_data(item[1]),item[2], item[-1]] for item  in content 
                if len(item[0])==len(item[1])and len(item[0])==size and parse_data(item[0])!=None and parse_data(item[1])!=None]) 
        else:
            res.append([[parse_data(item['seq']), item['label']] for item in content if len(item['seq'])==size])
    return res

FILELIST=args.train
TESTLIST=args.test
if FILELIST is not None:
    Model=model(len(FILELIST), lr=0.1e-4)
    traindata=loaddata(FILELIST, size=1000, snp=False)
    b=args.batchsize/len(FILELIST)

    Model.train(traindata, args.step, b, args.save, save_step=500, random_neg=True)
else:
    print "#############LOAD EXISTING MODEL##############"
    Model=model(args.n_task, calsnp=args.snp, lr=1e-4)

if TESTLIST is not None:
    testdata=loaddata(TESTLIST, size=1000, snp=args.snp)

if args.snp:
    train=[]
    print len(testdata)
    for item in testdata[:-1]:
        train+=item
    res=Model.calSNP(train, testdata[-1], args.loadpath, task=args.tasknum)
    r=[abs(item[0]) for item in res]
    l=[item[-1] for item in res]
    res=[[res[i], testdata[-1][i][2]] for i in range(len(res))]
    fpr, tpr, t=roc_curve(l,r,pos_label=1)
    print auc(fpr, tpr)
    with open('MBNN_snp_output',"w") as f:
        f.write(json.dumps(res))
elif args.test is not None:
    for i,item in zip(range(len(testdata)),testdata):
        res=Model.test(testdata[i],  args.loadpath,i)
        r=[item[0] for item in res]
        l=[item[-1] for item in res]
        fpr, tpr, t=roc_curve(l, r, pos_label=1)
        auc_=auc(fpr, tpr)
        print auc_
        with open(TESTLIST[i]+'MBNN_test_out', 'w') as f:
            f.write(json.dumps(res))
