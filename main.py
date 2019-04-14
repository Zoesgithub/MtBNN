from model import model
import argparse 
import json
import random
import os
import tensorflow as tf

parser=argparse.ArgumentParser()
parser.add_argument("-t",'--test', nargs='*', help="the path to test file", default=None)
parser.add_argument("-f",'--file',  help="the name to test file", default=None)
parser.add_argument("-s", "--save", help="the save path", default=None)
parser.add_argument("-r","--train", nargs='*', help='the path to train file', default=None)
parser.add_argument('-n','--snp', help='whether to cal snp', type=bool, default=False)
parser.add_argument("-a", "--tasknum", help="", type=int)
parser.add_argument("-k", "--n_task", help="", type=int, default=21)
parser.add_argument("-d", "--random_neg", help="", type=int, default=1)
parser.add_argument("-p", "--loadpath", help="path to load model", default='/home/xuchencheng/Data/Bayes_NN/merge_random_neg_1214/model.ckpt-20500')
args=parser.parse_args()

def parse_data(data):
    diction={'a':0, 'c':1, 'g':2, 't':3}
    data=data.lower()
    try:
        return [diction[i] for i in data]
    except:
        print data
        return None

def loaddata(FILELIST,size=1000, snp=False):
    res=[]
    for FILE in FILELIST:
        with open(FILE, 'r') as f:
            content=json.load(f)
            if len(content)>80000:
                pastdict={}
                content_=[]
                while len(content_)<80000:
                    p=random.randint(0, len(content)-1)
                    if not pastdict.has_key(p):
                        content_.append(content[p])
                        pastdict[p]=1
                content=content_

        if snp:
            content=[[item[0].lower(), item[1].lower(),item[2], item[-1]] for item in content]
            res.append([[parse_data(item[0]), parse_data(item[1]),item[2], item[-1]] for item  in content 
                if len(item[0])==len(item[1])and len(item[0])==size and parse_data(item[0])!=None and parse_data(item[1])!=None]) 
        else:
            res.append([[parse_data(item['seq']), item['label']] for item in content if len(item['seq'])==size])
    return res
FILELIST=args.train
TESTLIST=args.test
print TESTLIST
if FILELIST is not None:
    if args.random_neg:
        print "use random neg"
    else:
        print "not use random neg"
    Model=model(len(FILELIST), lr=0.1e-4)
    traindata=loaddata(FILELIST, size=1000, snp=False)
    b=200*2/len(FILELIST)
    if args.random_neg>0:
        rn=True
    else:
        rn=False
    print "########################TRAIN DATA:",len(traindata),'###############################'
    Model.train(traindata, 6500*10, b, args.save, save_step=500, random_neg=rn)
else:
    print "load model"
    Model=model(args.n_task, calsnp=args.snp, lr=1e-4)
if TESTLIST is not None:
    testdata=loaddata(TESTLIST, size=1000, snp=args.snp)
from sklearn.metrics import *
if args.snp:
    train=[]
    print len(testdata)
    for item in testdata[:-1]:
        train+=item
    res=Model.calSNP(train, testdata[-1], args.loadpath, task=args.tasknum)
    #Model=model(14, calsnp=args.snp)
    #res.extend(res_)
    r=[abs(item[0]) for item in res]
    l=[item[-1] for item in res]
    res=[[res[i], testdata[-1][i][2]] for i in range(len(res))]
    fpr, tpr, t=roc_curve(l,r,pos_label=1)
    print auc(fpr, tpr)
    with open('MBNN_snp_output',"w") as f:
        f.write(json.dumps(res))
else:
    import edward as ed
    print len(testdata)
    for i,item in zip(range(len(testdata)),testdata):
        res=Model.test(testdata[i],  args.loadpath,i)
        r=[item[0] for item in res]
        l=[item[-1] for item in res]
        fpr, tpr, t=roc_curve(l, r, pos_label=1)
        auc_=auc(fpr, tpr)
        print auc_
        with open(TESTLIST[i]+'MBNN_test_out'+args.file, 'w') as f:
            f.write(json.dumps(res))
