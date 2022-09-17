import os
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import torch
import json
from globalconfig import config
from pyfasta import Fasta
import numpy as np
import random

def parse_name(name):
    if "Dnase" in name:
        return "Dnase"
    if "Gm12878" in name:
        name=name.split("Gm12878")[1]
    elif "Hepg2" in name:
        name=name.split("Hepg2")[1]
    idx=1
    while idx<len(name):
        if ord("Z")>=ord(name[idx])>=ord("A"):
            return name[:idx]
        idx+=1
    return name

class MutGenerator(Dataset):
    def __init__(self,  jsonfile, taskname, taskList):
        with open(jsonfile, "r") as f:
            self.data = json.load(f)
        with open(taskList, "r") as f:
            self.tasklist=f.readlines()
            self.tasklist=sorted([x.strip() for x in self.tasklist])
        for idx, v in enumerate(self.tasklist):
            if v==taskname:
                self.taskid=idx
                break
        else:
            assert False, "the task name is not found in task list"
        self.fafile=Fasta(config.HgPath)
        self.taskname=taskname

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        c, p, ref, mut,y=d
        start, end=p-config.SeqLength//2, p+config.SeqLength//2
        seq = self.fafile[c][start:end].upper()
        assert seq[p-start]==ref
        mutseq=seq[:p-start]+mut+seq[p-start+1:]

        seq = np.array([config.SeqTable[_] for _ in seq])
        mutseq = np.array([config.SeqTable[_] for _ in mutseq])

        return {"x": seq, "mutx":mutseq, "y": y, "task":self.taskid, "taskname":self.taskname}


class DataGenerator(Dataset):
    def __init__(self,  taskList, path, endfix):
        with open(taskList, "r") as f:
            self.tasklist=f.readlines()
            self.tasklist=sorted([x.strip() for x in self.tasklist])
            self.tasklist={x:i for i,x in enumerate(self.tasklist)}

        self.data=[]
        for t in self.tasklist:
            with open(os.path.join(path, t.replace("narrowPeak", "json"))+endfix, "r") as f:
                data=json.load(f)
                data=[x+[t] for x in data]
                self.data.extend(data)

        self.fafile=Fasta(config.HgPath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        c, p, y, task=d
        start, end=p-config.SeqLength//2, p+config.SeqLength//2
        seq = self.fafile[c][start:end].upper()
        seq =np.array([config.SeqTable[_] for _ in seq])
        assert len(seq)==1000
        return {"x": seq, "y": y, "task":self.tasklist[task], "taskname":task}

class CrossvalidationGenerator(Dataset):
    def __init__(self,  data, Idxs):
        self.data=data
        self.Idxs=Idxs

    def __len__(self):
        return len(self.Idxs)

    def __getitem__(self, index):
        return self.data[self.Idxs[index]]

class GetSummaryStatisticsCallback():
    def __init__(
        self,
        model,
        train_data,
        validation_data,
        test_data=None,
        mut_data=None,
        model_save_path=None,
        ispretrain=True,
        num_fold=5
    ):
        super().__init__()
        if isinstance(test_data, list):
            if len(test_data) == 0:
                test_data = None
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.mut_data = mut_data
        self.model_save_path = model_save_path
        self.ispretrain=ispretrain
        self.num_fold=num_fold

        if self.model_save_path is not None and not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)

    def train_one_epoch(self):
        tloss = 0

        for i, d in enumerate(self.train_data):
            if self.ispretrain:
                loss = self.model.pretrain_onestep(d)
            else:
                loss=self.model.finetune_onestep(d)
            logout = "|".join([k+":"+str('%.3g' % loss[k]) for k in loss])
            logger.info(logout)

            if i % 1000 == 0 and self.model_save_path is not None:
                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "model.ckpt-temp"))

        if self.model.scheduler is not None:
            for v in self.model.scheduler:
                v.step()
            logger.info("scheduler learning rate to {}".format(self.model.optimizer[0].param_groups[0]['lr']))
        return {"loss": tloss, "val_loss": float("inf")}

    def fine_tuning(self, num_epoch, mutdata, batchsize, load_path):

        self.model.load_state_dict(torch.load(load_path))
        traindata=DataLoader(mutdata, batch_size=batchsize, shuffle=True)
        logger.info("finetuning with train size {}".format(len(traindata)))
        for _ in range(num_epoch):
            for d in traindata:
                loss=self.model.finetune_onestep(d)
                logout = "|".join([k+":"+str('%.3g' % loss[k]) for k in loss])
                logger.info(logout)
        torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "model.ckpt-ft"))
        logger.info("finish finetuing")

    def cross_validation(self, num_epoch, mutdata, batchsize, load_path, num_fold=5):
        mutdatasize=len(mutdata)
        Idxs=list(range(mutdatasize))
        random.seed(321)
        random.shuffle(Idxs)
        split_idxs=[[] for _ in range(num_fold)]
        for idx, v in enumerate(Idxs):
            split_idxs[idx%num_fold].append(v)
        Eval_GT=[]
        Eval_Pred=[]
        Eval_Task=[]
        #Numsteps=500

        for nf in range(num_fold):
            Numiters=0
            self.model.load_state_dict(torch.load(load_path))
            evalidx=split_idxs[nf]
            trainidx=[]
            for i in range(num_fold):
                if i!=nf:
                    trainidx.extend(split_idxs[i])
            evaldata=DataLoader(CrossvalidationGenerator(mutdata, evalidx), batch_size=batchsize, shuffle=False)
            traindata=DataLoader(CrossvalidationGenerator(mutdata, trainidx), batch_size=batchsize, shuffle=True)
            logger.info("Cross validation {} train size {} eval size {}".format(nf, len(traindata), len(evaldata)))

            for _ in range(max(num_epoch, 150//len(traindata))):
                for d in traindata:
                    loss=self.model.finetune_onestep(d)
                    logout = "|".join([k+":"+str('%.3g' % loss[k]) for k in loss])
                    logger.info(logout)
                    Numiters+=1

            gt, pred, task=self.model.eval(evaldata, ismut=True)
            Eval_GT.extend(gt)
            Eval_Pred.extend(pred)
            Eval_Task.extend(task)
        logger.info("AUC is {} AUPRC is {}".format(roc_auc_score(Eval_GT, Eval_Pred), average_precision_score(Eval_GT, Eval_Pred)))
        writeFile([Eval_GT, Eval_Pred, Eval_Task], self.model_save_path+"_"+int(self.model.usebayesian)+"_"+self.model.mutscoretype)



    def fit(self, num_epoch):
        for epoch in range(num_epoch):
            logs = self.train_one_epoch()
            self.on_epoch_end(epoch, logs)

    def _write_line(self, data, epoch):
        GT,Pred,Task=self.model.eval(data, not self.ispretrain)
        auc=roc_auc_score(GT, Pred)
        auprc=average_precision_score(GT, Pred)
        logger.info("The AUC is {} AUPRC is {} for Epoch {}".format(auc, auprc, epoch))

    def on_epoch_end(self, epoch, logs={}):
        # save model

        if self.model_save_path is not None:
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "model.ckpt-{}".format(epoch)))
        # write logs
        logger.info("Epoch: {}".format(epoch))
        logger.info("Validation data:")
        self._write_line(self.validation_data,  epoch)

        if not self.test_data is None:
            logger.info("Test data")
            self._write_line(self.test_data,epoch)

def writeFile(Pred, Path):
    GT, Pred, Task=Pred
    taskset=set(Task)
    taskfiles={f:open(Path+f, "w") for f in taskset}

    for a,b,t in zip(GT, Pred, Task):
        taskfiles[t].writelines("{}_{}\n".format(a,b))
    for f in taskfiles:
        taskfiles[f].close()
