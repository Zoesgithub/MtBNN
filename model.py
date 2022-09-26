import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import parse_name
from functools import partial

LogStdIni=-4
Scale=0.001
class BRNN(nn.Module):
    def __init__(self, inputsize, num_units, activation=None, usebayes=True) -> None: # defauls as Normal distribution
        super().__init__()
        self.usebayes=usebayes
        self.w_mean=nn.Parameter(torch.rand(inputsize, 2*num_units))
        self.b_mean=nn.Parameter(torch.zeros(2*num_units))
        self.outw_mean=nn.Parameter(torch.rand(inputsize, num_units))
        self.outb_mean=nn.Parameter(torch.zeros(num_units))
        nn.init.xavier_uniform_(self.w_mean)
        nn.init.xavier_uniform_(self.outw_mean)
        if self.usebayes:
            self.log_w_std=nn.Parameter(torch.ones(inputsize, 2*num_units))
            self.log_b_std=nn.Parameter(torch.ones(2*num_units))
            self.log_outw_std=nn.Parameter(torch.ones(inputsize, num_units))
            self.log_outb_std=nn.Parameter(torch.ones(num_units))

        self.sigmoid=nn.Sigmoid()

        self.activation=activation

    def forward(self, x):
        if self.usebayes:
            w_rand=torch.randn(self.w_mean.shape).cuda()
            b_rand=torch.randn(self.b_mean.shape).cuda()
            outw_rand=torch.randn(self.outw_mean.shape).cuda()
            outb_rand=torch.randn(self.outb_mean.shape).cuda()

            w=w_rand*self.log_w_std*Scale+self.w_mean
            b=b_rand*self.log_b_std*Scale+self.b_mean
            outw=outw_rand*self.log_outw_std*Scale+self.outw_mean
            outb=outb_rand*self.log_outb_std*Scale+self.outb_mean
        else:
            w=self.w_mean
            b=self.b_mean
            outw=self.outw_mean
            outb=self.outb_mean


        inputs, state=x
        if state is None:
            state=inputs
        assert len(inputs.shape)==2
        gate=self.sigmoid(torch.matmul(torch.cat([inputs, state], 1), w)+b)
        r, u=gate[:, :gate.shape[1]//2], gate[:, gate.shape[1]//2:]

        r_state=r*state
        candidate=torch.matmul(torch.cat([inputs, r_state], 1),outw)+outb
        if self.activation is not None:
            c=self.activation(candidate)
        else:
            c=candidate
        out=u*state+(1-u)*c
        return out, out



class BCNN(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride=1, padding=0, usebayes=True) -> None: # defauls as Normal distribution
        super().__init__()
        self.w_mean=nn.Parameter(torch.rand(outc, inc, kernel_size))
        self.b_mean=nn.Parameter(torch.zeros(outc))
        nn.init.xavier_uniform_(self.w_mean)
        if usebayes:
            self.log_w_std=nn.Parameter(torch.ones(outc, inc, kernel_size))
            self.log_b_std=nn.Parameter(torch.ones(outc))
        self.stride=stride
        self.padding=padding
        self.sigmoid=nn.Sigmoid()
        self.usebayes=usebayes

    def forward(self, x):
        if self.usebayes:
            w_rand=torch.randn(self.w_mean.shape).cuda()
            b_rand=torch.randn(self.b_mean.shape).cuda()
            w=w_rand*self.log_w_std*Scale+self.w_mean
            b=b_rand*self.log_b_std*Scale+self.b_mean
        else:
            w=self.w_mean
            b=self.b_mean
        return nn.functional.conv1d(x, w, b, stride=self.stride, padding=self.padding)

class BFC(nn.Module):
    def __init__(self, ins, outs, usebayes=True) -> None: # defauls as Normal distribution
        super().__init__()
        self.w_mean=nn.Parameter(torch.rand(outs, ins))
        self.b_mean=nn.Parameter(torch.zeros(outs))
        nn.init.xavier_uniform_(self.w_mean)
        self.usebayes=usebayes
        if usebayes:
            self.log_w_std=nn.Parameter(torch.ones(outs, ins))
            self.log_b_std=nn.Parameter(torch.ones(outs))
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        if self.usebayes:
            w_rand=torch.randn(self.w_mean.shape).cuda()
            b_rand=torch.randn(self.b_mean.shape).cuda()
            w=w_rand*self.log_w_std*Scale+self.w_mean
            b=b_rand*self.log_b_std*Scale+self.b_mean
        else:
            w=self.w_mean
            b=self.b_mean
        return nn.functional.linear(x, w, b)

class bmodel(nn.Module):
    def __init__(self, embedsize=100, numtask=1, usebayes=True) -> None:
        super().__init__()
        self.embed=nn.Embedding(5, embedsize)
        self.usebayes=usebayes
        if usebayes:
            FC=partial(BFC, usebayes=True)
            rnn=partial(BRNN, usebayes=True)
            CNN=partial(BCNN, usebayes=True)
        else:
            FC=partial(BFC, usebayes=False)
            rnn=partial(BRNN, usebayes=False)
            CNN=partial(BCNN, usebayes=False)

        self.task_z_mean=nn.Embedding(numtask+1, embedsize)
        if self.usebayes:
            self.task_z_logstd=nn.Embedding(numtask+1, embedsize, _weight=torch.ones([numtask+1,embedsize]))
        self.znet=nn.Sequential(FC(embedsize, 256), nn.ReLU(), FC(256, 256), nn.ReLU(), FC(256, 256*2), nn.Sigmoid())
        self.cnnnet=nn.Sequential(
                                    CNN(100, 512, 10), nn.LayerNorm( (512, 991)), nn.ReLU(),
                                    CNN(512, 512, 16),nn.LayerNorm((512, 976)), nn.ReLU(),
                                  CNN(512, 256, 16, stride=5), nn.LayerNorm((256, 193)),nn.ReLU(),
                                  CNN(256, 256, 16), nn.LayerNorm((256, 178)), nn.ReLU(),
                                  CNN(256, 256, 32), nn.LayerNorm((256, 147)), nn.ReLU(),
                                  CNN(256, 256, 32, stride=7), nn.LayerNorm((256, 17)), nn.ReLU())
        self.forwardgru=rnn(256*2, 256, None)
        self.backwardgru=rnn(256*2, 256, None)
        self.outbn=nn.LayerNorm((256*2))
        self.anet=nn.Sequential(FC(256*2, 128), nn.ReLU(), FC(128, 256*2), nn.Softmax(1))

        self.prednet=nn.Sequential(FC(256*2, 256), nn.ReLU(), FC(256, 1))
        self.mutnet=nn.Sequential(FC(256*2*4, 256), nn.ReLU(), FC(256, 1))
        self.sigmoid=nn.Sigmoid()

    def forward(self, seq, task=None, predmut=False, usegate=False):
        seq=self.embed(seq)
        if task is not None:
            z_mean=self.task_z_mean(task)
            if self.usebayes:
                z_std=self.task_z_logstd(task)*Scale
            else:
                z_std=torch.zeros_like(z_mean)
        else:
            z_mean=self.task_z_mean.weight
            if self.usebayes:
                z_std=self.task_z_logstd.weight*Scale
            else:
                z_std=torch.zeros_like(z_mean)
        z=self.znet(torch.randn(z_mean.shape).cuda()*z_std+z_mean)

        seq=self.cnnnet(seq.permute(0,2,1))
        seqforward, stateforward=[], None
        seqbackward, statebackward=[], None

        for i in range(seq.shape[-1]):
            vf, stateforward=self.forwardgru([seq[:, :, i], stateforward])
            vb, statebackward=self.backwardgru([seq[:, :, -i-1], statebackward])
            seqforward.append(vf.unsqueeze(1))
            seqbackward.append(vb.unsqueeze(1))

        ## attention section
        seqforward=torch.cat(seqforward, 1)
        seqbackward=torch.cat(seqbackward[::-1], 1)
        seqfeature=torch.cat([seqforward, seqbackward], 2)
        attention=self.anet(seqfeature)
        seqfeature=self.outbn((seqfeature*attention).sum(1))
        if usegate:
            if task is not None:
                seqfeature=seqfeature*z
            else:
                seqfeature=torch.einsum("bf,lf->bf", seqfeature, z)
        if predmut:
            n=len(seqfeature)
            feature=torch.cat([seqfeature[n//2:]-seqfeature[:n//2],
                               seqfeature[:n//2]-seqfeature[n//2:],
                               seqfeature[n//2:],
                               seqfeature[:n//2]], 1)
            return self.mutnet(feature)

        return self.prednet(seqfeature)


class Model():
    def __init__(self, embedsize, numtask, learning_rate=1e-3, mutscoretype=None, usebayesian=True):
        logger.info("Constructing model with {} tasks and use bayes is {}".format(numtask, usebayesian))
        self.net=bmodel(embedsize, numtask, usebayes=usebayesian).cuda()

        self.usebayesian=usebayesian
        self.sig=nn.Sigmoid()
        def loss_func(pred, label, eps=1e-10):
            pred=self.sig(pred).reshape(-1)
            return -(label*torch.log(pred+eps)+(1-label)*torch.log(1-pred+eps)).mean()
        self.loss=loss_func
        self.optimizer=torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler=None
        self.mutscoretype=mutscoretype
        self.pretrainedstatedict=None

    def get_param_loss(self, bs):
        loss=0
        variance=1 ## noninformative prior
        klweight=1./10**5
        for name, w in self.net.named_parameters():
            if self.pretrainedstatedict is not None and "mutnet" not in name:
                prev=self.pretrainedstatedict[name] # informative prior
            else:
                prev=None
            if w.requires_grad:
                if "std" in name:
                    if prev is None:
                        loss=loss-torch.log(w**2).sum()+(w**2*Scale**2).sum()/variance
                    else:
                        loss=loss-torch.log(w**2).sum()+(w**2/(prev**2+1e-10)).sum()
                elif "mean" in name:
                    if prev is None:
                        loss=loss+(w**2).sum()/variance
                    else:
                        if name.replace("mean", "logstd") in self.pretrainedstatedict:
                            std=self.pretrainedstatedict[name.replace("mean", "logstd")]*Scale
                        else:
                            newname=name.split(".")
                            newname=".".join(newname[:-1]+["log_"+newname[-1].replace("mean", "std")])
                            std=self.pretrainedstatedict[newname]*Scale
                        loss=loss+((w**2-2*prev*w)/(std**2+1e-10)).sum()

        return loss*0.5*klweight

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
        self.pretrainedstatedict=state_dict

    def pretrain_onestep(self, d):
        self.net.train()
        self.optimizer.zero_grad()
        x, task, y=d["x"], d["task"], d["y"]
        pred=self.net.forward(x.cuda().long(), task.cuda().long(), predmut=False, usegate=True)
        loss=self.loss(pred, y.float().cuda())

        if self.usebayesian:
            bloss=self.get_param_loss(len(y))
            tloss=loss+bloss
        else:
            tloss=loss
            bloss=0
        tloss.backward()
        self.optimizer.step()
        return {"pretrainloss":loss.item(), "bloss":bloss.item() if self.usebayesian else bloss}

    def finetune_onestep(self, d):
        self.net.train()
        self.optimizer.zero_grad()
        x, mutx, task, y=d["x"],d["mutx"] ,d["task"], d["y"]
        x=torch.cat([x, mutx], 0)
        if self.mutscoretype=="all":
            pred=self.net.forward(x.cuda().long(), predmut=True, usegate=True)
        elif self.mutscoretype=="generic":
            pred=self.net.forward(x.cuda().long(), predmut=True, usegate=False)
        else:
            assert self.mutscoretype=="single"
            pred=self.net.forward(x.cuda().long(), torch.cat([task.cuda().long(),task.cuda().long()], 0), predmut=True, usegate=True)
        loss=self.loss(pred, y.float().cuda())
        if self.usebayesian:
            bloss=self.get_param_loss(len(y))
            tloss=loss+bloss
        else:
            tloss=loss
            bloss=0
        tloss.backward()
        self.optimizer.step()
        return {"ftloss":loss.item(), "bloss":bloss.item() if self.usebayesian else bloss}

    def eval(self, data, ismut=False):
        Gt, Pred, Task=[], [],[]
        self.net.eval()
        numrep=5
        with torch.no_grad():
            for d in data:
                x, task, y, taskname=d["x"], d["task"], d["y"], d["taskname"]
                if not ismut:
                    if self.usebayesian:
                        pred=sum([self.net.forward(x.cuda().long(), task.cuda().long(), predmut=False, usegate=True) for _ in range(5)])/5.
                        if "mutx" in d:
                            mutx=d["mutx"]
                            mutpred=sum([self.net.forward(mutx.cuda().long(), task.cuda().long(), predmut=False, usegate=True) for _ in range(5)])/5.
                            pred=mutpred-pred
                    else:
                        pred=self.net.forward(x.cuda().long(), task.cuda().long(), predmut=False, usegate=True)
                        if "mutx" in d:
                            mutx=d["mutx"]
                            mutpred=self.net.forward(mutx.cuda().long(), task.cuda().long(), predmut=False, usegate=True)
                            pred=mutpred-pred
                else:
                    mutx=d["mutx"]
                    x=torch.cat([x, mutx], 0)
                    if self.mutscoretype=="all":
                        if self.usebayesian:
                            pred=sum([self.net.forward(x.cuda().long(), predmut=True, usegate=True) for _ in range(numrep)])/numrep
                        else:
                            pred=self.net.forward(x.cuda().long(), predmut=True, usegate=True)
                    elif self.mutscoretype=="generic":
                        if self.usebayesian:
                            pred=sum([self.net.forward(x.cuda().long(), predmut=True, usegate=False) for _ in range(numrep)])/numrep
                        else:
                            pred=self.net.forward(x.cuda().long(), predmut=True, usegate=False)
                    else:
                        if self.usebayesian:
                            pred=sum([self.net.forward(x.cuda().long(), torch.cat([task.cuda().long(),task.cuda().long()], 0), predmut=True, usegate=True) for _ in range(numrep)])/numrep
                        else:
                            pred=self.net.forward(x.cuda().long(), torch.cat([task.cuda().long(),task.cuda().long()], 0), predmut=True, usegate=True)
                if isinstance(y, list):
                    Gt.extend(y)
                else:
                    Gt.extend(y.cpu().numpy().tolist())
                Pred.extend(pred.cpu().numpy().tolist())
                Task.extend(taskname)
        if isinstance(Gt[0], int):
            auc_score=roc_auc_score(Gt, Pred)
            auprc_score=average_precision_score(Gt, Pred)
            logger.info("The auc score is {} auprc is {} for length of {}".format(auc_score, auprc_score, len(Gt)))
            tasksets=set(Task)
            for t in tasksets:
                tgt=[x for x,y in zip(Gt, Task) if y==t]
                tpred=[x for x,y in zip(Pred, Task) if y==t]
                logger.info("For {} the auc score is {} auprc is {} for length of {}".format(parse_name(t), roc_auc_score(tgt, tpred), average_precision_score(tgt, tpred), len(tgt)))
        return Gt, Pred, Task
