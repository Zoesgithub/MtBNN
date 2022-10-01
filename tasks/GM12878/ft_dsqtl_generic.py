from model import Model
import torch
import os

savepath = "tasks/GM12878/bnn_pretrain_scale3/"
with open("script/GM12878_files", "r") as f:
    Numtask=len(f.readlines())
Seed=497
torch.manual_seed(Seed)
class config:
    log_path = savepath + "ft_dsqtl_generic.log"
    save_path = savepath
    seed = Seed
    batch_size = 32
    tasklist="script/GM12878_files"
    path="data/GM12878"
    model=Model(100, Numtask, 1e-3, mutscoretype="generic" ,usebayesian=True)
    epoch_num = 10
    num_workers = 5
    state="ft"
    trainjsonfile="data/GM12878/gm12878_dsqtl.csvjson"
    taskname="wgEncodeOpenChromDnaseGm12878Pk.narrowPeak"
    load_path=os.path.join(savepath, "models", "model.ckpt-best")
