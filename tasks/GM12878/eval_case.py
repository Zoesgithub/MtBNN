from model import Model
import torch
import os

savepath = "tasks/GM12878/bnn_pretrain_scale3/gm12878_dsqtl.csvjson/"
with open("script/GM12878_files", "r") as f:
    Numtask=len(f.readlines())
Seed=49
torch.manual_seed(Seed)
class config:
    log_path = savepath + "ft_dsqtl_generic.log"
    save_path = savepath
    seed = Seed
    batch_size = 32
    tasklist="script/GM12878_files"
    path="data/GM12878"
    model=Model(100, Numtask, 1e-3, mutscoretype="generic" ,usebayesian=True)
    epoch_num = 5
    num_workers = 5
    state="evmut"
    trainjsonfile=[os.path.join("data/CaseStudy/Disease/", x) for x in os.listdir("data/CaseStudy/Disease/") if x.endswith("json")]
    taskname="wgEncodeOpenChromDnaseGm12878Pk.narrowPeak"
    load_path=os.path.join(savepath,  "model.ckpt-ft")
