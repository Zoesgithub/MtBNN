from model import Model
import torch
import os

savepath = "tasks/HepG2/bnn_pretrain_scale3/"
with open("script/HepG2_files", "r") as f:
    Numtask=len(f.readlines())
Seed=49
torch.manual_seed(Seed)
class config:
    log_path = savepath + "train.log"
    save_path = savepath
    seed = Seed
    batch_size = 512
    tasklist="script/HepG2_files"
    path="data/HepG2"
    model=Model(100, Numtask, 1e-3, None ,usebayesian=True)
    epoch_num = 10
    num_workers = 5
    load_path=os.path.join(savepath, "models", "model.ckpt-best")
    state="eval"

