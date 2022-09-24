from model import Model
import torch

path = "tasks/HepG2/dnn_pretrain/"
with open("script/HepG2_files", "r") as f:
    Numtask=len(f.readlines())
Seed=49
torch.manual_seed(Seed)
class config:
    log_path = path + "train.log"
    save_path = path
    seed = Seed
    batch_size = 512
    tasklist="script/HepG2_files"
    path="data/HepG2"
    model=Model(100, Numtask, 1e-3, None ,usebayesian=False)
    epoch_num = 10
    num_workers = 5
    state="pretrain"

