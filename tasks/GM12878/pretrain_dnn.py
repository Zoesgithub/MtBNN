from model import Model
import torch

path = "tasks/GM12878/dnn_pretrain/"
with open("script/GM12878_files", "r") as f:
    Numtask=len(f.readlines())
Seed=49
torch.manual_seed(Seed)
class config:
    log_path = path + "train.log"
    save_path = path
    seed = Seed
    batch_size = 512
    tasklist="script/GM12878_files"
    path="data/GM12878"
    model=Model(100, Numtask, 1e-3, None ,usebayesian=False)
    epoch_num = 10
    num_workers = 5
    state="pretrain"
