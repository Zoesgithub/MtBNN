from model import Model
import torch
import os

savepath = "tasks/TCell/bnn_pretrain_scale3/"
with open("script/TCell_files", "r") as f:
    Numtask=len(f.readlines())
Seed=49
torch.manual_seed(Seed)
class config:
    log_path = savepath + "cv_all.log"
    save_path = savepath
    seed = Seed
    batch_size = 32
    tasklist="script/TCell_files"
    path="data/TCell"
    model=Model(100, Numtask, 1e-3, mutscoretype="all" ,usebayesian=True)
    epoch_num = 10
    num_workers = 5
    state="cv"
    trainjsonfile="data/TCell/atacqtl_tcell.txtjson"
    taskname="tcell_atac.txt"
    load_path=os.path.join(savepath, "models", "model.ckpt-best")
