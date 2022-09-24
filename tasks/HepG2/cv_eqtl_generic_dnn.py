from model import Model
import torch
import os

savepath = "tasks/HepG2/dnn_pretrain/"
with open("script/HepG2_files", "r") as f:
    Numtask=len(f.readlines())
Seed=49
torch.manual_seed(Seed)
class config:
    log_path = savepath + "cv_generic_dnn.log"
    save_path = savepath
    seed = Seed
    batch_size = 32
    tasklist="script/HepG2_files"
    path="data/HepG2"
    model=Model(100, Numtask, 1e-3, mutscoretype="generic" ,usebayesian=False)
    epoch_num = 10
    num_workers = 5
    state="cv"
    trainjsonfile="data/HepG2/eqtl_hepg2.csvjson"
    taskname="wgEncodeOpenChromDnaseHepg2Pk.narrowPeak"
    load_path=os.path.join(savepath, "models", "model.ckpt-best")
