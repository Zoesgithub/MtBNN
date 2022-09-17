import numpy as np
class config:
    SeqLength=1000
    HgPath="/data2/xcc/code/eSplice_baseline/MMSPLICE/fafiles/hg19.fa"
    IN_MAP=np.eye(5)[:, :4]
    SeqTable={"A":0, "C":1, "G":2, "T":3, "N":4}