import sys
from pyfasta import Fasta
from globalconfig import config
import json

Genome=Fasta(config.HgPath)
def load_mut(File):
    with open(File, "r") as f:
        content=f.readlines()
    if File.endswith("csv"):
        content=[x.split(",") for x in content if "#" not in x]
    else:
        content=[x.split("\t") for x in content if "#" not in x]
        content=[[x[0], x[1], "{}-{}-{}-{}".format(x[0], x[1], x[3].upper(), x[4].upper()), x[-1]] for x in content if len(x[0])>0]
    content=[x for x in content if len(x[0])>0]
    for line in content:
        c, p, name=line[:3]
        name=name.split("-")
        assert Genome[c][int(p)-1].upper()==name[-2].upper()
    return [[x[0], int(x[1])-1, x[2].split("-")[-2].upper(), x[2].split("-")[-1].upper(), max(int(x[-1]), 0)] for x in content]

def main(Path):
    data=load_mut(Path)
    with open(Path+"json", "w") as f:
        f.write(json.dumps(data))

if __name__=="__main__":
    main(sys.argv[1])
