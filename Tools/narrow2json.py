import sys
import os
from globalconfig import config
from pyfasta import Fasta
import random
import json
from loguru import logger
import numpy
from multiprocessing import Pool

Genome=Fasta(config.HgPath)
GenomeGCcontent={}

def GCcontent(key):
    Genome=Fasta(config.HgPath)
    seq=Genome[key]
    name=key
    ## calculate gccontent of genome
    logger.info("calculate gccontent for length of {}".format(len(seq)))
    Length=config.SeqLength
    assert Length%2==0, "The length of sequence should be able to be divide by 2"
    if os.path.exists(name):
        return numpy.memmap(name, dtype="float", shape=(len(seq),),mode='r+')
    rettable=numpy.memmap(name, dtype="float", shape=(len(seq),),mode='w+') #numpy.zeros(len(seq))
    value=0

    for i in range(Length):
        if seq[i].upper()=="G" or seq[i].upper()=="C":
            value+=1
    idx=Length//2
    rettable[idx]=value
    while idx+Length//2<len(seq):
        if idx%100000==0:
            print(idx//100000, len(seq)//100000)
        if seq[idx+Length//2].upper()=="G" or seq[idx+Length//2].upper()=="C":
            value+=1
        if seq[idx-Length//2].upper()=="G" or seq[idx-Length//2].upper()=="C":
            value-=1
        idx+=1
        rettable[idx]=value
    return rettable

def F2D(File):
    ## load narrowPeak file
    with open(File, "r") as f:
        content=f.readlines()
    content=[x.replace("\n", "").split("\t") for x in content if x[0]!="#"]
    return [[x[0], (int(x[1])+int(x[2]))//2] for x in content]

def sample_neg(P, GenomeGCcontent, positiveMark):
    ## sampling neg samples, considering GCcontent, number and overlapping
    Neg=[]
    Mark={}
    for idx, line in enumerate(P):
        c, p=line
        pgc=int(GenomeGCcontent[c][p])
        if pgc not in Mark:
            Mark[pgc]=[]
        Mark[pgc].append(idx)
    samplenum=0
    while len(Neg)<len(P) :
        c=random.choice(list(GenomeGCcontent.keys()))
        idx=random.randint(0, len(GenomeGCcontent[c])-1)
        samplenum+=1
        if idx not in positiveMark[c]:
            npgc=int(GenomeGCcontent[c][idx])
            if npgc in Mark and len(Mark[npgc])>0:
                Neg.append([c, idx])
                positiveMark[c].add(idx)
                Mark[npgc].pop(-1)
                if len(Neg)%100==0:
                    print(len(Neg), len(P))
        if (samplenum>len(P)*5 and len(Neg)>=len(P)-100):
            break
    return Neg

def load_mut(File):
    with open(File, "r") as f:
        content=f.readlines()
    if File.endswith("csv"):
        content=[x.split(",")[:3] for x in content if "#" not in x]
    else:
        content=[x.split("\t") for x in content if "#" not in x]
        content=[[x[0], x[1], "{}-{}-{}-{}".format(x[0], x[1], x[3].upper(), x[4].upper())] for x in content if len(x[0])>0]
    content=[x for x in content if len(x[0])>0]
    for line in content:
        c, p, name=line
        name=name.split("-")
        assert Genome[c][int(p)-1].upper()==name[-2].upper()
    return [[x[0], int(x[1])-1] for x in content]


def main(LFile, path, mutfiles):
    with open(LFile, "r") as f:
        Files=f.readlines()
    Files=[x.strip() for x in Files]
    Data={} ## save positive samples
    PSet={} ## save the excluded sites
    ExcludeSites=set() ## save the excluded sites in mutations
    GCC={} ## save the gc content
    for m in mutfiles:
        logger.info("start handling {}".format(m))
        p=os.path.join(path, m)
        data=load_mut(p)
        for v in data:
            c, p=v
            if c not in PSet:
                PSet[c]=set()
            for i in range(max(p-config.SeqLength, 0), min(p+config.SeqLength, len(Genome[c]))):
                ExcludeSites.add("{}_{}".format(c, i))
                PSet[c].add(i)

    for f in Files:
        p=os.path.join(path, f)
        data=F2D(p)
        Data[os.path.join(path, f.replace("narrowPeak", "json"))]=[]
        logger.info("start handling {}".format(f))
        for line in data:
            c, p=line
            if c not in PSet:
                PSet[c]=set()
            if "{}_{}".format(c, p) not in ExcludeSites and p-config.SeqLength//2>0 and p+config.SeqLength//2<len(Genome[c]): ## if in exclude sites, skip
                for i in range(max(p-config.SeqLength, 0), min(p+config.SeqLength, len(Genome[c]))):
                    PSet[c].add(i)
                Data[os.path.join(path, f.replace("narrowPeak", "json"))].append(line)
        logger.info("finish handling {}".format(f))
    pool=Pool(10)

    pool.map(GCcontent, PSet.keys())
    for key in PSet:
        GCC[key]=GCcontent(key)
    logger.info("start sample neg ...")
    for name in Data.keys():
        data=Data[name]
        random.shuffle(data)
        trainnum, validnum=int(len(data)*0.8), int(len(data)*0.1)
        train, valid, test=data[:trainnum], data[trainnum:trainnum+validnum], data[trainnum+validnum:]
        trainneg=sample_neg(train, GCC, PSet)
        validneg=sample_neg(valid, GCC, PSet)
        testneg=sample_neg(test, GCC, PSet)
        print(len(train), len(trainneg))
        with open(name+"_train", "w") as f:
            f.write(json.dumps([[x[0], x[1], 1] for x in train]+[[x[0], x[1], 0] for x in trainneg]))
        with open(name+"_valid", "w") as f:
            f.write(json.dumps([[x[0], x[1], 1] for x in valid]+[[x[0], x[1], 0] for x in validneg]))
        with open(name+"_test", "w") as f:
            f.write(json.dumps([[x[0], x[1], 1] for x in test]+[[x[0], x[1], 0] for x in testneg]))
        logger.info("finish writing".format(name))




if __name__=="__main__":
    Lfile, path, mutfiles=sys.argv[1:]
    mutfiles=mutfiles.split(",")
    main(Lfile, path, mutfiles)