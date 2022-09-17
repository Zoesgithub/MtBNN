'''
This script is used for generating data for Deopen training.
Usage:
    python Gen_data.py  -pos <positive_bed_file> -neg <negative_bed_file> -out <outputfile>
    python Gen_data.py -l 1000 -s 100000 -in <inputfile> -out <outputfile>
'''
from random import sample
import numpy as np
from pyfasta import Fasta
import hickle as hkl
import argparse
import json

#transfrom a sequence to one-hot encoding matrix
def seq_to_mat(seq):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':4, 'N':4}
    mat = np.zeros((len(seq),5))
    for i in range(len(seq)):
        mat[i,encoding_matrix[seq[i]]] = 1
    mat = mat[:,:4]
    return mat

#transform a sequence to K-mer vector (default: K=6)
def seq_to_kspec(seq, K=6):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}
    kspec_vec = np.zeros((4**K,1))
    for i in range(len(seq)-K+1):
        sub_seq = seq[i:(i+K)]
        index = 0
        for j in range(K):
            index += encoding_matrix[sub_seq[j]]*(4**(K-j-1))
        kspec_vec[index] += 1
    return kspec_vec


#assemble all the features into a dictionary
def get_all_feats(spot,genome,label):
    ret = {}
    ret['spot'] = spot
    ret['seq'] = genome[spot[0]][spot[1]:spot[2]]
    ret['mat'] = seq_to_mat(ret['seq'])
    ret['kmer'] = seq_to_kspec(ret['seq'])
    ret['y'] = label
    return ret

def get_mut_feats(spot,genome,label, ref, mut):
    ret = {}
    mutret={}
    ret['spot'] = spot
    mutret["spot"]=spot
    assert ref.upper()==genome[spot[0]][(spot[1]+spot[2])//2].upper()
    ret['seq'] = genome[spot[0]][spot[1]:spot[2]]
    mutret["seq"]=genome[spot[0]][spot[1]:(spot[1]+spot[2])//2]+mut+genome[spot[0]][(spot[1]+spot[2])//2+1:spot[2]]
    ret['mat'] = seq_to_mat(ret['seq'])
    mutret["mat"]=seq_to_mat(mutret["seq"])
    ret['kmer'] = seq_to_kspec(ret['seq'])
    mutret["kmer"]=seq_to_kspec(mutret["seq"])
    ret['y'] = label
    mutret["y"]=label
    return ret, mutret

#save the preprocessed data in hkl format
def  save_dataset(origin_dataset,save_dir):
    dataset = {}
    for key in origin_dataset[0].keys():
        dataset[key] = [item[key] for item in origin_dataset]
    dataset['seq'] = [item.encode('ascii') for item in dataset['seq']]
    for key in origin_dataset[0].keys():
        dataset[key] = np.array(dataset[key])

        if key=="seq":
            dataset[key].astype('S1')
        elif key!='spot':
            dataset[key].astype(int)
    hkl.dump(dataset,save_dir, mode='w', compression='gzip')
    print ('Training data generation is finished!')


#generate dataset
def  generate_dataset(inp_file,genome_file,dataset, sample_length = 1000):
    genome = Fasta(genome_file)
    with open(inp_file, "r") as f:
        content=json.load(f)
        for line in content:

            if len(line)==3:
                c, p, y=line

                task="total"
                if task not in dataset:
                    dataset[task]=[]
                dataset[task].append(get_all_feats([c, p-sample_length//2, p+sample_length//2], genome, y))
            else:
                assert len(line)==5
                c, p, ref, mut,y=line

                task="total"
                if task not in dataset:
                    dataset[task]=[]
                dataset[task].extend(get_mut_feats([c, p-sample_length//2, p+sample_length//2], genome, y, ref, mut))

    return  dataset


if  __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Deopen data generation')
    parser.add_argument('-inp', dest='inp',nargs="*", type=str, help='input json file')
    parser.add_argument('-genome', dest='genome', type=str, help='genome file in fasta format')
    parser.add_argument('-l', dest='length', type=int, default=1000, help='sequence length')
    parser.add_argument('-out', dest='output', type=str, help='output file')
    args = parser.parse_args()
    dataset={}
    for d in args.inp:
        print(d)
        generate_dataset(d, args.genome, dataset, args.length)
    assert len(dataset)==1
    for key in dataset:
        save_dataset(dataset[key],args.output)


