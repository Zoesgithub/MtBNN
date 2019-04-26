# MtBNN
The python script for MtBNN. The help information can be obtain by

    python main.py -h

Some toy data are given in *testdata* folder. The format of input files should be the same as these toy examples.

## requirement
Python 2.7.14

tensorflow-gpu==1.4.0

edward==1.3.5

numpy==1.14.5

scipy==1.1.0

scikit-learn==0.20.0

keras==2.1.3

## Train model
example:

    bash python main.py -r testdata/train1 testdata/train2 -s ./Model

Multiple training data can be used here. Each training data is in the json format.

## Test model
example:

    python main.py -t testdata/test1 testdata/test2 -p ./Model/model.ckpt-0 -k 2

In the testing mode, the number of tasks used in the training process must be specified, i.e. the value of k must be given. The order of tasks in the testdata must be the same as that in the traindata.

The test result will be saved with a suffix of *MtBNN_test_out*

## Fine-tuning and validation on SNP data
example:

    python main.py -t testdata/snp_train testdata/snp_test -n True -a 1 -k 2 -p ./Model/model.ckpt-0

    if a==k, MtBNN_ALL will be calculated
    if a<k, MtBNN_SINGLE will be calculated with a-th task-specific feature
    if a==k+1, MtBNN_GENERIC will be calculted

a==k is suggested. In the snp mode, the number of tasks and the index of task must be specified, i.e. the value of k (the number of tasks used in the training process) and the value of a (the index of the task which the snp data is related to) are needed.

The output will be saved with a suffix of *MtBNN_snp_out*
