# MtBNN
The objective function of MtBNN is the ELBO

<img align="center" src="https://i.upmath.me/svg/%0A%5Cnonumber%0A%26%26%20%5Clog%20p(%5Cbm%7BD%7D)-%5Ctextrm%7BKL%7D%20%5Cleft%5B%20q(%5Cbm%7BW%7D%2C%20%5Cbm%7Bz%7D%2C%20%5Calpha)%20%5C%7C%20p(%5Cbm%7BW%7D%2C%20%5Cbm%7Bz%7D%2C%20%5Calpha%20%7C%20%5Cbm%7BD%7D)%20%5Cright%5D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%20%5Clog%20p(%5Cbm%7BD%7D)-%5Cmathbb%7BE%7D_q%20%5Clog%20%5Cfrac%7Bq(%5Cbm%7BW%7D%7C%20%5Cbm%7Bz%7D%2C%5Calpha)q(%5Cbm%7Bz%7D)%7D%20%7Bp(%5Cbm%7BW%7D%2C%20%5Cbm%7Bz%7D%7C%20%20%5Cbm%7BD%7D%2C%5Calpha)%7D-%5Cmathbb%7BE%7D_q%20%5Clog%20%5Cfrac%7Bq(%5Calpha)%7D%7Bp(%5Calpha%7C%20%5Cbm%7BD%7D)%7D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%5Clog%20p(%5Cbm%7BD%7D)-%20%5Cmathbb%7BE%7D_q%20%5Clog%20%5Cfrac%7Bq(%5Cbm%7BW%7D%7C%20%5Cbm%7Bz%7D%2C%5Calpha)q(%5Cbm%7Bz%7D)p(%5Cbm%7BD%7D%7C%5Calpha)%7D%20%7Bp(%5Cbm%7BW%7D%2C%20%5Cbm%7Bz%7D%2C%20%5Cbm%7BD%7D%7C%20%5Calpha)%7D-%5Cmathbb%7BE%7D_q%20%5Clog%20%5Cfrac%7Bq(%5Calpha)%7D%7Bp(%5Calpha%7C%20%5Cbm%7BD%7D)%7D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%20%5Clog%20p(%5Cbm%7BD%7D)-%5Csum_i%20%5Cmathbb%7BE%7D_%7Bq%7D%20%5Clog%20%5Cfrac%7B%20q(W_i%7Cz_i%2C%20%5Calpha)q(z_i)%7D%20%7Bp(W_i%2Cz_i%2C%20D_i%7C%5Calpha)%7D-%5Cmathbb%7BE%7D_%7Bq(%5Calpha)%20%7D%20%5Clog%20%5Cfrac%7Bq(%5Calpha)p(%5Cbm%7BD%7D%7C%5Calpha)%7D%7Bp(%5Calpha%7C%5Cbm%7BD%7D)%7D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%20%5Clog%20p(%5Cbm%7BD%7D)-%5Csum_i%20%5Cmathbb%7BE%7D_%7Bq%7D%20%5Clog%20%5Cfrac%7B%20q(W_i%7Cz_i%2C%5Calpha)q(z_i)%7D%20%7Bp(D_i%7CW_i)p(W_i%7Cz_i%2C%5Calpha)p(z_i)%7D-%5Cmathbb%7BE%7D_%7Bq(%5Calpha)%20%7D%20%5Clog%20%5Cfrac%7Bq(%5Calpha)p(%5Cbm%7BD%7D%7C%5Calpha)%7D%7Bp(%5Calpha%7C%5Cbm%7BD%7D)%7D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%20%5Clog%20p(%5Cbm%7BD%7D)-%5Csum_i%20%5Cmathbb%7BE%7D_%7Bq(z_i)%7D%20%5Clog%20%5Cfrac%7Bq(z_i)%7D%20%7Bp(D_i%7CW_i)p(z_i)%7D-%20%5Cmathbb%7BE%7D_%7Bq(%5Calpha)%20%7D%20%5Clog%20%5Cfrac%7Bq(%5Calpha)%7D%7Bp(%5Calpha)%7D-%5Clog%20p(%5Cbm%7BD%7D)%5C%5C%0A%25%5Cnonumber%20%0A%25%5Cnonumber%0A%26%3D%26-%5Csum_i%20%5Ctextrm%7BELBO%7D_i-%5Cmathbb%7BE%7D_%7Bq(%5Calpha)%7D%5Clog%20%5Cfrac%7Bq(%5Calpha)%7D%7Bp(%5Calpha)%7D%0A" alt="
\nonumber
&amp;&amp; \log p(\bm{D})-\textrm{KL} \left[ q(\bm{W}, \bm{z}, \alpha) \| p(\bm{W}, \bm{z}, \alpha | \bm{D}) \right] \\
\nonumber
&amp;=&amp; \log p(\bm{D})-\mathbb{E}_q \log \frac{q(\bm{W}| \bm{z},\alpha)q(\bm{z})} {p(\bm{W}, \bm{z}|  \bm{D},\alpha)}-\mathbb{E}_q \log \frac{q(\alpha)}{p(\alpha| \bm{D})} \\
\nonumber
&amp;=&amp;\log p(\bm{D})- \mathbb{E}_q \log \frac{q(\bm{W}| \bm{z},\alpha)q(\bm{z})p(\bm{D}|\alpha)} {p(\bm{W}, \bm{z}, \bm{D}| \alpha)}-\mathbb{E}_q \log \frac{q(\alpha)}{p(\alpha| \bm{D})} \\
\nonumber
&amp;=&amp; \log p(\bm{D})-\sum_i \mathbb{E}_{q} \log \frac{ q(W_i|z_i, \alpha)q(z_i)} {p(W_i,z_i, D_i|\alpha)}-\mathbb{E}_{q(\alpha) } \log \frac{q(\alpha)p(\bm{D}|\alpha)}{p(\alpha|\bm{D})} \\
\nonumber
&amp;=&amp; \log p(\bm{D})-\sum_i \mathbb{E}_{q} \log \frac{ q(W_i|z_i,\alpha)q(z_i)} {p(D_i|W_i)p(W_i|z_i,\alpha)p(z_i)}-\mathbb{E}_{q(\alpha) } \log \frac{q(\alpha)p(\bm{D}|\alpha)}{p(\alpha|\bm{D})} \\
\nonumber
&amp;=&amp; \log p(\bm{D})-\sum_i \mathbb{E}_{q(z_i)} \log \frac{q(z_i)} {p(D_i|W_i)p(z_i)}- \mathbb{E}_{q(\alpha) } \log \frac{q(\alpha)}{p(\alpha)}-\log p(\bm{D})\\
%\nonumber 
%\nonumber
&amp;=&amp;-\sum_i \textrm{ELBO}_i-\mathbb{E}_{q(\alpha)}\log \frac{q(\alpha)}{p(\alpha)}
" />

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
