# MtBNN
The python script for MtBNN. The objective function of MtBNN is the ELBO
    <img align="center" src="https://i.upmath.me/svg/%0A%5Cnonumber%0A%26%26%20%5Clog%20p(%5Cbm%7BD%7D)-%5Ctextrm%7BKL%7D%20%5Cleft%5B%20q(%5Cbm%7BW%7D%2C%20%5Cbm%7Bz%7D%2C%20%5Calpha)%20%5C%7C%20p(%5Cbm%7BW%7D%2C%20%5Cbm%7Bz%7D%2C%20%5Calpha%20%7C%20%5Cbm%7BD%7D)%20%5Cright%5D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%20%5Clog%20p(%5Cbm%7BD%7D)-%5Cmathbb%7BE%7D_q%20%5Clog%20%5Cfrac%7Bq(%5Cbm%7BW%7D%7C%20%5Cbm%7Bz%7D%2C%5Calpha)q(%5Cbm%7Bz%7D)%7D%20%7Bp(%5Cbm%7BW%7D%2C%20%5Cbm%7Bz%7D%7C%20%20%5Cbm%7BD%7D%2C%5Calpha)%7D-%5Cmathbb%7BE%7D_q%20%5Clog%20%5Cfrac%7Bq(%5Calpha)%7D%7Bp(%5Calpha%7C%20%5Cbm%7BD%7D)%7D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%5Clog%20p(%5Cbm%7BD%7D)-%20%5Cmathbb%7BE%7D_q%20%5Clog%20%5Cfrac%7Bq(%5Cbm%7BW%7D%7C%20%5Cbm%7Bz%7D%2C%5Calpha)q(%5Cbm%7Bz%7D)p(%5Cbm%7BD%7D%7C%5Calpha)%7D%20%7Bp(%5Cbm%7BW%7D%2C%20%5Cbm%7Bz%7D%2C%20%5Cbm%7BD%7D%7C%20%5Calpha)%7D-%5Cmathbb%7BE%7D_q%20%5Clog%20%5Cfrac%7Bq(%5Calpha)%7D%7Bp(%5Calpha%7C%20%5Cbm%7BD%7D)%7D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%20%5Clog%20p(%5Cbm%7BD%7D)-%5Csum_i%20%5Cmathbb%7BE%7D_%7Bq%7D%20%5Clog%20%5Cfrac%7B%20q(W_i%7Cz_i%2C%20%5Calpha)q(z_i)%7D%20%7Bp(W_i%2Cz_i%2C%20D_i%7C%5Calpha)%7D-%5Cmathbb%7BE%7D_%7Bq(%5Calpha)%20%7D%20%5Clog%20%5Cfrac%7Bq(%5Calpha)p(%5Cbm%7BD%7D%7C%5Calpha)%7D%7Bp(%5Calpha%7C%5Cbm%7BD%7D)%7D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%20%5Clog%20p(%5Cbm%7BD%7D)-%5Csum_i%20%5Cmathbb%7BE%7D_%7Bq%7D%20%5Clog%20%5Cfrac%7B%20q(W_i%7Cz_i%2C%5Calpha)q(z_i)%7D%20%7Bp(D_i%7CW_i)p(W_i%7Cz_i%2C%5Calpha)p(z_i)%7D-%5Cmathbb%7BE%7D_%7Bq(%5Calpha)%20%7D%20%5Clog%20%5Cfrac%7Bq(%5Calpha)p(%5Cbm%7BD%7D%7C%5Calpha)%7D%7Bp(%5Calpha%7C%5Cbm%7BD%7D)%7D%20%5C%5C%0A%5Cnonumber%0A%26%3D%26%20-%5Csum_i%20%5Cmathbb%7BE%7D_%7Bq(z_i)%7D%20%5Clog%20%5Cfrac%7Bq(z_i)%7D%20%7Bp(D_i%7CW_i)p(z_i)%7D-%20%5Cmathbb%7BE%7D_%7Bq(%5Calpha)%20%7D%20%5Clog%20%5Cfrac%7Bq(%5Calpha)%7D%7Bp(%5Calpha)%7D%5C%5C%0A%25%5Cnonumber%20%0A%25%5Cnonumber%0A%26%3D%26-%5Csum_i%20%5Ctextrm%7BELBO%7D_i-%5Cmathbb%7BE%7D_%7Bq(%5Calpha)%7D%5Clog%20%5Cfrac%7Bq(%5Calpha)%7D%7Bp(%5Calpha)%7D%0A" alt="
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
&amp;=&amp; -\sum_i \mathbb{E}_{q(z_i)} \log \frac{q(z_i)} {p(D_i|W_i)p(z_i)}- \mathbb{E}_{q(\alpha) } \log \frac{q(\alpha)}{p(\alpha)}\\&amp;=&amp;-\sum_i \textrm{ELBO}_i-\mathbb{E}_{q(\alpha)}\log \frac{q(\alpha)}{p(\alpha)}" />

The task-dependent part is <img align="center" src="https://i.upmath.me/svg/%0A-%5Csum_i%20%5Ctextrm%7BELBO%7D_i%0A" alt="-\sum_i \textrm{ELBO}_i" /> and the task-commom part is <img align="center" src="https://i.upmath.me/svg/%0A-%5Cmathbb%7BE%7D_%7Bq(%5Calpha)%7D%5Clog%20%5Cfrac%7Bq(%5Calpha)%7D%7Bp(%5Calpha)%7D%0A" alt="-\mathbb{E}_{q(\alpha)}\log \frac{q(\alpha)}{p(\alpha)}" />. The help information can be obtain by

    python main.py -h

Some toy data are given in *testdata* folder. The format of input files should be the same as these toy examples.
## Updated in 2022.09
* Reconstruct the code
* Discard edward and make the sampling process transparent
* Add the code for generating data
* Use Python 3.8 & torch
* Optimize the model architecture, hyperparameters and training process. The model has better performance than the previous version.
* Update scripts for making data/running experiments
* Optimize the memory usage

## Prepare data
Set genome.fa path in globalconfig.py

> bash script/preparejsonfile.sh # it might take a while

## Run experiments
Pretrain the Bayesian model:

> bash script/run_pretrain.sh;

Eval with the pretrain model:

> bash script/run_eval.sh

Five-fold cross-validation:

> bash script/run_cv.sh;

## requirement
Python==3.8

torch==1.11.0
