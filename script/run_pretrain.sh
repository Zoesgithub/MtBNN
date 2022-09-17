
rm -r tasks/GM12878/bnn_pretrain_scale3
python main.py -c tasks/GM12878/pretrain_bnn_scale3
rm -r tasks/HepG2/bnn_pretrain_scale3
python main.py -c tasks/HepG2/pretrain_bnn_scale3
rm -r tasks/TCell/bnn_pretrain_scale3
python main.py -c tasks/TCell/pretrain_bnn_scale3

rm -r tasks/GM12878/dnn_pretrain
python main.py -c tasks/GM12878/pretrain_dnn
rm -r tasks/HepG2/dnn_pretrain
python main.py -c tasks/HepG2/pretrain_dnn
rm -r tasks/TCell/dnn_pretrain_scale3
python main.py -c tasks/TCell/pretrain_dnn