python main.py -r testdata/train0 testdata/train1 testdata/train2 testdata/train3 -s ./Model -b 200
python main.py -t testdata/test0 testdata/test1 testdata/test2 testdata/test3 -p ./Model/model.ckpt-64999 -k 4
python main.py -t snp_train snp_test -n True -a 1 -k 4 -p ./Model/model.ckpt-64999
