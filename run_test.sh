python main.py -r testdata/train -s ./model
python main.py -t testdata/test -p ./model/model.ckpt-0 -k 22
python main.py -t snp_train snp_test -n True -a 18 -k 22 -p ./model/model.ckpt-0
