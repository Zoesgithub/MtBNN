# MtBNN
The python script for MtBNN
##requirement
Python 2.7.14
tensorflow-gpu==1.4.0
edward==1.3.5
numpy==1.14.5
scipy==1.1.0
scikit-learn==0.20.0
keras==2.1.3
## Train model
python main.py -r TRAINDATALIST -s MODELSAVEDIR 

## Test model
python main.py -t TESTFILELIST -p LOADPATH -k NUMBER_OF_TASKS

## Fine-tuning and validation on SNP data
python main.py -t TRAINDATALIST TESTDATA -n True -a INDEX_OF_TASK -k NUM_OF_TASK -p LOADPATH
