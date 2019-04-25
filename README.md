# MtBNN
The python script for MtBNN

# Train model
python main.py -r TRAINDATALIST -s MODELSAVEDIR 

# Test model
python main.py -t TESTFILELIST -p LOADPATH -k NUMBER_OF_TASKS

# Fine-tuning and validation on SNP data
python main.py -t TRAINDATALIST TESTDATA -n True -a INDEX_OF_TASK -k NUM_OF_TASK -p LOADPATH
