## GM12878
python main.py -c tasks/GM12878/cv_eqtl_generic
python main.py -c tasks/GM12878/cv_eqtl_all

python main.py -c tasks/GM12878/cv_dsqtl_generic
python main.py -c tasks/GM12878/cv_dsqtl_all
python main.py -c tasks/GM12878/cv_dsqtl

## hepg2
python main.py -c tasks/HepG2/cv_eqtl_generic_dnn
python main.py -c tasks/HepG2/cv_eqtl_all_dnn

## tcell
python main.py -c tasks/TCell/cv_atacqtl_generic_dnn
python main.py -c tasks/TCell/cv_atacqtl_all_dnn

### ft for dnn #####
python main.py -c tasks/GM12878/cv_eqtl_generic_dnn
python main.py -c tasks/GM12878/cv_eqtl_all_dnn

python main.py -c tasks/GM12878/cv_dsqtl_generic_dnn
python main.py -c tasks/GM12878/cv_dsqtl_all_dnn
python main.py -c tasks/GM12878/cv_dsqtl_dnn

## hepg2
python main.py -c tasks/HepG2/cv_eqtl_generic
python main.py -c tasks/HepG2/cv_eqtl_all

## tcell
python main.py -c tasks/TCell/cv_atacqtl_generic
python main.py -c tasks/TCell/cv_atacqtl_all



