python -m Tools.narrow2json script/GM12878_files /data2/xcc/code/MtBNN/data/GM12878 gm12878_dsqtl.csv,eqtl_gm12878.csv
python -m Tools.narrow2json script/HepG2_files /data2/xcc/code/MtBNN/data/HepG2 eqtl_hepg2.csv
python -m Tools.narrow2json script/TCell_files /data2/xcc/code/MtBNN/data/TCell atacqtl_tcell.txt
python -m Tools.mut2json data/GM12878/eqtl_gm12878.csv
python -m Tools.mut2json data/GM12878/gm12878_dsqtl.csv
python -m Tools.mut2json data/HepG2/eqtl_hepg2.csv
python -m Tools.mut2json data/TCell/atacqtl_tcell.txt