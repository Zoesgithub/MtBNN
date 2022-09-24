cd data
mkdir CaseStudy
cd CaseStudy
mkdir GTEx
cd GTEx
wget https://storage.googleapis.com/gtex_analysis_v7/single_tissue_eqtl_data/GTEx_Analysis_v7_eQTL.tar.gz
tar -xzvf  GTEx_Analysis_v7_eQTL.tar.gz
gzip -d GTEx_Analysis_v7_eQTL/*
