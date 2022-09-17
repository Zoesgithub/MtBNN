cat ${1}| while read line;
do
echo ${line}
wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/${line}.gz
wget https://egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/narrowPeak/${line}.gz
wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeOpenChromDnase/${line}.gz
done