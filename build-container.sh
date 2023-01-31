set -e

sudo docker build -f Dockerfile . -t metadata:tsv

sudo docker run -v /home/jose_almeida/projects/metadata-sequence-classification/metadata_tmp:/data:ro metadata:tsv /data/metadata-from-stelios.tsv

sudo docker tag \
    metadata:tsv \
    pcr.procancer-i.eu/metadata-classification/metadata-classification:tsv

sudo docker login pcr.procancer-i.eu

sudo docker push \
    pcr.procancer-i.eu/metadata-classification/metadata-classification:tsv