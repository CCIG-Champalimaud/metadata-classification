set -e

tag=$1

sudo docker build -f Dockerfile . -t metadata:$tag

echo Testing with test_jose.tsv
sudo docker run -v /home/jose_almeida/projects/metadata-sequence-classification/metadata_tmp:/data:ro metadata:$tag /data/test_jose.tsv > /dev/null
echo Testing with ecrfs-series-20230309.parquet
sudo docker run -v /home/jose_almeida/projects/metadata-sequence-classification/metadata_tmp:/data:ro metadata:$tag /data/ecrfs-series-20230309.parquet > /dev/null

sudo docker tag \
    metadata:$tag \
    pcr.procancer-i.eu/metadata-classification/metadata-classification:$tag

sudo docker login pcr.procancer-i.eu

sudo docker push \
    pcr.procancer-i.eu/metadata-classification/metadata-classification:$tag