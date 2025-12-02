
#!/bin/bash

wget https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip -P ../data

unzip -q ./data/vangogh2photo.zip -d ../data

echo "Dataset downloaded and extracted to 'data/vangogh2photo'."

rm ../data/vangogh2photo.zip