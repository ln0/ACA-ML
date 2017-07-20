./fasttext supervised -input "$1" -output model_amzn -lr 0.1 -wordNgrams 2 -bucket 3000000 -thread 10 -dim 20 -loss hs

./fasttext test model_amzn.bin "$2"