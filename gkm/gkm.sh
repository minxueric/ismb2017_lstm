#!/bin/bash

if [ -n "$1" ]
then
    echo "Dataset is $1"
fi

name=$1

# for i in {0..9}; do
#     ./gkmtrain -T 4 /home/xumin/deepenhancer/data/${name}_train_positive_${i}.fa /home/xumin/deepenhancer/data/${name}_train_negative_${i}.fa ${name}_lsgkm_${i}
#     ./gkmpredict /home/xumin/deepenhancer/data/${name}_test_negative_${i}.fa ${name}_lsgkm_${i}.model.txt ${name}_negative_lsgkm_test_${i}
#     ./gkmpredict /home/xumin/deepenhancer/data/${name}_test_positive_${i}.fa ${name}_lsgkm_${i}.model.txt ${name}_positive_lsgkm_test_${i}
# done

./gkmtrain -T 16 /home/xumin/bak_open_lstm/gkm/data/${name}_train_positive.fa /home/xumin/bak_open_lstm/gkm/data/${name}_train_negative.fa ${name}_lsgkm
./gkmpredict -T 16 /home/xumin/bak_open_lstm/gkm/data/${name}_test_negative.fa ${name}_lsgkm.model.txt ${name}_negative_lsgkm_test
./gkmpredict -T 16 /home/xumin/bak_open_lstm/gkm/data/${name}_test_positive.fa ${name}_lsgkm.model.txt ${name}_positive_lsgkm_test

