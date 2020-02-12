#!/bin/bash


set -e
set -x
source ../paths.sh

## paths to training and development datasets\
train_data_prefix=$DATA_DIR/train
dev_data_prefix=$DATA_DIR/dev
dev_data_m2=$DATA_DIR/dev.all.m2

# path to subword nmt
SUBWORD_NMT=$SOFTWARE_DIR/subword-nmt
# path to Fairseq-Py
FAIRSEQPY=$SOFTWARE_DIR/fairseq-py

######################
# subword segmentation
mkdir -p models/bpe_model
bpe_operations=30000



cat $train_data_prefix.tok.trg | $SUBWORD_NMT/learn_bpe.py -s $bpe_operations > models/bpe_model/train.bpe.model
mkdir -p processed/
$SCRIPTS_DIR/apply_bpe.py -c models/bpe_model/train.bpe.model < $train_data_prefix.tok.src > processed/train.all.src
$SCRIPTS_DIR/apply_bpe.py -c models/bpe_model/train.bpe.model < $train_data_prefix.tok.trg > processed/train.all.trg
$SCRIPTS_DIR/apply_bpe.py -c models/bpe_model/train.bpe.model < $dev_data_prefix.tok.src > processed/dev.src
$SCRIPTS_DIR/apply_bpe.py -c models/bpe_model/train.bpe.model < $dev_data_prefix.tok.trg > processed/dev.trg
#cp $dev_data_m2 processed/dev.m2
#cp $dev_data_prefix.all.tok.src processed/dev.input.txt

##########################
#  getting annotated sentence pairs only
python3 $SCRIPTS_DIR/get_diff.py  processed/train.all src trg > processed/train.annotated.src-trg
cut -f1  processed/train.annotated.src-trg > processed/train.src
cut -f2  processed/train.annotated.src-trg > processed/train.trg


#########################
# preprocessing
python3 $FAIRSEQPY/preprocess.py --source-lang src --target-lang trg --trainpref processed/train --validpref processed/dev  --nwordssrc 30000 --nwordstgt 30000 --destdir processed/bin

