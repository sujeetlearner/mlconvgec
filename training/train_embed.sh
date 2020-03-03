#!/bin/bash

set -x
set -e

source ../paths.sh
FAIRSEQPY=$SOFTWARE_DIR/fairseq-py
URL="https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hi.vec"
# download embeddings if necessary
EMBED_PATH=$MODEL_DIR/embeddings/wiki_model.vec
if [ ! -f $EMBED_PATH ]; then
    mkdir -p $MODEL_DIR/embeddings
    curl -L -o $MODEL_DIR/embeddings/wiki_model.vec $URL
fi

SEED=1000
DATA_BIN_DIR=processed/bin
OUT_DIR=models/mlconv_embed/model$SEED/
mkdir -p $OUT_DIR
#PYTHONPATH=$FAIRSEQPY:$PYTHONPATH CUDA_VISIBLE_DEVICES="0,1,2" python3 $FAIRSEQPY/train.py --save-dir $OUT_DIR --encoder-embed-dim 500 --encoder-embed-path $EMBED_PATH --decoder-embed-dim 500 --decoder-embed-path $EMBED_PATH --decoder-out-embed-dim 500 --dropout 0.2 --clip-norm 0.1 --lr 0.25 --min-lr 1e-4 --encoder-layers '[(1024,3)] * 7' --decoder-layers '[(1024,3)] * 7' --momentum 0.99 --max-epoch 100 --batch-size 32  --seed $SEED $DATA_BIN_DIR

#PYTHONPATH=$FAIRSEQPY:$PYTHONPATH CUDA_VISIBLE_DEVICES="0,1,2" python3 $FAIRSEQPY/train.py --save-dir $OUT_DIR --encoder-embed-dim 500 --encoder-embed-path $EMBED_PATH --decoder-embed-dim 500 --decoder-embed-path $EMBED_PATH --decoder-out-embed-dim 500 --dropout 0.2 --clip-norm 0.1 --lr 0.25 --min-lr 1e-4 --encoder-layers '[(1024,3)] * 7' --decoder-layers '[(1024,3)] * 7' --momentum 0.99 --max-epoch 100 --batch-size 32  --seed $SEED $DATA_BIN_DIR
PYTHONPATH=$FAIRSEQPY:$PYTHONPATH CUDA_VISIBLE_DEVICES="0" python3 $FAIRSEQPY/train.py --save-dir $OUT_DIR --encoder-embed-dim 300 --encoder-embed-path $EMBED_PATH --decoder-embed-dim 300 --decoder-embed-path $EMBED_PATH --decoder-out-embed-dim 300 --dropout 0.2 --clip-norm 0.1 --lr 0.25 --min-lr 1e-4 --encoder-layers '[(1024,3)] * 7' --decoder-layers '[(1024,3)] * 7' --momentum 0.99 --max-epoch 50 --batch-size 96  --seed $SEED --arch fconv $DATA_BIN_DIR
