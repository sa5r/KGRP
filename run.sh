#!/bin/bash
# TRAIN_FLAG = $1
# CHECKPOINT_PATH = $2
# DATA_PATH = $3 # data/webnlg

if [ $# -eq 0 ]
then
  echo "Please specify test or train."
  exit 0
fi

if [ $1 == "train" ]
then
nohup python -u train.py \
  --data_path $3/train.json \
  --validation_path $3/valid.json \
  --relations_path $3/rel2id.json \
  --train_size 1 \
  --learning_rate 0.0015 \
  --padding_size 100 \
  --glove_path glove.6B.300d.txt \
  --embedding_dimensions 300 \
  --lstm_units 500 \
  --pool_size 80 \
  --strides 2 \
  --dropout 0.15 \
  --learning_decay 3e-5 \
  --batch_size 32 \
  --epochs 50 \
  --patience 5 \
  --checkpoint_path $2
elif [ $1 == "test" ]
then
nohup python -u evaluate.py \
  --data_path $3/test.json \
  --relations_path $3/rel2id.json \
  --padding_size 100 \
  --glove_path glove.6B.300d.txt \
  --embedding_dimensions 300 \
  --lstm_units 500 \
  --pool_size 80 \
  --strides 2 \
  --dropout 0.15 \
  --batch_size 32 \
  --checkpoint_path $2
else
  echo 'Please specify test or train.'
fi