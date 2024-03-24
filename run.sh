#!/bin/bash
# TRAIN_FLAG = $1
# CHECKPOINT_PATH = $2
# DATA_PATH = $3 # data/FB15K

if [ $# -eq 0 ]
then
  echo "Please specify test or train."
  exit 0
fi

if [ $1 == "train" ]
then
nohup python -u train.py \
  --data_path $3 \
  --structural_path $4 \
  --text_path $5 \
  --train_size 1 \
  --learning_rate 0.001 \
  --padding_size 40 \
  --embedding_dimensions 300 \
  --lstm_hidden_size 400 \
  --lstm_hidden_lstm_layerssize 2 \
  --dropout 0.2 \
  --learning_decay 0.35 \
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