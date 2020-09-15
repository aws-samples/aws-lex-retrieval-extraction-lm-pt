# retrieval extraction pre-training with BERT/DistilBERT model architecture

export SQUAD_TRAIN=squad2.0/train-v2.0.json
export SQUAD_DEV=squad2.0/dev-v2.0.json
export WIKI_TRAIN=/data/ELM/preproc/doc_split/processed/data_resplit_0.json
export WIKI_DEV=/data/ELM/preproc/doc_split/processed/test_s10_1k.json
export EXP_NAME=$1
export LR_WIKI=5e-5
export LR_SQUAD=3e-5
export BS_WIKI=32
export BS_SQUAD=2
export WD=0.0
export WARMUP=0
export CHECKPOINT=checkpoints/elm_sent/checkpoint-120000/

rm -r runs/$EXP_NAME
rm -r checkpoints/$EXP_NAME

python run_rex.py \
  --model_type bert \
  --model_name_or_path distilbert-base-cased \
  --tokenizer_name distilbert-base-cased \
  --do_train \
  --do_lower_case \
  --version_2_with_negative \
  --train_file $SQUAD_TRAIN \
  --predict_file $SQUAD_DEV \
  --per_gpu_train_batch_size $BS_WIKI \
  --per_gpu_eval_batch_size 4 \
  --learning_rate $LR_WIKI \
  --num_train_epochs 8.0 \
  --max_query_length 64 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --threads 20 \
  --exp_name $EXP_NAME \
  --data_dir /mnt/efs/Wikipedia
  --output_dir checkpoints/$EXP_NAME/ \
  --logging_steps 200 \
  --save_steps 10000 \
  --warmup_steps $WARMUP \
  --weight_decay $WD
