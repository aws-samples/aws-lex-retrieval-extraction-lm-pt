# Finetuning pretrained models on Squad 2

export SQUAD_TRAIN=squad2.0/train-v2.0.json
export SQUAD_DEV=squad2.0/dev-v2.0.json
export EXP_NAME=$1
export LR_WIKI=5e-5
export LR_SQUAD=3e-5
export BS_WIKI=12
export BS_SQUAD=4
export WD=0.0
export WARMUP=0
export CHECKPOINT=checkpoints/rex-2data/checkpoint-170000/

rm -r runs/$EXP_NAME
rm -r checkpoints/$EXP_NAME

python run_squad.py \
  --model_type bert \
  --model_name_or_path $CHECKPOINT \
  --tokenizer_name distilbert-base-cased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --version_2_with_negative \
  --train_file $SQUAD_TRAIN \
  --predict_file $SQUAD_DEV \
  --per_gpu_train_batch_size $BS_SQUAD \
  --per_gpu_eval_batch_size 4 \
  --learning_rate $LR_SQUAD \
  --num_train_epochs 4.0 \
  --max_query_length 64 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --threads 32 \
  --exp_name $EXP_NAME \
  --output_dir checkpoints/$EXP_NAME/ \
  --save_steps 10000 \
  --warmup_steps $WARMUP \
  --weight_decay $WD
