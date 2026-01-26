#!/bin/bash


# Configuration
LR=0.01
# TEST_SIZE=0.02
TEST_SIZE=0.2
EPOCHS=400
FUNC="goldstein"

uv run gpkan-training.py \
  --learning_rate=$LR \
  --test_size=$TEST_SIZE \
  --epochs=$EPOCHS \
  --function="$FUNC" \
  --model_size 2 5 5 1
