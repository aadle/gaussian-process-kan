#!/bin/bash

# 3 hour 20 minutes runtime...

# Configuration
LR=0.01
# TEST_SIZE=0.02
TEST_SIZE=0.2
EPOCHS=400
FUNC="trollveggen"

uv run gpkan-training.py \
  --learning_rate=$LR \
  --test_size=$TEST_SIZE \
  --epochs=$EPOCHS \
  --function="$FUNC" \
  --model_size 2 7 7 1 \
  --standardize_data
