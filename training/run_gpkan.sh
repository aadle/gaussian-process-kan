#!/bin/bash

# MORE DATA SOLVES THE ISSUE????
# sqrt transform of entire y. 
# no scale back.
# CV sensitive to near 0 mean values.
# Less training data -> decrease LR

# Configuration
LR=0.01
# TEST_SIZE=0.02
TEST_SIZE=0.2
EPOCHS=400
FUNC="himmelblau"

uv run gpkan-training.py \
  --learning_rate=$LR \
  --test_size=$TEST_SIZE \
  --epochs=$EPOCHS \
  --function="$FUNC"
