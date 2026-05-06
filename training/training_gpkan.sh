#!/bin/bash

LR=0.01
TEST_SIZE=0.2
EPOCHS=300
FUNCTIONS=("himmelblau" "goldstein" "trig" "trollveggen" "grandcanyon")

# Loop over each function
# --model_size 2 5 5 1
for FUNC in "${FUNCTIONS[@]}"; do
  echo "Running training for function: $FUNC"
  
  uv run gpkan-training-big.py \
    --learning_rate=$LR \
    --test_size=$TEST_SIZE \
    --epochs=$EPOCHS \
    --function="$FUNC" \
    --model_size 2 5 5 1
  
  echo "Completed training for function: $FUNC"
  echo "---"
done

echo "All training runs completed!"
