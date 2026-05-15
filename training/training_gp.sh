#!/bin/bash

LR=0.001
TEST_SIZE=0.2
EPOCHS=500
FUNCTIONS=("himmelblau" "goldstein" "trig" "trollveggen" "grandcanyon")
KERNELS=("rbf" "matern32" "matern52")

for FUNC in "${FUNCTIONS[@]}"; do
  for KERNEL in "${KERNELS[@]}"; do
    echo "Running training for function: $FUNC, kernel: $KERNEL"

    uv run gp_training.py \
      --learning_rate=$LR \
      --test_size=$TEST_SIZE \
      --epochs=$EPOCHS \
      --function="$FUNC" \
      --kernel="$KERNEL"

    echo "Completed: $FUNC / $KERNEL"
    echo "---"
  done
done

echo "All training runs completed!"
