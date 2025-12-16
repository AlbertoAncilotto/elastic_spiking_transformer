#!/bin/bash
# Find spikingjelly installation and fix np.bool deprecation

SPIKINGJELLY_PATH=$(python -c "import spikingjelly; import os; print(os.path.dirname(spikingjelly.__file__))")
FILE="$SPIKINGJELLY_PATH/datasets/cifar10_dvs.py"

if [ -f "$FILE" ]; then
    sed -i 's/np\.bool/bool/g' "$FILE"
    echo "Fixed np.bool in $FILE"
else
    echo "File not found: $FILE"
    exit 1
fi
