#!/bin/bash

echo "START"
echo "Arguments: $@"

# Check if two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Użycie: $0 '<lista modeli>' '<lista wyników>'"
    echo "Przykład: $0 'model1,model2,model3' 'output1,output2,output3'"
    exit 1
fi

# Parse comma-separated arguments
IFS=',' read -r -a MODELS <<< "$1"
IFS=',' read -r -a OUTPUTS <<< "$2"

# Ensure the number of models and outputs matches
if [ "${#MODELS[@]}" -ne "${#OUTPUTS[@]}" ]; then
    echo "Błąd: Liczba modeli i folderów wyjściowych musi być taka sama."
    exit 2
fi

# Base paths
BASE_MODEL_PATH= # "YOUR PATH"
BASE_OUTPUT_PATH= # "YOUR PATH"
DATA_PATH= # "YOUR PATH"

# Iterate over models and outputs
printf "Iteracyjne trenowanie"
for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[$i]}
    OUTPUT=${OUTPUTS[$i]}

    MODEL_PATH="$BASE_MODEL_PATH/$MODEL"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$OUTPUT"

    python train_new.py \
        --model "$MODEL_PATH" \
        -p "$DATA_PATH" \
        -x high_signal \
        --seed 1234567890 \
        --num_epochs 200 \
        --learning_rate 0.001 \
        --momentum 0.1 \
        --output "$OUTPUT_PATH"

    if [ $? -ne 0 ]; then
        echo "ERROR with model $MODEL and output $OUTPUT."
        exit 1
    fi
done

echo "Success"
echo "Koniec skryptu. Naciśnij Enter, aby zamknąć..."
read -r
