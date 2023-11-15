#!/bin/bash

# Loop three times
for i in {1..5}; do
    # Execute the command and save output to a file
    ./csmca.py task1OneThreadperEntryWarped.cu > benchmark/benchmark_OneThreadPerEntry${i}.txt 2>&1
    ./csmca.py task1.cu > benchmark/benchmark_SharedMem${i}.txt 2>&1
    ./csmca.py task1Warped.cu > benchmark/benchmark_Warped${i}.txt 2>&1
    ./csmca.py dot-product.cu > benchmark/benchmark_Dot${i}.txt 2>&1

    # Check if the execution was successful
    if [ $? -eq 0 ]; then
        echo "Execution $i successful."
    else
        echo "Error in execution $i."
    fi
done
