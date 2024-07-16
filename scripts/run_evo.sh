#!/bin/bash

# Ensure the script takes the first argument as evoname
if [ -z "$1" ]; then
    echo "Usage: $0 <evoname>"
    exit 1
fi

evoname=$1
ckptdir=$CKPT_DIR
state_file="${ckptdir}/${evoname}/state.json"

while true; do
    # Check the sum of the budgets in the state.json file
    sum_budgets=$(python -c "
import json
import sys

state_file = sys.argv[1]
with open(state_file) as f:
    state = json.load(f)

print(sum(state['budgets'].values()))
" "$state_file")
    
    if [ "$sum_budgets" -eq "0" ]; then
        echo "All budgets are 0, stopping the loop."
        break
    fi

    python -m model_discovery.evolution --mode evolve --evoname $evoname
    python -m model_discovery.evolution --mode verify --evoname $evoname
done
