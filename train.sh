#!/bin/bash
for model in dae vae cae; do
    echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] Starting $model..."
    python hparam_search.py --model $model --sweep sweeps/$model.yaml --n_trials 25
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $model.\n"
done
