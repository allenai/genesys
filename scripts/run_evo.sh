#!/bin/bash

while true; do
    python -m model_discovery.evolution --mode evolve
    python -m model_discovery.evolution --mode verify
done
