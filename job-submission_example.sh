#!/bin/bash
#SBATCH --job-name=model_exploration
#SBATCH --output=results/exploration_%A_%a.out
#SBATCH --error=results/exploration_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2048mb
#SBATCH --partition=normal
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --array=0-2

RUNPATH=$HOME/projekte/mixed-precision-dnns
cd $RUNPATH
source $HOME/venvs/torch_exploration/bin/activate
python model_explorer/scripts/evaluate_individual.py workloads/resnet50.yaml dummy --progress -n 20


