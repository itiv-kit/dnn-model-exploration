#!/bin/bash
#SBATCH --job-name=model_exploration
#SBATCH --output=exploration.out
#SBATCH --error=exploration.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2048mb
#SBATCH --partition=normal
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --time=08:00:00
#SBATCH --array=0-7

RUNPATH=$HOME/projekte/mixed-precision-dnns
cd $RUNPATH
source $HOME/venvs/torch_exploration/bin/activate
python model_explorer/scripts/explore.py workloads/resnet50_haicore.yaml --progress -v


