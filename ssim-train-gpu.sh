#!/bin/bash
#SBATCH --gpus-per-node=1         # Number of GPU(s) per node
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=00-18:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=rrg-mcrowley 

module load cuda cudnn 
source tensorflow/bin/activate
python ./main_adp_samplew.py
