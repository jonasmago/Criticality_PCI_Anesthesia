#!/bin/bash
#SBATCH --account=def-michael9
#SBATCH --job-name=big_all
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G                 # Memory (up to 192 per 40 CPUs)
#SBATCH --output=logs/final_%j.out
#SBATCH --error=logs/final_%j.err
#SBATCH --mail-user=jonas.h.mago@gmail.com  # Optional: email notifications
#SBATCH --mail-type=ALL           # Email on (FIAL, ALL, BEGIN, END)

# #SBATCH --nice=1000

cd /home/jmago/projects/def-michael9/jmago/jhana_eeg/Criticality_PCI_Anesthesia/python_master_function

# Load your Python environment
module load python/3.10
source ../../../venv/bin/activate 

# Run the Python script
python FINAL_all.py \
  -device c \
  --name may_pred \
  --MAX_TRIALS 200 \
  --MAX_S 2000 \
  --RUN_PRED
  

# python FINAL_all.py \
#   -device c \
#   --name may \
#   --MAX_TRIALS 1 \
#   --MAX_S 2000 \
#   --start 0 \
#   --end 100 \
#   --RUN_EOC \
#   --RUN_PRED \
#   --RUN_SLOPE \
#   --RUN_SLOPE_PSD \
#   --RUN_DFA \
#   --RUN_AVC \
#   --RUN_STD_DIST \
#   --RUN_ANTROPY \
#   --RUN_BANDPOWER
