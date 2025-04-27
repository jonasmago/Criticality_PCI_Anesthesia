#!/bin/bash
#SBATCH --account=def-michael9     # Replace with your actual account
#SBATCH --job-name=big_all_03          # Job name
#SBATCH --time=40:00:00            # Max wall time (HH:MM:SS)
#SBATCH --cpus-per-task=40         # Number of CPUs
#SBATCH --mem=32G                 # Memory (up to 192 per 40 CPUs)
#SBATCH --output=logs/final_%j.out # Standard output (%j = job ID)
#SBATCH --error=logs/final_%j.err  # Standard error
#SBATCH --mail-user=jonas.h.mago@gmail.com  # Optional: email notifications
#SBATCH --mail-type=ALL           # Email on (FIAL, ALL, BEGIN, END)

# #SBATCH --nice=1000

cd /home/jmago/projects/def-michael9/jmago/jhana_eeg/Criticality_PCI_Anesthesia/python_master_function

# Load your Python environment
module load python/3.10
source ../../../venv/bin/activate 

# Run the Python script
python FINAL_all.py -device c -name big_all_03
