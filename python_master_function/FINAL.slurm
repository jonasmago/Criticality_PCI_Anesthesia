#!/bin/bash
#SBATCH --account=def-michael9     # Replace with your actual account
#SBATCH --job-name=slurm_test       # Job name
#SBATCH --time=00:10:00            # Max wall time (HH:MM:SS)
#SBATCH --cpus-per-task=32          # Number of CPUs
#SBATCH --mem=16G                  # Memory (adjust as needed)
#SBATCH --output=logs/final_%j.out # Standard output (%j = job ID)
#SBATCH --error=logs/final_%j.err  # Standard error
#SBATCH --mail-user=jonas.h.mago@gmail.com  # Optional: email notifications
#SBATCH --mail-type=ALL           # Email on (FIAL, ALL, BEGIN, END)

cd /home/jmago/projects/def-michael9/jmago/jhana_eeg/Criticality_PCI_Anesthesia/python_master_function

# Load your Python environment
module load python/3.10
source ../../../venv/bin/activate 

# Run the Python script
python FINAL.py -device c -name slurm_test
