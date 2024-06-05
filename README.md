# Critical_PCI
This repository contains all code used for the publication: 
"Critical dynamics in spontaneous EEG predict anesthetic-induced loss of consciousness and perturbational complexity"

The code was written by
Charlotte Maschke and Jordan O'Byrne



### requirements
`pip install -r requirements.txt`
also dependent on EdgeofPy toolbox (https://github.com/jnobyrne/edgeofpy)

## Available Features
the input parameters of all functions can be seen using the comand `python functionNAME.py --help`

Most functions output a .txt with all features over subjects and conditions. Some scripts additionally output space-resolved features per subject.

- #### Avalanche Criticality features
- `features_AVC.py`
  - alpha, tau, third
  - deviation from criticality coefficient
  - branching ratio
  - repertoire
  - intrinsic timescale (not validated and not used in the paper)
  - susceptibility (not validated and not used in the paper)
- `features_avc_std_dist.py`
  - distribution of Avalances over standard deviation
  - amount of Avalanches

- #### Edge of Chaos features
- `features_EOC.py`
  - 01 Chaos test (different options, see --help)

- #### Edge of Synchrony features
- `features_EOS.py`
  - Pair Correlation Function (different options, see --help)
  - Order parameter

- #### Criticality-related features
- `features_Comp.py`
  - Lempel-Ziv complexity univariate, shuffle and length normalized
  - Fractal dimension

- `features_DFA.py`
  - Hurst exponent

- `features_slope.py`
  - Spectral Slope

- #### Other features
- `features_power.py`
  - spectral power in frequency bands


## METHODS
Herein used utility functions `METHODS_chaos` and `METHODS_EOS` are now part of functions of EdgeofPy.

