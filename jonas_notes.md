### Data I need to create: 

[ ] participant info file 
    [ ] ID
    [ ] Cond
    [ ] Drug
[ ] data (currently in .mat format, change to .fif)


## Figure out the best hyper parameter space here
* AVC: 
    * FIL_FREQ = (1, 40) # bandpass frequencies
    * THRESH_TYPE = 'both' # Fosque22: 'both' —> ???
    * GAMMA_EXPONENT_RANGE = (0, 2)
    * LATTICE_SEARCH_STEP = 0.1
    * BIN_THRESHOLD = float(2)
    * MAX_IEI = float(0.008)
    * BRANCHING_RATIO_TIME_BIN = float(1)
* DFA
    * lfreq = 0.1  # Low frequency filter
    * hfreq = 40   # High frequency filter
    * maximum of 200s 
* EOC
    * k_type = 'flex'
* EOS
    * minfreq = 0.1
    * maxfreq = 40
* Pred
    * lfreq = 0.1
    * hfreq = 40
* Slope (did you do this for different frequency bins?)
    * lfreq = 1
    * hfreq = 40
