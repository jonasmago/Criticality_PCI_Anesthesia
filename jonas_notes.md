## Criticality/Complexity code applied to Jhana data

## Segment length choice
- `80 epochs` → this gives 72/80 good subjects ↔️ 90%
- `15 min = 900s` → this gives 73/80 good subjects ↔️ 91.25%

## Specific scripts, and waht still needs to be done: 
* `avc_std_dist`: for later
* `AVC`: 
    * ~ 20 min
    * -bin_treshold 2.0 -max_iei 0.008
* `DFA`: 
    * for all frequency bins
    * ran for 20 min segments
    * [ ] repeat for 15 min segments
* `EOC`: 
    * run with 'fixed`and `4Hz` cutoff
    * takes ˜ 2hr
* `EOS`: 
    * 1-45Hz
    * ran for 100 epochs 
    * [ ] repeat for 80 epochs 
* `Pred`: 
    * 1-45HZ
    * **takes super long!**
    * [ ] ran for 100 epochs, change to 80 epochs
* `Slope`: 
    * 1-45HZ
    * [ ] run for 80 epochs 



## What I am currently doing
* repair bad channels (I currently do not count how many I repair, possible exclusion criteria)
* filter data script specific (1-45Hz), or smaller bands

