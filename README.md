# CRYO2ICE radar/laser freeboards, snow depth on sea ice and comparison against auxiliary data during winter seasons 2020-2022 v3.0
 A first examination of CRYO2ICE (CryoSat-2 and ICESat-2) freeboards (radar or laser), derived snow depths, and comparison auxiliary data (daily snow depth maps from passive microwave or reanalysis-based models, and buoys) during the winter season (November-April) of 2020-2022. 
 This program computes CRYO2ICE comparable observations at CryoSat-2 locations (used as baseline) by a defined search radius using the European Space Agency's (ESA's) Baseline-E  L2 products for CryoSat-2 and ATL10 r005 from NASA's ICESat-2. These are later compared with freeboard products from ESA's CLimate Change Initiative (CCI+) and the lognormal altimetric re-tracker model (LARM, Landy et al. 2021), to investigate impact of different re-trackers, as well as daily snow depth maps from passive microwave (AMSR2/ASMR-E) and reanalysis-based snow models (SnowModel-Lagrangian/SMLG).

[![DOI](https://badgen.net/badge/DOI/10.1158%2FDTU.21369129/red)](https://data.dtu.dk/articles/dataset/CRYO2ICE_radar_laser_freeboards_snow_depth_on_sea_ice_and_comparison_against_auxiliary_data_during_winter_season_2020-2021/21369129)

Contact: Renée Mie Fredensborg Hansen @ rmfha@space.dtu.dk

### Versions

v1.0 - October 2022: Initial repository used to process CryoSat-2 and ICESat-2 (CRYO2ICE) observations using Baseline-D as baseline for the entire processing chain. Comparisons against auxiliary data products (AMSR2 and SM-LG), and buoy estimates.

v2.0 - February 2024: Updates based on reviews to preprint and addition of code related to analysis of sea ice drift and cross-over analysis (XO).  

v3.0 - March 2024: Final version of Jupyter Notebooks and additional python scripts used in Fredensborg Hansen et al. (2024), acceptance of paper (Fredensborg Hansen et al., 2024). 

### Data 
- ESA Baseline-E L2 CryoSat-2 Ice products. Available at identified CRYO2ICE locations for this study at DOI: 10.11583/DTU.21369129. Specific tracks identified using www.cs2eo.org tool, notebook is provided to download the exact files if wanted.  Download script provided by the same tool.
- NASA ATL10 Sea Ice Freeboard products. Available at identified CRYO2ICE locations for this study at DOI: 10.11583/DTU.21369129.  Specific tracks identified using www.cs2eo.org tool, notebook is provided to download the exact files if wanted.   Download script provided by the same tool.
- AMSR2 passive microwave sea ice concentration and snow depth: available at National Snow and Ice Data Center (NSIDC) as AMSR-E/AMSR2 Unified L3 Daily 12.5 km Brightness Temperatures, Sea Ice Concentration, Motion & Snow Depth Polar Grids, Version 1 (AU\SI12) by Meier et al. (2018). Available at identified CRYO2ICE locations for this study at DOI: 10.11583/DTU.21369129. 
- SM-LG: available at NSIDC as Lagrangian Snow Distributions for Sea-Ice Applications, Version 1 (NSIDC-0758) by Listion et al. (2021). Available at identified CRYO2ICE locations for this study at DOI: 10.11583/DTU.21369129. 
- CCI CryoSat-2 freeboard product: available from XXX, presented in Rinne and Hendricks (2023). Available at identified CRYO2ICE locations for this study at DOI: 10.11583/DTU.21369129. 
- LARM CryoSat-2 freeboard product: presented in Landy et al. (2020), provided by Jack Landy for this study. Available at identified CRYO2ICE locations for this study at DOI: 10.11583/DTU.21369129. 

### Installation and about the code 
The code is written in Jupyter Notebook or in python, and requires various python packages which are denoted in the CRYO2ICE_environment.yml file. These packages can be installed and activated when creating a conda environment using the .yml file. 

## Open research
This github is compiled of several Jupyter Notebooks used to compute the data and produce the study presented in: Fredensborg Hansen et al. (2024). Data compiled and used for the study is presented in: DOI 10.11583/DTU.21369129 

## References
Fredensborg Hansen, R.M, Henriette Skourup, Eero Rinne, et al. (2024). (PROVIDED SOON). 

Landy, J. C., Petty, A. A., Tsamados, M., & Stroeve, J. C. (2020). Sea ice roughness overlooked as a key source of uncertainty in CryoSat-2 ice freeboard retrievals. Journal of Geophysical Research: Oceans, 125, e2019JC015820. https://doi.org/10.1029/2019JC015820

Liston, G. E., J. Stroeve, and P. Itkin. (2021). Lagrangian Snow Distributions for Sea-Ice Applications, Version 1 [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/27A0P5M6LZBI. Date Accessed 10-24-2022.

Meier, W. N., T. Markus, and J. C. Comiso. 2018. AMSR-E/AMSR2 Unified L3 Daily 12.5 km Brightness Temperatures, Sea Ice Concentration, Motion & Snow Depth Polar Grids, Version 1. [Indicate subset used]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. doi: https://doi.org/10.5067/RA1MIJOYPK3P. [12 October 2022].

Rinne, E., & Hendricks, S. (2023). CCI+ Sea Ice ECV Sea Ice Thickness Product User Guide (PUG) . (Reference D4.2, Phase 1).
