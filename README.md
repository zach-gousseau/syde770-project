# syde770-project
## Sea Ice Concentration Forecasting Over the Kivalliq Polynya Using Convolutional LSTM Networks

Repository containing work completed for the SYDE 770 term project.

### Abstract
Despite their importance in the global climate system, the predictability of coastal polynyas has received relatively little scientific attention. In this paper, we explore the use of convolutional long short-term memory (ConvLSTM) networks to model the behavior of the Kivalliq polynya along the Northwest coast of the Hudson Bay. We train 60 neural network models with varying depths, number of parameters and prediction modes to forecast sea ice concentration (SIC) over the polynya, and compare our model performance to climatology and persistence as naive baselines. We find that at the daily timescale, our model out-performs baselines to forecast SIC with a 1-day and 7-day lead time, achieving a pixel-wise RMSE of 2.47 and 4.27, respectively, while failing to show skill in forecasting 14 days ahead. At the weekly timescale, our model shows skill in forecasting SIC for the next week, one week out and two weeks out with an RMSE of 2.75, 3.16 and 3.36, respectively. A sensitivity analysis showed that the models rely heavily on the SIC timeseries inputs rather than any exogenous variables such as wind, temperature and heat fluxes which are known to influence the formation of polynyas, indicating that the models have room for improvement.

## Project Navigation
```
.
├── data
│   ├── aoi  <-- Shapefile bounding the Hudson Bay 
│   ├── download_ceda_sic.ipynb  <-- Download ESA SIC from CEDA
│   ├── download_era5.py  <-- Download ERA5 data from CDS
│   ├── era5.ipynb        <-- Download ERA5 data from CDS 
│   └── stack.ipynb  <-- Reproject/regrid ERA5 and stack with ESA SIC
├── model
│   ├── model_tester.py  <-- Contains main class which defines the model
│   └── run_tests.ipynb  <-- For training models given a set of parameters
└── results
    ├── eval.ipynb  <-- Read results from run_tests.ipynb & create plots, sensitivity analysis, etc
    ... (model results contained in subfolders here)
```
