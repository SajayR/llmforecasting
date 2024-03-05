# LLMs for Weather Forecasting

Submission for Prof. Nipun Batra's internship application.
## Task 1: Data Collection
### Overview

For this project, air quality data was gathered from the CAAQM data repository, focusing on the years 2022 and 2023. The selected IGI Airport weather station in Delhi provides hourly measurements, striking an optimal balance between granularity and manageability of data for our forecasting tasks.

### Data Parameters

The raw dataset holds a wide array of air quality indicators:

    Timestamp: Marking the date and hour of each observation.
    Particulate Matter: PM2.5 and PM10 (µg/m³), critical for assessing air pollution levels.
    Gases: NO, NO2, and NOx (µg/m³ for NO and NO2, ppb for NOx), representing major pollutants.
    Temperature (AT), Relative Humidity (RH), Wind Speed (WS), Wind Direction (WD), and other environmental factors were also recorded but are beyond the scope of this initial analysis.

|Timestamp          |PM2.5 (µg/m³)|PM10 (µg/m³)|NO (µg/m³)|NO2 (µg/m³)|NOx (ppb)|NH3 (µg/m³)|SO2 (µg/m³)|CO (mg/m³)|Ozone (µg/m³)|Benzene (µg/m³)|Toluene (µg/m³)|Xylene (µg/m³)|O Xylene (µg/m³)|Eth-Benzene (µg/m³)|MP-Xylene (µg/m³)|AT (°C)|RH (%)|WS (m/s)|WD (deg)|RF (mm)|TOT-RF (mm)|SR (W/mt2)|BP (mmHg)|VWS (m/s)|
|-------------------|-------------|------------|----------|-----------|---------|-----------|-----------|----------|-------------|---------------|---------------|--------------|----------------|-------------------|-----------------|-------|------|--------|--------|-------|-----------|----------|---------|---------|
|2022-01-01 00:00:00|273.55       |386.22      |249.03    |54.67      |303.61   |NA         |NA         |5.59      |8.22         |NA             |NA             |NA            |NA              |NA                 |NA               |NA     |NA    |NA      |NA      |NA     |0.00       |NA        |NA       |NA       |
|2022-01-01 01:00:00|268.87       |432.99      |294.77    |50.75      |345.51   |NA         |NA         |5.74      |8.14         |NA             |NA             |NA            |NA              |NA                 |NA               |NA     |NA    |NA      |NA      |NA     |0.00       |NA        |NA       |NA       |
|2022-01-01 02:00:00|258.02       |396.28      |247.48    |50.37      |297.77   |NA         |NA         |5.27      |8.12         |NA             |NA             |NA            |NA              |NA                 |NA               |NA     |NA    |NA      |NA      |NA     |0.00       |NA        |NA       |NA       |
|2022-01-01 03:00:00|194.91       |297.69      |152.66    |45.18      |198.09   |NA         |NA         |4.01      |8.03         |NA             |NA             |NA            |NA              |NA                 |NA               |NA     |NA    |NA      |NA      |NA     |0.00       |NA        |NA       |NA       |
|2022-01-01 04:00:00|198.00       |314.81      |123.04    |41.78      |164.95   |NA         |NA         |3.55      |7.90         |NA             |NA             |NA            |NA              |NA                 |NA               |NA     |NA    |NA      |NA      |NA     |0.00       |NA        |NA       |NA       |


### Data Simplification

To streamline the analysis, the dataset was truncated to focus on the first six parameters: Timestamp, PM2.5, PM10, NO, NO2, and NOx. Trimming down the number of parameters made it easier to get the data ready for analysis. This way, we could dive straight into modeling and checking the results without having to spend time in redundant processes, and also gets rid of any columns with missing values (NA)

|Timestamp          |PM2.5 (µg/m³)|PM10 (µg/m³)|NO (µg/m³)|NO2 (µg/m³)|NOx (ppb)|
|-------------------|-------------|------------|----------|-----------|---------|
|2022-01-01 00:00:00|273.55       |386.22      |249.03    |54.67      |303.61   |
|2022-01-01 01:00:00|268.87       |432.99      |294.77    |50.75      |345.51   |
|2022-01-01 02:00:00|258.02       |396.28      |247.48    |50.37      |297.77   |
|2022-01-01 03:00:00|194.91       |297.69      |152.66    |45.18      |198.09   |
|2022-01-01 04:00:00|198.00       |314.81      |123.04    |41.78      |164.95   |

### Data Cleaning
Owing to unknown variables at the weather stations, there are multiple missing values particular time periods. To mitigate this issue and maintain the integrity of our time-series analysis, we employed linear interpolation for filling these gaps. This method ensures a coherent dataset by estimating missing values based on the linear relationships observed in the surrounding data points, 

## Task 2: Setup LLM model locally and showcase zero-shot performance

Model Selection: In selecting models for this task, our priority was to focus on pure predictive capabilities of pre-trained LLMs without the additional layers tailored for chat or instruction following, which could potentially detract from performance in a forecasting context. 
The primary models chosen were 7 billion parameter versions of 
   * Llama2
   * Gemma
   * Falcon

The consistent limitation of 7B parameters for all the models was due to local memory limitations.
A point to be noted is that we expect better results from larger models, as some students working on similiar projects have noted on online forums.





## Task 3: LSTM Baseline

There are multiple different LSTM models we built, with the point being that we wanted the best representation from LSTM's for each scenario. 
Two LSTM models were trained for two different context lengths; a week and 3 days. For both, we split the datapoints into sets of 8 days each, where the 7 days acted as context and 8th day acted as the target output, and careful consideration was taken to make sure the sets did not overlap to prevent data leakage.
The following metrics were obtained for comparision with LLM's

* Mean Absolute Error (On scaled-down data): **0.0570**
* Mean Squared Error (On scaled-down data): **0.0076**
* Root Mean Squared Error (On scaled-down data): **0.0874**
* Mean Absolute Error (On original scale): **33.3631**
* Mean Squared Error (On original scale): **3018.4822**
* Root Mean Squared Error (On original scale): **54.9407**
* Mean Absolute Percentage Error: **78.25%**


#### Sample Predictions

Graph with full-context length plotted

![PM10_week](https://github.com/SajayR/llmforecasting/assets/62949586/3b22680f-ba2f-4811-9c13-75277d770f9d)

Zoomed in Graphs for better visualisation 

![NOx_day](https://github.com/SajayR/llmforecasting/assets/62949586/9a752474-8f27-4099-aae3-d05aebcf6e08)

![pm10_day](https://github.com/SajayR/llmforecasting/assets/62949586/4287448b-a65c-4b71-bfdd-8497cc9418d6)


![pm25_day](https://github.com/SajayR/llmforecasting/assets/62949586/10570dda-3b5e-4e8b-9f5a-d9f7c48eca0b)


![NO2_day](https://github.com/SajayR/llmforecasting/assets/62949586/85e1e1c2-3edc-4488-91d5-ec2f881f46a5)

![NO_day](https://github.com/SajayR/llmforecasting/assets/62949586/857bc432-f418-4607-b8c9-04632d753cd2)


## Task 4: Showcase performance of OS LLM made for time-series forecasting


