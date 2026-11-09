# Bitcoin Volatility Forecasting — AI Use Log

**Name:** Fernando Villegas  
**Course:** MATH546  
**Assignment:** Capstone Final Project  
**Date:** 2026-11-09

## 1. Platforms and Models

- ChatGPT (GPT-6, web interface).

## 2. Agents and Tooling

- ChatGPT was used through the web interface for inspection of the instructor data packet, conversion and alignment of Blockchain.com on-chain data, explanation of time-series diagnostics, selection and evaluation of ARCH/GARCH models, Python code for Random Forest and XGBoost, chronology-aware cross-validation, on-chain feature ablations, forecast visualization, and Diebold–Mariano-style comparisons.
- I supplied questions, modeling decisions, tables of results, and Jupyter notebooks; ChatGPT provided code and statistical interpretation. The notebooks show modeling code and outputs executed during the project. ChatGPT did not run the modeling experiment inside my local Python/Jupyter environment.
- ChatGPT examined the final uploaded notebook to check the final forecast rankings and DM-HAC tables when preparing this log. Some prior explanations referred to documentation for the `arch`, `scikit-learn`, and XGBoost packages.
- ChatGPT generated this Markdown log at my request. My prompts appear verbatim, AI responses are summarized. I will complete each **Outcome** field and the **AI Audit** section myself.

## 3. Communication History

> **Conversation note.** This log begins with the initial Bitcoin capstone project-design prompt, followed by the instructor data-packet/on-chain download discussion and the daily API data/alignment discussion. It then continues through seven-day-target diagnostics and the 2024 comparison of naive, GARCH, Random Forest, and XGBoost forecasts.

> **Transcript-format note.** User prompts are reproduced verbatim in chronological order where the original text is available, including tables, spelling, and punctuation. AI responses are summarized rather than reproduced in full. All Outcome fields have intentionally been left blank.


### Exchange 1 — Initial Bitcoin capstone project design and forecasting plan

**Me:**

````text
I am working on my Time Series final project. I upload the guidelines of the project so that you are familiar with it. I have chosen the BTC dataset, where the forecasting target is the squared log returns with horizon of 7 days. The assignment asks for forecasting the next 7 days squared log returns or their mean, and we should have a naive-baseline benchmark, use a classical statistical model and optionally, use machine learning forecasting. The idea I have for the project is to forecast the 7 day mean, so for the baseline we would simply do a trailing seven-day average, that is, the next seven day average is equal to the previous seven day average. For the classical model, since we are dealing with squared returns, an ARCH/GARCH process would be the next step, following the modeling pipeline I saw in my class to choose the best lag and model. We need at least one more model, and originally I wanted to use some exogenous variables related to on-chain metric such as hash rate, mining difficulty and total transaction fees, but I realized that Python doesn't have directly an implementation for GARCH with exogeneous variables to forecast the variance, so we would need to implement by hand the model to maintain the restriction that the variance is positive, which is more work than for which I have time. Hence, we will simply do ARCH/GARCH on their own, but let us also do two ML forecasts using tree-based methods (XGBoost or Random Forests) simply using the market data (lagged) and then another where we use as features the hash rate, difficulty and transaction fees. I am aware that there might be relationships between these variables, but I only want to determine if they are useful for prediction, not to prove a causal relation. I also do not want to forecast these variables, so I would take the 7-lagged values as features so that they are available at the moment of forecasting. We can run different fitting and diagnostics to determine if a particular variable is doing more than others. We can obtain this data from Blockchain.com. I would like you to help me incrementally with this project, particularly with the coding tasks. Like, I mentioned, I have uploaded my projects guidelines, and I'll attach the lecture notes for volatility forecasting and tree methods from my class so that you have access to the forecasting pipelines from my class and you can see what tools we have available. This is also to keep us accountable and we follow a disciplined process. Make sure you understand the assignment and let me know if you agree or if you want to suggest any design choices for the project. From there, I think we should begin with gathering the data and particularly for the exogenous variables, make sure it is aligned with the provided data for the project.
````

**AI:**

Reviewed the proposed capstone design and outlined a common seven-day forward average squared-log-return target for all models. Discussed the naive trailing-seven-day baseline, classical ARCH/GARCH modeling and diagnostics, market-only and on-chain-augmented Random Forest and XGBoost comparisons, and time-ordered training, validation, and testing without forecast-information leakage. Clarified the importance of a seven-day availability lag on blockchain predictors, proposed a reproducible Python data-gathering and date-alignment procedure, and indicated that the fixed instructor-provided Bitcoin dataset should remain the market-data source.

**Outcome:**

---
> On this initial exchange I proposed my initial design for the experiment, as well as providing the instructions for the project. Indicated that I wanted to work with the BTC dataset, with forecasting target the seven day average of the squared log returns. I explained that I wanted to explore ARCH/GARCH modeling, as well as exploring tree based methods. In particular, the central research question I was interested was if using on-chain exogeneous variables such as mining-difficulty, hash rate, transaction fees, etc, could help predict the Bitcoin squared log return. The outcome of this first prompt was both feedback on my original design, as well as the initial code to clean and align the data. First, I noticed that ARCH/GARCH Python packages do have support for using exogeneous variables, but to model the mean. Since our goal was to model the squared log return, having the exogeneous variables as regression coefficients is a little tricky because you have to add constraints to make the variance positive, so it seemed like it was going to require more work and a direct implementation of ARCH-X/GARCH-X or something like that. ChatGPT agreed with me, and we decided to simply use ARCH/GARCH as the classical statistical model, while for the machine learning models, we tested both the market lagged data, as well as the market data together with the on-chain data. ChatGPT warned that the on-chain variables I had chosen (difficulty, hash-rate and transaction-fees) are actually correlated, since difficulty and hash-rate are estimated from each other. This was later confirmed but we agreed to run ablation tests when the time came. AI then provided me with code for downloading the on-chain directly from the API of various sources with blockchain data. I had already chosen Blockchain.com as my provider since I've used it before. Moreover, I knew that one can download the data from Blockchain.com directly as a JSON file, so I downloaded the data first in this way. The next exchange explains why I reverted from this and ended up calling the API.

### Exchange 2 — Inspecting the instructor's data packet and downloaded Blockchain.com JSON files

**Me:**

````text
I attach the data loader as well as the contents of the capstone zip my instructor provided. The data loader already explains the information in the data. I downloaded by hand the hash_rate, difficulty and transaction_fees (BTC) json files from Blockchain.com. This should be all the historical data, but it is not indexed by dates, rather it is indexed by n. The x-value of each data point seems like sort of timestamp. I haven't used the API directly, but in any of the two cases we need to convert the timestamp to datetime.
````

**AI:**

Examined the instructor's data-loader code, manifest, and validation metadata, alongside the three manually downloaded Blockchain.com JSON files. Identified the JSON `x` fields as Unix timestamps in milliseconds and supplied Pandas code to convert them to dates and align hash rate, difficulty, and BTC-denominated transaction fees by their actual timestamps. Reported that the all-history downloads were sampled at roughly four-day intervals rather than containing daily observations. Explained the instructor's fixed 2020–2024 Bitcoin dataset and 60/20/20 split, provided a cautious method for creating lagged on-chain features without interpolation from future observations, and recommended obtaining unsampled daily data from the API before final modeling.

**Outcome:**

> Upon examining the data I had downloaded (which was all the histotical data for each of the on-chain features), I saw that the quantities were labeled with timestamps, and it wasn't clear which date each corresponded to. I asked ChatGPT to confirm this and indeed they were Unix timestamps. Moreover, it confirmed that the data was not daily, but rather that it was sampled from 4-days intervals. Hence, it was necessary for me to go back and use the Blockchain.com API to download the data daily. I inspected the code from Exchange 1 that downloads the data: it does it daily and it begins with data from a month before the BTC data that was provided begins (that is, it requested data starting from December 2019). Now, I had mentioned this on the first prompt: the target is 7 day average of squared log returns. I did not want to forecast these quantities either, so the easiest thing was to use the 7-day lagged values of the mining difficulty, hash-rate and transaction fees. Hence, all of these quantities are available at the forecast origin, and in particular, getting the data from December 2019 allowed us to also have the lagged on-chain data available at the beginning of the training data. However, the rest of the code first aligned the on-chain data with the BTC provided data, and then it lagged it, producing missing entries for the lagged data for the first 7-days. I inquired about this for my next prompt, but the solution is simple: lag first and then align the data together. I should also say that the code also produces a nice CSV file with all the aligned data. 

### Exchange 3 — Using earlier on-chain observations and beginning exploratory analysis

**Me:**

````text
You are correct that the manually downloaded data skips observations, so I went back and used the API to download the data. Here is the raw data from the API call from Blockchain.com, getting the data from December 2019, just a month before our BTC data of interest. Obviously after that, we lag the on-chain features by seven days, so on the aligned and combined data, we do not have the 7-day lagged features until; January 8. However, I wonder if we first lag the features and then we align with the BTC data we can have the lagged quantities available for every day. Would this count as data leakage? We would still be using information from the past, they would just be data outside the training period, which I then understand why we would want to avoid it to preserve the training period. Regardless of that, I think we can move onto doing some exploratory analysis.
````

**AI:**

Explained that lagging the full on-chain series before joining to the Bitcoin index is appropriate and does not cause data leakage, because December 2019 observations are valid historical predictor context for January 2020 forecast origins. Inspected the new API-derived daily data and the research notebook, identified that applying the seven-day lag after restricting to the Bitcoin date range had unnecessarily created seven missing rows, and supplied corrected Pandas alignment code. Proposed initial exploratory analysis of training-period Bitcoin prices, percentage log returns, squared mean-model residuals, rolling volatility, and on-chain predictors; provided ACF, Ljung–Box, ADF, Jarque–Bera, and ARCH-LM code. Interpreted preliminary heavy-tail and ARCH-test findings cautiously and recommended comparing candidate ARCH/GARCH specifications rather than selecting GARCH(1,1) without diagnostics.

**Outcome:**

> As mentioned on the previous outcome, AI provided me with code that after obtaining the dataframe, it lags the on-chain variables by 7 days, and then it aligns the data with the BTC returns data. This way we have observations for the lagged features at the start of the training period, given that the API requested data from December 2019. That is, the last days of December 2019 provide the lagged features available for the first days of January 2020. I asked out of caution if this was data leakage since the information was technically from before the training period, but ChatGPT advised me this was okay. Per the projects instructions, we only need the day t data to be fully available in order to forecast day t+1. With the data complete and aligned, I also asked for the code for the exploratory analysis. I should have been more clear about what that meant, but AI understood I wanted plots of the series, ACF, Portmanteau tests, ADF tests for stationarity and Jarque-Bera tests on the residuals from a constant mean model. I noticed that most of these tests were done on the series $residuals^2 = (log_return - mean)^2$. I suggested we already have `squared_log_returns`, so we can run diagnostics on this series instead. The mean is close to zero so results should be really similar anyway. Another choice from AI I adopted was to multiply log_returns by 100 so that they are in log-percent. Then this means the squared log returns get multiplied by $100^2 = 10000$. Moreover, our target is the seven day average, so my next prompt asked to run exploratory diagnosis on the series we are trying to understand. Additionally, there was code computing the correlation for each of the on-chain features. As expected, the correlation between difficulty and hash-rate is incredibly high. We still proceeded to work with them for the sake of science, but we might run ablation tests to get a better grasp at what is helping and what isn't.


### Exchange 4 — Refocusing exploratory diagnostics on the seven-day target

**Me:**

````text 
I noticed you started from log_return, subtracted the mean, and then squared those residuals. We are trying to forecast the seven day average of squared log-returns, so why are we not starting using the squared log_return column from the data? I just did the exploratory analysis you suggested. Again, why are we not doing it on the series we are trying to forecast. I also need the code for the rest of the exploratory diagnostics.
````

**AI:**

Clarified the distinction among daily log returns, squared daily log returns, and the actual forward seven-day mean of squared returns. Supplied Python code to construct the forward target within the training split, calculate descriptive statistics, plot the target and rolling moments, examine its ACF/PACF, run ADF and Ljung–Box tests, and separately perform return-based ARCH/GARCH diagnostics. Explained the mechanical autocorrelation induced by overlapping forward seven-day targets.

**Outcome:**

> The outcome for this was similar code but now using the log squared returns, as well as diagnostics for the seven-day mean of the squared log return. The code takes the square log return, used `chronological_split` (which splits the data 60/20/20 respecting time, as required). It also constructs the 7-day mean series, and runs ADF tests for stationarity on the seven day mean series. It also plots the series and plots the distribution of the values of the series. It plots the ACF and PACF of the series. ChatGPT also warned that for the seven day mean serires, it would be natural to observe high autocorrelations for the first few lags, so we also plotted the ACF for the squared returns. Additionally, the code also computed the Ljung-Box test on the target series for several lags, which I accepted. The last diagnostic was to use the log return series with a constant mean specification to do the ARCH-LM tests. The AI code only tested lags 1, 2, 5 and 10. I reported the results from these lags, from which the test only rejected at lag 10, but I expanded the candidate list for lags to be more exhaustive. More on this later.

### Exchange 5 — Interpreting initial stationarity, Ljung–Box, and ARCH-LM tests

**Me:**

````text
I've completed these diagnostics, the ADF strongly rejects, as well as the ljung-box for all lags 7, 14, 21, 28, though for the daily log returns, it only rejects for lag 7. The LM test rejects only at lag 10.
````

**AI:**

Interpreted the strong ADF rejection as evidence against a unit root, the target's Ljung–Box rejections as partly caused by overlapping windows, and the return Ljung–Box and ARCH-LM results as more limited evidence of serial dependence and conditional heteroskedasticity. Proposed initial ARCH and GARCH candidates and code to fit them using percentage returns and comparable AIC/BIC calculations.

**Outcome:**
> AI provided a few candidates for ARCH/GARCH models. It provided interpretation for the results for the statistical tests, which confirmed my own conclusions. However, I found the ARCH-LM test for accurate lags to be very narrow. We know from class that we know that a rejection of the test does not say that a particular smaller lag is zero, or that particular lag at which the test rejects is optimal. I raised this issue to AI on my next prompt. Another particular observation was that the ACF of the squared log returns had a single spike at lag 7.

### Exchange 6 — Expanded ARCH-LM lag-order results

**Me:**

````text
I actually repeated the LM test on several lags more and I found the test rejects at smaller lags:&#x20;

| <br>**Lags** | **LM statistic** | **p-value** |          |
| -----------: | ---------------: | ----------: | -------- |
|            0 |                1 |    2.594165 | 0.107258 |
|            1 |                2 |    2.663560 | 0.264007 |
|            2 |                3 |    2.675522 | 0.444403 |
|            3 |                4 |    6.321179 | 0.176412 |
|            4 |                5 |    6.319583 | 0.276352 |
|            5 |                6 |    6.312258 | 0.389135 |
|            6 |                7 |   20.074982 | 0.005410 |
|            7 |                8 |   20.412226 | 0.008884 |
|            8 |                9 |   20.417773 | 0.015502 |
|            9 |               10 |   20.473561 | 0.025079 |
|           10 |               11 |   21.230956 | 0.031056 |
|           11 |               12 |   21.224582 | 0.047187 |
|           12 |               13 |   21.374215 | 0.065849 |
|           13 |               14 |   21.482782 | 0.089885 |
|           14 |               15 |   21.669255 | 0.116774 |
|           15 |               16 |   21.652501 | 0.154770 |
|           16 |               17 |   21.659625 | 0.198166 |
|           17 |               18 |   22.009902 | 0.231546 |
|           18 |               19 |   22.298831 | 0.269637 |
|           19 |               20 |   22.352516 | 0.321693 |
````

**AI:**

Highlighted that the first significant ARCH-LM regression involved seven lagged squared residuals and discussed a possible weekly pattern, while cautioning that varying lag orders introduces multiple-testing concerns. Recommended inspecting squared-residual autocorrelations and the coefficients in an ARCH-LM auxiliary regression; suggested including ARCH(7) and ARCH(10) as exploratory candidates alongside parsimonious GARCH alternatives.

**Outcome:**

> Indeed it is true that inspecting this many lags introduces concerns of observing a false-positive, especially since none of the rejections were particularly very strong. So the recommendation was not to necessarily treat these results as evidence, but more indication for candidates to test during validation. Because the ACF for the squared log-returns does show a spike a lag 7, that did indicate that lag 7 was perhaps a stronger candidate.

### Exchange 7 — Lag-seven ACF spike in squared daily returns

**Me:**

````text
Indeed, and this was something I examined before, the ACF of the squared log-returns shows a singular spike at lag 7
````

**AI:**

Connected the isolated lag-seven ACF spike to the ARCH-LM pattern. Explained the difference between conventional ARCH(7), which includes squared shocks at all seven lags, and a restricted model involving only lag seven. Suggested checking the lag-seven correlation separately by year and across the later part of training to assess whether a weekly effect persisted.

**Outcome:**

> I took the suggestion of examining the ACF lag by year (within the training data of course), and found that only within the first two years the lag 7 ACF spiked, while for 2022, the spike dissapeared. This was a good suggestion by AI, the idea was to avoid an exceptional period during 2020 (most likely corresponding to COVID), where the squared log returns spike massively. Unfortunately, as discussed below, the lag-7 ACF spike did not carry through. The idea was that if it carried through in a somewhat stable manner, we might have some weekly structure. Sadly, it seems this was not the case.

### Exchange 8 — Checking whether the seventh-lag effect is stable

**Me:**

````text
Unfortunately, the lag-7 ACF is not stable across all training years. The results were;

|     **Year** | **Observations** | **ACF lag 7** |          |
| -----------: | ---------------: | ------------: | -------- |
|            0 |             2020 |           365 | 0.119326 |
|            1 |             2021 |           365 | 0.117754 |
|            2 |             2022 |           365 | 0.003785 |

It remains similar on 2020 and 2021, but not 2022. From the ACF plot for 2021-2022, the spike at lag 7 is mostly inside the confidence intervals.
````

**AI:**

Interpreted the positive 2020–2021 lag-seven autocorrelations and near-zero 2022 correlation as evidence against a stable weekly volatility cycle. Distinguished full-sample ARCH-LM significance from stability over time, advised against imposing a special seventh-lag variance model, and proposed moving to a limited conventional ARCH/GARCH comparison and later chronological validation.

**Outcome:**
> As mentioned above, this ruled out somewhat a stable weekly structure. What we have so far is that the ARCH-LM test suggest there might be some conditional time-varying variance across the entire training period, but examining each year, the structure is not stable to be fully confident on it. Moving on from here, we assumed Gaussian innovations, and AI proposed a list of models to fit and then test. These were ARCH(1), ARCH(5), ARCH(7), GARCH(1,1) and GARCH(2,1). I allowed ARCH(1) and ARCH(5) since they were simpler models in theory, and while ARCH-LM did not reject at these lags, if their forecast performance during validation was good, then the model simplicity makes it a better choice. I inquired on my next prompt about the reasoning for suggestng these GARCH specificiations.

### Exchange 9 — Fitting ARCH/GARCH candidates and explaining model orders

**Me:**

````text
We should I would like the code to fit the models to the training data. Report AIC, BIC and the fitted coefficients for each of the models, as well as the persistence. I also want to know your reasoning for suggesting GARCH(1,1) as well as GARCH(2,1).
````

**AI:**

Explained the variance equations for GARCH(1,1) and GARCH(2,1), emphasizing parsimonious volatility persistence and a modest additional short-lag shock coefficient. Defined persistence as the sum of ARCH and GARCH coefficients. Supplied `arch` code to fit Gaussian constant-mean ARCH(1), ARCH(5), ARCH(7), GARCH(1,1), and GARCH(2,1) to percentage training returns, with common `hold_back`, and tables of likelihood, AIC, BIC, convergence, coefficient estimates, standard errors, and persistence.

**Outcome:**
> I asked for the code to fit the suggested models to the training data, as well as for the AIC/BIC of each of the models, their coefficients, and the persistence for each of the models. I accepted the GARCH suggestions on the basis that letting the variance remember might help with the forecasting, specially since we know more generally that the markets have regimes and so integrating this information into the model might help. The code fits the models under a constant mean specification and uses the `log_return` series (in percentage) as its input. The code also checks for convergence of the model, which they all converged.
---

### Exchange 10 — Comparing training fit and moving to weekly validation

**Me:**

````text
The results are the following:&#x20;

|                        |**Log-Likelihood** |    **AIC** |   **BIC** | **Persistence** | **Convergence** |   |
| ---------------------: | ---------: | --------: | --------------: | --------------: | - |
|              **Model** |            |           |                 |                 |   |
|                ARCH(7) | -2951.7512 | 5921.5024 |       5966.4313 |          0.5139 | 0 |
|                ARCH(5) | -2963.6911 | 5941.3821 |       5976.3268 |          0.4218 | 0 |
|             GARCH(1,1) | -2991.9274 | 5991.8547 |       6011.8231 |          0.9361 | 0 |
|             GARCH(2,1) | -2989.6473 | 5989.2945 |       6014.2550 |          0.9261 | 0 |
|                ARCH(1) | -3016.3426 | 6038.6853 |       6053.6616 |          0.0345 | 0 |

From this ARCH(7) has both the lowest AIC and BIC. For experimentation, ARCH(8);

| ARCH(8) | -2949.8469 | 5919.6938 | 5969.6148 | 0.5624 | 0 |
| ------: | ---------- | --------- | --------- | ------ | - |

Of course I am doing a bit of selection bias here, but I thought it would be interesting to check. We can now move onto validation to select a model
````

**AI:**

Observed that ARCH(7) led the originally prespecified models on both information criteria, whereas ARCH(8) slightly improved AIC but not BIC; noted the need for a common estimation sample when comparing likelihood criteria. Provided a 2023 expanding-window, weekly-refitted backtest across six candidates and the naive seven-day historical mean, with analytical seven-step ARCH/GARCH forecasts transformed into the average future squared log-return target. Supplied MAE/RMSE comparison code and forecast plots.

**Outcome:**
> After observing the AIC/BIC results for the candidate list, for which I also included ARCH(8) since I noticed it had better AIC (though worse BIC), for the sake of experimentation I thought it would be interesting to also test it in validation, though I agree being careful with adding more models after seeing results from other candidates. Since it was only one extra model, there was no harm on adding it. Now that we had a final list of candidate models, AI provided me with the code for generating the forecasts for each of these models for validation. The code specifies the parameters for each proposed model, it computed the forecasting origins in order to do expanding window validation, for which I checked the first forecast origin occurs on Dec 31 2022 and we have 52 nonoverlapping target windows through 2023. The model produces seven forecasts of conditional variances, and from those we average them to get our target. The code also uses the actual seven days average for comparison, as well as constructing the naive benchmark. We then use these forecasts to compute MAE and RMSE for each of the models over the entire validation period. AI included a column where it scales the error by $10000$ for better readability. This is exactly the kind of output that I wanted, so now I could compare the models performance based on how they compare to the actual observations and if they beat the baseline. I also asked for the code to produce the plots for the best two models according to RMSE and also of the Naive baseline.

### Exchange 11 — Validation metrics and considering Diebold–Mariano inference

**Me:**

````text
The table is the following:&#x20;

|                **MAE** | **RMSE** | **MAE (x1e4)** | **RMSE (x1e4)** |           |
| ----------: | -------: | -------------: | --------------: | --------- |
|   **Model** |          |                |                 |           |
|  GARCH(1,1) | 0.000645 |       0.000746 |        6.446823 | 7.458555  |
|  GARCH(2,1) | 0.000651 |       0.000754 |        6.506509 | 7.542421  |
|       Naive | 0.000530 |       0.000755 |        5.296405 | 7.550006  |
|     ARCH(8) | 0.000678 |       0.000787 |        6.778306 | 7.867284  |
|     ARCH(7) | 0.000695 |       0.000801 |        6.945267 | 8.008262  |
|     ARCH(5) | 0.000766 |       0.000862 |        7.655748 | 8.624212  |
|     ARCH(1) | 0.000936 |       0.001011 |        9.364105 | 10.112730 |

So from this perspective, the GARCH models perform better. Particularly looking at RMSE, the differences are very small. I wonder if something like Diebold-Mariano is something that make sense here to test to see if these differences are significant, though again testing after seeing the results is not the best practice.
````

**AI:**

Contrasted GARCH(1,1)'s narrow RMSE lead on validation with the naive baseline's superior MAE. Explained the Diebold–Mariano loss-differential test, the relevance of nonoverlapping weekly targets, and the need for HAC standard errors. Supplied a `statsmodels` implementation for squared and absolute losses, including sensitivity to HAC bandwidth, and warned that using validation-period DM p-values after choosing a winner introduces post-selection issues.

**Outcome:**
> Indeed as I observed, GARCH(1,1) was the best choice in RMSE, which means that for this model the total "size" of the error vector is the smallest for this validation forecast, but the Naive baseline had actually the smallest mean absolute error. Choosing a model based on better RMSE is defensible, since MAE does not penalize for large errors, while RMSE does. However, the performance in RMSE from GARCH vs the naive baseline is almost microscopic, which begs the question of whether this error difference is significant or if its by chance. I suggested running a Diebold-Mariano test on the pair GARCH(1,1), Naive. I do understand that performing the test after seeing the MAE/RMSE results is introducing selection bias, I treated this more as diagnostics and exploration, rather than evidence that one model was better than the other. In other words, before running the DM-test, I had already chosen GARCH(1,1) to be the final forecasting method based on RMSE. The DM-test was performed for both the squared error, as well as the absolute error, and the AI code provided the code to compute it using a `dm_hac` function that AI gave that directly implements the test. While packages must already include some function that already implements the test, AI's direct implementation scaled the errors by a factor of $10,000$ for numerical stability (the quantities we were working are really small, and theres a chance that they might become too small for a coputer to handle, particularly if the difference of errors is really close). Hence, I kept this implementation and used it to run the test. Both comparisons, with squared error and with absolute error failed to reject.

### Exchange 12 — Freezing GARCH(1,1) and testing against the naive model

**Me:**

````text
I agree we should fix GARCH(1,1) based on the results and compare with the baseline. Let's freeze this choice and fit on the test data and compare.
````

**AI:**

Clarified that fixing the model specification does not mean fitting once to all 2024 test observations: the correct protocol re-estimates Gaussian constant-mean GARCH(1,1) using data available through each weekly origin. Supplied code for 52 nonoverlapping seven-day forecasts in 2024, paired naive forecasts, MAE/RMSE, relative improvements, diagnostic plots, and the preplanned squared-loss DM-HAC comparison.

**Outcome:**
> The output was the code to now use the GARCH(1,1) fitted to the available history and forecast the last year of the available data. The code again builds the dates for the 52 forecasting origins we will need for forecasting and uses the historical returns available at the forecast origin. It constructs the naive benchmark for the test period. We fit the frozen GARCH(1,1) using the historical data available at the forecast origin and forecasts the next seven days conditional mean and variance. We take their mean over 7 days and use these to compute the forecast for the squared log return. The procedure was to forecast the variance via GARCH, then add the mean squared, and that recovers the forecast for the log squared return by definition of the variance. Naturally, we then compute MAE and RMSE for both models and we compare the improvement of GARCH  over the baseline. The code for the plots using the forecasted values was also given, where the plot includes the forecast from GARCH, the baseline and the actual test values. While the plot was good, I also wanted to see some prediction intervals, so for my next prompt I asked to produce the code for computing the intervals. Based on the DM test I suggested on my previous prompts, we ran the DM test as well to see if the difference in performance is significant.

### Exchange 13 — Interpreting GARCH test scores and constructing prediction intervals

**Me:**

````text
GARCH(1,1) indeed does improve RMSE;

               | **MAE** | **RMSE**       | **MAE (x1e4)**  | **RMSE (x1e4)** |          |
| ----------: | -------: | -------------: | --------------: | -------- |
|  GARCH(1,1) | 0.000551 |        0.00071 |        5.508137 | 7.101403 |
|       Naive | 0.000591 |        0.00086 |        5.905032 | 8.596143 |

but by a small amount. It also improves MAE. I would like the forecasting plots to include confidence intervals, though I understand these are computed based on our assumptions of the distribution. The DM-test is very close to rejecting&#x20;

|                   **Test Result** |            |
| -------------------: | ---------- |
|                    N | 52.000000  |
| Mean loss difference | -23.463755 |
|       Standard error | 11.997941  |
|     DM-HAC statistic | -1.955649  |
|              p-value | 0.055996   |

but it fails to do so, though the statistic does favor GARCH(1,1).
````

**AI:**

Calculated the test-period relative RMSE improvement over naive and interpreted the two-sided DM-HAC result (p about 0.056) as suggestive rather than significant at 5%. Distinguished parameter confidence intervals from predictive intervals for the *realized* seven-day average of squared returns. Supplied `arch` simulation code using 10,000 Gaussian GARCH seven-day paths per origin to construct 95% equal-tailed prediction intervals, plus coverage and interval-width calculations and plots.

**Outcome:**
> The main thing I wanted from this exchange was the new plots with the prediction intervals. Since we were under the assumption of normally distributed innovations, AI produced code that computed the intervals for the point forecasts via simulation with normal shocks, and then it adds these on top of the previous plot from the previous prompt. To be more specific, the `arch` function generated 10,000 possible seven-day return paths using the fitted GARCH model. It then calculates the 2.5th and 97.5th percentiles. Theres some additional code that measures how many of the actual observations fall outside the range of these intervals, which we expect to be 95% of the observations to fall within the interval. In the end the result was close to 88, but we only had 52 observations, so there wasn't a lot of data and the measurements could be noisy. Overall, the results tell us that in test data, GARCH(1,1) indeed beat the baseline, and running the DM test yet again it also failed to reject, if ever so slightly. This was the last exchange regarding GARCH and classical statistical models.

### Exchange 14 — Moving from GARCH coverage diagnostics to machine learning

**Me:**

````text
I agree we can now move on to the machine learning forecasting methods. The covergae results are interesting;

|                                     **Metric** | **Value** |           |
| -------------: | ----------------------------: | --------- |
|              0 |              Nominal coverage | 0.950000  |
|              1 |            Empirical coverage | 0.884615  |
|              2 |           Observations inside | 46.000000 |
|              3 |           Observations below | 5.000000  |
|              4 |           Observations above | 1.000000  |
|              5 | Average interval width (x1e4) | 26.729852 |

It doesn't quite reach the 95%, but it is somewhat close at 88%. There are not enough observations, so there is definitely some uncertainty in this measurement.
````

**AI:**

Interpreted the 46/52 empirical prediction-interval coverage cautiously, noting the small sample and imbalance between lower and upper misses, while recommending no test-informed changes to frozen GARCH. Proposed supervised direct prediction of the forward seven-day squared-return mean using lagged market volatility, returns, price range, volume, and seven-day-lagged on-chain hash rate, difficulty, and fees. Supplied data construction, leakage-aware target construction, initial Random Forest/XGBoost specifications, weekly expanding-window validation code, and MAE/RMSE comparison against naive and GARCH.

**Outcome:**
> This was a follow-up to my previous discussion, which I discussed above. While AI gave a brief discussion on why we may have a discrepancy with only having 88% coverage, as I explained above this could be due partly to the small sample size, allowing for noisy results. Aside from this, AI now provided the initial steps and code to move on with XGBoost and Random Forest leafs. AI provided code for data construction, which we had already done early on. In particular, I intended to use the seven day lagged on-chain variables as features, and I have already described previously how we obtained this data and constructed the data frame so that at time t, the lagged t-7 on-chain variables are fully available. This was done not only because we wanted to forecast the seven day mean, but also because it was unclear whether for instance the on-chain data becomes available after or before closing price, particularly since cryptocurrency markets are 24/7. Hence, the safe avenue was to take data from a previous week that is assured to have been available at the forecast origin. AI randomly chose "conservative" hyperparameters to run the random forest and XGBoost. It separated the features into market features (returns, squared log returns, seven, fourteen and 30 day rolling mean for momentum strength, seven day mean return, log volume and change in volume over seven days) and on-chain features (hash-rate, difficulty and transaction fees). It fitted the models with the proposed hyperparameters and measured RMSE, MAE on the validation set using the same validation procedure as before. The code combines the validation results from the GARCH forecast for better comparison.
---

### Exchange 15 — Interpreting initial Random Forest and XGBoost validation

**Me:**

````text
Here  are the validation results:&#x20;

|                                **MAE (x1e4)** | **RMSE (x1e4)** |         |
| --------------------------: | --------------: | ------- |
|                   **Model** |                 |         |
|      RF - Market + On-chain |          4.7368 | 6.1368  |
|                 RF - Market |          4.8560 | 6.1936  |
| XGBoost - Market + On-chain |          5.4107 | 7.3402  |
|                  GARCH(1,1) |          6.4468 | 7.4586  |
|                       Naive |          5.2964 | 7.5500  |
|            XGBoost - Market |          7.0061 | 13.2810 |

These are very promising, on validation, Random Forest with On Chain variables was the better performing model.
````

**AI:**

Noted that Random Forest improved substantially over the classical benchmarks, while adding on-chain variables produced a smaller incremental gain over market-only Random Forest. Flagged the poor market-only XGBoost RMSE as possibly driven by an outlying forecast. Provided code for time-indexed validation errors, worst-error dates, and observed-versus-forecast Random Forest plots; suggested a bounded hyperparameter comparison, maintaining separate market-only and on-chain controls.

**Outcome:**
> I reported the validation result from the proposed hyperparameters. The first observation was that the machine learning models perform much better than GARCH and the baseline (except XGBoost with only Market features). AI wanted to investigate what caused the dramatic error for XGBoost, so I obliged since it was interesting to see what happened, and it is also a good experience to dig deeper into what went wrong for a model. AI gave me the code to plot the point errors and this indicated that XGBoost with only market features makes a single error forecast, which makese sense why the RMSE exploded from this single outlier. The diagnostic was useful for learning purposes but I do not think I'll include this in the final report (though I'll keep it in the Jupyter Notebook). The reason is that right now AI simply chose parameters and ran the pipeline, without doing a more systematic search for better parameters. For my next few prompts I directed AI to give me code for searching more systematically.
---

### Exchange 16 — Extreme forecast errors, grid search, and on-chain ablation

**Me:**

````text
The market only XGBoost misses a forecast between August and September by a big amount, which is what makes the error be so large for this model;

|                              **Mean Error** | **Error Std**              | **Largest Absolute Error** | **Worst Forecast Date** |            |
| --------------------------: | ------------: | -------------------------: | ----------------------: | ---------- |
|                 RF - Market |      1.728115 |                   6.005623 |               22.775574 | 2023-03-11 |
|      RF - Market + On-chain |      1.033038 |                   6.108239 |               24.997717 | 2023-03-11 |
|            XGBoost - Market |      3.606676 |                  12.906580 |               78.047189 | 2023-08-19 |
| XGBoost - Market + On-chain |      1.977282 |                   7.137833 |               24.540339 | 2023-03-11 |

For the other models, they all suffer from a spike in March 2023 that none of the models could capture fully, they underestimate how big the spike became. It seems that aside from that, with these hyperparameters, all models remain somewhat close, at least qualitatively from the plots. We should try to tune these parameters in training and validation before we freeze the models. We should do a grid search through the different options to find optimal fits. I think we should also do some ablation test with the on-chain variables to figure out which ones are adding predictive value.
````

**AI:**

Proposed time-series-aware hyperparameter grid searches for RF and XGBoost, using chronological folds within 2020–2022 with a label embargo to prevent leakage and nonoverlapping weekly validation origins. Provided parameter grids, a clipped squared-loss scorer, `GridSearchCV`, and weekly-refit validation evaluation. Designed an eight-subset ablation of hash rate, difficulty, and transaction fees for Random Forest, holding the full-feature model's hyperparameters fixed to isolate the effects of changing predictors.

**Outcome:**
> I suggested a grid search through various configurations of hyperparameters and then choose the configurations that do well in validation. The search was concentrated on `max_depth`, `min_samples_leaf`, `max_features` for Random Forest, keeping the numbers of estimators fixed at 300 (which according to AI it is the common, conservative initial choice, for this project I judged this appropriately). For XGBoost the parameters were `n_estimators`, `max_depth`, `learning_rate`, and `min_child_weight`. The grid search was relatively small for computation/complexity purposes and so it was still somewhat arbitrary with respect to the values in the grid. The grid search was performed using chronological folds, each validation block containing 91 days, each also using a 7 day gap so that the final training observation is known before the first validation forecast origin. The code then chooses the hyperparameters that maximize negative mean squared error (so the error is minimized). Then the code returns to the expanding windows validation, tunes the models using the hyperparameters chosen and then computes validation performance. This exchange also generated the code for the first ablation study, where the hyperparameters are those selected for the full on-chain feature set for the Random Forest (there was no ablation code produced here for XGBoost). The ablation study goes through computing the validation performance of different configurations of the feature set to see which of the on-chain features is actually adding predictive power. Ultimately, these were AIs first attempts, but  I was still not satisfied with how we chose parameters, it still felt very arbitrary to me, and even the parameters chosen randomly before we did the grid search performed better in validation using the full set of features. Of course, theres danger of overfitting, but I asked to try a different method for choosing hyperparameters.
---

### Exchange 17 — Understanding differences between tuned RF and ablation scores

**Me:**

````text
Very interesting results across the board:

|  0 | RF      | Market            | 14.517461 | {'max_depth': 5, 'max_features': 1.0, 'min_sam... |
| -: | ------- | ----------------- | --------- | ------------------------------------------------- |
|  1 | RF      | Market + On-chain | 15.932578 | {'max_depth': 4, 'max_features': 0.8, 'min_sam... |
|  2 | XGBoost | Market            | 15.207432 | {'learning_rate': 0.03, 'max_depth': 2, 'min_c... |
|  3 | XGBoost | Market + On-chain | 15.306017 | {'learning_rate': 0.07, 'max_depth': 2, 'min_c... |

From the output I can't quite read which were the selected parameters for the min_samples_leaf or for the min_child_weight parameters. With the chosen hyperparameters, the RMSE and MAE were;

|                                **MAE (x1e4)** | **RMSE (x1e4)** |         |
| --------------------------: | --------------: | ------- |
|                   **Model** |                 |         |
|                 RF - Market |          4.8662 | 6.2487  |
|      RF - Market + On-chain |          5.0949 | 6.5783  |
| XGBoost - Market + On-chain |          5.0945 | 7.2163  |
|                  GARCH(1,1) |          6.4468 | 7.4586  |
|                       Naive |          5.2964 | 7.5500  |
|            XGBoost - Market |          7.1705 | 17.2288 |

and this time the random forest without the on-chain variables won. Running the ablation tests, we find that difficulty_lag_7 is the parameter provides the better performance.

|                                                     **MAE (x1e4)** | **RMSE (x1e4)** |        |
| -----------------------------------------------: | --------------: | ------ |
|                                        **Model** |                 |        |
|                                  difficulty_lag7 |          4.9820 | 6.4620 |
|                  difficulty_lag7 + fees_btc_lag7 |          5.0087 | 6.4806 |
|                 hash_rate_lag7 + difficulty_lag7 |          5.0561 | 6.5677 |
| hash_rate_lag7 + difficulty_lag7 + fees_btc_lag7 |          5.0949 | 6.5783 |
|                   hash_rate_lag7 + fees_btc_lag7 |          5.2720 | 6.7880 |
|                                   hash_rate_lag7 |          5.2657 | 6.8088 |
|                                    fees_btc_lag7 |          5.4482 | 6.9437 |
|                                      Market only |          5.4657 | 7.0195 |

I am a little confused though, because for the ablation studies, the market RF was the worst performer, so there is a discrepancy here.
````

**AI:**

Explained that the independently tuned market-only RF and the ablation's market-only RF used different hyperparameters: the ablation deliberately held the full on-chain RF parameters constant. Distinguished the predictive value of a feature under a fixed estimator from the performance of a separately tuned forecasting procedure. Supplied code to print complete selected parameters and to run a controlled two-by-two comparison crossing market/full feature sets with market/full hyperparameter configurations. Highlighted difficulty as promising but not yet conclusively superior after independent tuning.

**Outcome:**
> Before requesting a more systematic tuning method, I was confused why on the ablation study using market only for example, we got different MAE and RMSE than when we evaluated the validation forecast of Random Forest with Market Only features. This was first a misunderstanding of what AI had done for the ablation studies, which was clarified on this exchange, as well as from me further investigating how Cross-Validation and Ablation Studies works. The discrepancy, as I mentioned on what the code does in the previous entry, comes from the fact that the ablation tests use a different set of hyperparameters than RF-Market. In fact they use the hyperparameters of RF- Market+On-Chain, where we see actually that the validation error scores for hash rate + difficulty + fees are the same as for RF - Market + On Chain. Once understood why, we moved on to my concern about not having a fully systematic approach to the tuning.

### Exchange 18 — Planning one final systematic tuning without random time splitting

**Me:**

````text
Yes, I think we need a more systematic approach for tuning the hyperparameters, the grid search feels very arbitrary. However, I agree that testing more and more configurations will lead us into overfitting. I suggest we try one more hyperparameter tuning. I also suggest that maybe we can split the data manually, I am worried train_test_split is mixing the data randomly
````

**AI:**

Clarified that `TimeSeriesSplit` had not randomly shuffled dates; random *hyperparameter sampling* and random *data splitting* are different. Proposed one final fixed-budget `RandomizedSearchCV` using four manually specified chronological validation periods within 2021–2022, expanding training sets, seven-day target-label gaps, and weekly-spaced score dates. Supplied reproducible parameter spaces and code for five model/feature-set combinations, fold-level diagnostics, and one final 2023 validation table, with an explicit commitment to stop tuning afterward.

**Outcome:**
> The new tuning procedure approaches the problem using `RandomizedSearchCV`, where we chose some ranges over which we wanted the paramaters to be, and then the function samples certain configurations to try for a fixed number of iterations (24) in this case. This is more computationally effective than doing a grid search over the possible values that could be specified, and it scores the parameters to tune maximize the negative square error. This was a more principled approach. Since I wasn't familiar with the function and its random component, I also asked AI to make the folds manually instead of using `TimeSeriesSplit` since I got worried that the general purposes `RandomizedSearchCV` from `scikit-learn` would mess up with the temporal dependence. AI assured me this was not the case, but still constructed the folds manually. Further, AI gave me comments on the ablation results, where difficulty was the feature that helped predict the squared log returns better, so this motivated a shortlisted second try with the better tuned models that would check for full-on-chain vs market vs difficulty only. In principle, this was most likely no the better approach, we should have still done all the tests, but due to time, we decided to move forward with it, as it still gave interesting results. It is also worth mentioning that I didn't want to keep adding tests in this way, since testing repreatedly various times after seing results is making me feel like uncertainty is increasing. Perhaps a better initial design could have prevented this, but my lack of experience with ML methods created the limitation. However, I am happy to have learned and expanded my experience.

### Exchange 19 — Interpreting final training-CV and 2023 validation selections

**Me:**

````text
I upload the results from the most recent tests. The model with the smallest CV RMSE was the random forest with market plus difficulty. Across all folds except the last one, this is the model that also performed the best. For the last fold, the RMSE of RF-Market was actually better, but this shows that across most validation periods, RF-Market+Difficulty is the winner. Considering only XGBoost, XGBoost-Market+On Chain has the best CV RMSE, and across the folds it does perform better than XGBoost-Market. Finally, the final validation score puts RF-Market+On Chain as the winer
````

**AI:**

Distinguished the best model under 2020–2022 chronological cross-validation (RF with market plus difficulty) from the model with the best 2023 validation RMSE (RF with all on-chain predictors). Explained why fold wins and error magnitudes can differ across regimes and recommended freezing the five shortlisted RF/XGBoost configurations rather than launching another search. Supplied code to record final estimator parameters and feature sets, conduct leakage-aware expanding-window 2024 forecasts, and compare all frozen models with GARCH and naive.

**Outcome:**
> I uploaded the Jupyter notebook I was working with since by this stage, the results and data were starting to become a little too big to manually copy them in the chatbox. I reported that across the cross validation folds, RF- Market + difficulty won 3 out of 4 times, but in validation data, RF - Market + full on chain was the model that generalized the best. XGBoost on its own, XGBoost - Market + All On Chain was also the winner, even though compared to the random forest, it was still not the best. This was a good point to stop validation and freeze the parameters chosen for both RF and for XGBoost, and to fit them to the historical data using All-On-Chain, since that was the best performer in the validation data. In order to make the comparison and to answer our research question whether the on-chain variables help as predictors here, we naturally should also freeze the parameters for the Market only feature models in order to compare final performance. We tested RF- Market + Difficulty only as well for comparison purposes too. AI finally gave me the code for the 2024 final forecast and performance comparison. It additionally compares the performance with the naive baseline and the GARCH(1,1) model we had previously fit.
---

### Exchange 20 — Final 2024 ML results, forecast plots, and DM comparisons

**Me:**

````text
Here are the final results. The model that generalized better was XGBoost+On chain, surprisingly. Across the board, the on-chain factor help to predict the target, in fact the worst performing model (not counting the baseline) was RF- Market, GARCH(1,1) outperforms it. I would like to see some plots of the forecasts as well. I want a plot as well that visualizes the performance of each of the models. I think we can also run the Diebold-Mariano test to finalize our conclusions, but not to choose models or tune even further.
````

**AI:**

Summarized the 2024 rankings: XGBoost with market and on-chain predictors had the lowest RMSE; RF with all on-chain predictors had the lowest MAE; market-only RF was worse than GARCH. Provided Matplotlib code for observed-versus-predicted time-series plots, forecast-error plots, horizontal RMSE/MAE comparison charts, and improvement relative to naive. Supplied a HAC-based DM-style function and eight prespecified paired comparisons under squared and absolute loss, plus cumulative loss-differential plotting; emphasized that the tests were for final interpretation, not additional tuning.

**Outcome:**
> I submitted the final results from the notebook and I asked for code for plots. The code outputs a plot with all the forecasts model that we chose, as well as one with the top performers. I also asked for a plot of performance comparison, which produced a horizontal bar chart for RMSE and MAE. AI also gave me the code for a the forecast erros, which is very useful for interpretation and so I decided to keep it. Finally, I asked for a Diebold-Mariano test to measure the siginificance on the difference in forecasting errors between the models. This was once again for diagnostic and interprestation, but not for model selection. At this stage, the experiment is over.

### Exchange 21 — Final DM results and request for the AI-use log

**Me:**

````text
I upload the final version with the DM tables. From what I can see, the difference of the naive baseline versus each of the ML models was significant, but no other pair of comparisons rejected. This is only for RMSE, for MAE the results were not that promising. To finish off our exchanges, please produce the AI log document for me with the same format we have been using so far. Leave the outcome and audit sections blank, as I will fill them out.
````

**AI:**

Reviewed the final uploaded notebook and its paired DM-HAC tables. Verified that the two-sided squared-loss comparisons against naive rejected at 5% for XGBoost with all on-chain features (p=0.0472) and RF with all on-chain features (p=0.0409); other listed squared-loss comparisons and all listed absolute-loss comparisons did not reject at 5%. Prepared this chronological Markdown AI-use log in the format of the previous Time Series logs, reproducing user prompts and summarizing AI replies while leaving every Outcome field and the AI Audit section blank.

**Outcome:**
> This was the final exchange, I submitted the final results for completeness, almost all DM comparisons failed to reject, except for XGBoost with all on-chain and RF with all on-chain, which was promising for our research questions, but more on this on the final report. I asked then for this AI log document that reproduced my prompts verbatim and the AI responses to be summarized. I verified it included all our exchanges and that the outcomes and audit section were blank, so that I could fill those sections in manually.
---

## 4. AI Audit

Given that this was a heavier project, I did use AI more heavily than I had for my previous assignments, particularly in the area of coding and even more explicitly for the Machine Learning aspect of the project, since we didn't have an assignment on that so I needed a bit of assisstance on using what we learned in class.

I gave on my initial prompt the project design I had in mind for the project  with several details on how to proceed. Through every new prompt I made sure the design was respected, unless there was a compelling reason to alter it. I also made sure the AI produced code or statistical decisions respected the projects guidelines.

Initially, the AI had given me code to retrieve the data using the API of Blockchain.com. I went ahead and directly downloaded the data from their website, but I noticed the indexing was using timestamps that were not in datetime. I told AI and I used the code that takes the data from the API but I verified that the data corresponded to our period of interest. Particularly, because we intended to use seven day lagged features, I verified the code retrieved data from days prior the beginning of the training period in order to have all available data ready to use. The AI code however first retrieved the data, then built a joint dataframe using the retrieved data, aligning the data by converting the timestamp to datetime and then it lagged the on-chain features by seven days. This resulted in losing the very first 7 days of lagged features, even though those would be available at the start of the training period. Hence instead I told AI and the new code retrieves the data, lags it, and aligns it appropriately, making sure that for the $t$-th row, the $t-7$ feature is the one available. I verified this by going into the produced CSV files for both the raw data that was extracted from the API, and checking that the lagged columns on the aligned CSV matched the seven day prior data on the raw file. I then verified the data was split into training, validation and test using a 60/20/20 split and respecting chronology by checking the implementation of `chronological_split` from the helper file.

For subsequent code on the first stage for ARCH/GARCH, most of the times was ocassional usage of ADF tests, Ljung-Box, ACF plots, which we are familiar by now and their verification came down to using the statsmodels correct implementation, using the traiuning data for diagnostics and that it called the function appropriately with the right or relevant parameters, such as choosing the number of lags. For instance, the AI code only did the ARCH-LM test for lags 1,2,5, and 10, finding a rejection of the null at lag 10. AI's interpretation was then to move forward with trying ARCH(10), but I suggested we should try lags in between, as there might be a smaller lag that rejects earlier. We know from class a rejection at a particular lag does not mean that all the previous lags are zero. Indeed, the test rejected at lag 7, which was further supported by a spike at lag 7 of the ACF of the log squared return series.

I verified that whenever AI implemented their own version of the tests or constructed prediction intervals by hand, it followed the assumptions of Gaussian innovations we had fixed on the original design. I should mention as well that a limitation of the project was that we did not have evidence of conditional Gaussian residuals, so we should have also tested perhaps a student-t configuration, but due to time we simply moved forward with this assumption. We do have evidence of non-normality for the residuals unconditionally, but this does not mean the conditional variance isn't Gaussian.

When it implemented the DM test, I also made sure that it computed the correct statistic given the type of error we seeked to measure, as well as a correct comparison to the percentiles of the student-t distribution required from the test, which was verified by checking against the lecture notes definition of the DM test.

When AI proposed a list of candidates models, I asked for justification and reasoning for models that we did not have direct motivation for from the diagnostics, such as GARCH(1,1) and GARCH(2,1). Upon reading the explanation, I accepted the AI's reasoning that these models, while the data was not necessarily shouting that we should use them, are common models that perform well and are not as complex in the number of lags they used, so if they performed better than the more complicated models, it made sense to keep them in the running. Also, the mathematical descriptions of GARCH models are suitable for these types of series that show volatility clustering, and variance remembering past shocks.

Moving on to the ML models and forecasting, I inspected the codes more carefully since I was less familiar with how to work with these libraries. I made sure the data splits respected chronology and that training and validation and cross validation tests were not contaminated by future data or from data that shoudl not touch different periods that we were examinating.

AI initially had choosing a set of hyperparameters to fit the models, but I kept pushing for more systematic ways of tuning the parameters, while being mindful of overfitting. I suggested first a grid search, but due to computational limitations, AI had only suggested a small set of parameters from where to search. This made the tuning feel somewhat arbitrary, so I kept pushing for something more systematic, so eventually used a random search that allowed to search through many more configurations more efficiently, choosing at random the parameters within ranges or selected larger sets of possible parameter values thorugh various iterations and then choosing the ones that performed best.

I verified AI choose these hyperparameters on the training data via cross-validation across several hand-constructed folds (which were also verified to only include their own historical data and were not contaminated from data from other folds), choosing parameters for the different sets of features we were interested in measuring their impact on prediction. On validation, we chose teh set of features that performed the best and froze the models, parameters and features that performed better. For comparison purposes we still used other  feature sets (Market only feature sets) to compare if the better performing on validation Market + On Chain features also generalized better on unseen data. I should say that the way the forecasts were computed during cross validations were not exactly the same as the way the forecasts were done for the final period, which in practice we should've tried to have all of our methods be the same. In particular, for cross validation, the models were fitted once on each fold rather that an at every weekly origin, which is the way we did it for the forecast during the test period. I accepted the AI methodology here mainly for computational efficiency, or otherwise cross-validation would have taken longer, but this is a recognized limitation that we take into account and that we also add to the final report. Thus, the CV was a sorts of approximation of the final forecasting procedure.

AI produced several tables and visualizations. When they were not satisfactory to me, I asked for better code that include details I would have liked to see (such as when I asked for prediction intervals) or when a particular table did not display correctly a value that we seeked to understand (for instance, after the parameter tuning happened for the first time, the table was too long horizontally and it was impossible to see from the table which parameters got cut, since the output was cropped). 

Throughout the project I kept AI accountable for the code it produced, and I guided the design decisions of how to move forward. I was also interested in following the pipelines we discussed in class, so I also made sure AI followed the guidelines for the project and that we followed the forecasting pipelines we have been working with throughout the semester.
