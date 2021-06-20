# Bitcoin-Price-Forecasting


Forecasting the price of cryptocurrency using ML/DL algorithms

### Following models were used
1. **XgBoost Regressor:** XGBoost is a popular and efficient open-source implementation of the gradient boosted trees algorithm. Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models. When using gradient boosting for regression, the weak learners are regression trees, and each regression tree maps an input data point to one of its leafs that contains a continuous score. XGBoost minimizes a regularized (L1 and L2) objective function that combines a convex loss function and a penalty term for model complexity. The training proceeds iteratively, adding new trees that predict the residuals or errors of prior trees that are then combined with previous trees to make the final prediction. It's called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.
2. **LSTM:** Long Short-Term Memory networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. It is a variety of recurrent neural networks (RNNs) that are capable of learning long-term dependencies, especially in sequence prediction problems. LSTM has feedback connections which make it capable of processing the entire sequence of data, apart from single data points such as images. An LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.
3. **ARIMA:** Auto Regressive Integrated Moving Average is actually a class of models that explains a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values. This acronym is descriptive, capturing the key aspects of the model itself. Briefly, they are:
   * AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
   * I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
   * MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.<br/><br/>
   We use ARIMA model to analyse our time series first. We know that for it, we need our series to be stationary. So we use the below described techniques to realise if our series is stationary

   * Seasonal Trend Decomposition We use seasonal decomposition to visualise the seasonal and trend components of the time series. We aim to get a residual that is free of trends and seasonality.

   * Dicky Fuller test Dicky Fuller test considers the null hypothesis that the time series under consideration is non-stationary. If p-value is sufficiently low ( less than 0.05) while hypothesis testing, then only we reject the null hypothesis and consider the series to be stationary. The DF test provides us with the p-value which we use to determine the stationarity of our series.<br/>
  
   I've also used certain transformations on the data to see if any of those help in making our model stationary.
   1. Log Transformation
   2. Regular time shift applied to Log transformed prices
   3. Box_Cox power transform
   4. Regular time shift applied on Box Cox Transformed prices

4. **SARIMAX - GARCH:** Seasonal Auto Regressive Integrated Moving Average with Exogenous factors is an extenstion of ARIMA class of models with advantages in seasonality and exogenous factors(variables that affect a model without being affected by it). Generalized AutoRegressive Conditional Heteroskedasticity (GARCH) is a statistical model used in analyzing time-series data where the variance error is believed to be serially autocorrelated. GARCH models assume that the variance of the error term follows an autoregressive moving average process. GARCH assumes that returns have a constant mean but, in real world this won’t completely capture the skewness and leptokurtosis that is present. That’s why ARIMA and GARCH models are so often combined. An ARIMA model estimates the conditional mean, where subsequently a GARCH model estimates the conditional variance present in the residuals of the ARIMA estimation.<br/>

### Dataset Used:

  https://www.kaggle.com/mczielinski/bitcoin-historical-data?select=bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv

### Files present:
  <br/> 
  Data Analysis: DATA_EDA.ipynb<br/>
  Basic ML/DL models: LSTM_and_XgBoost.ipynb<br/>
  ARIMA model: ARIMA.ipynb<br/>
  SARIMAX-GARCH model: SARIMAX_GARCH.ipynb<br/>
