import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fitter import Fitter, get_common_distributions, get_distributions
from scipy.stats import norm, expon, lognorm, gamma, beta, weibull_min, chi2, pareto, uniform, t, gumbel_r, burr, invgauss, triang, laplace, logistic
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from pandas.tseries.offsets import CustomBusinessDay


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Different distributions list
distributions = {
    'Normal': norm,
    'Exponential': expon,
    'Log-Normal': lognorm,
    'Gamma': gamma,
    'Beta': beta,
    'Weibull_Min': weibull_min,
    'Chi-Squared': chi2,
    'Pareto': pareto,
    'Uniform': uniform,
    'T-Distribution': t,
    'Gumbel_R': gumbel_r,
    'Burr': burr,
    'Inverse Gaussian': invgauss,
    'Triangular': triang,
    'Laplace': laplace,
    'Logistic': logistic

}


def read_csv_file(filename, field_name=None, index_col=None, parse_dates=True):
    """
        Args:
        filename (str): Path to the CSV file.
        field_name (str): Name of the field to be read.
        index_col (str): Column to set as index.
        parse_dates (bool or list of str): Whether to parse dates.
    Returns:
        pandas.DataFrame: The data from the specified field.
    """
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(filename, index_col=index_col,
                         parse_dates=parse_dates)

        # Check if the specified field exists
        if field_name is not None:
            if field_name not in df.columns:
                raise ValueError(
                    f"Field '{field_name}' not found in the CSV file.")
            # Return the data from the specified field as a pandas Series
            return df[field_name]

        # Return the whole DataFrame if no specific field is requested
        return df

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except ValueError as e:
        print(e)


# When we want to use all of dataset for trend analysis
# stock_name = 'Historical Alphabet'
# stock_name = 'Historical META'
# stock_name = 'Historical Microsoft'

# When we want to do
# stock_name = 'ALPHABET'
# stock_name = 'META'
stock_name = 'MICROSOFT'
analyzed_field = 'Adj Close'

# Load data
close_prices = read_csv_file("C:/Stock Price lists/" + stock_name + ".csv",
                             field_name=analyzed_field, index_col="Date", parse_dates=True)

close_prices.index = pd.to_datetime(close_prices.index)

# Load the actual next prices data
actual_next_prices_filename = "C:/Stock Price lists/" + \
    stock_name + "_actual_next_prices.csv"
actual_next_prices = read_csv_file(
    actual_next_prices_filename,
    field_name=analyzed_field,
    index_col="Date",
    parse_dates=True
)

if actual_next_prices is not None:
    actual_next_prices.index = pd.to_datetime(actual_next_prices.index)

# Saves a pandas DataFrame to a file.


def save_dataframe_to_file(df, filename, overwrite, index):
    """
    Args:
      df (pandas.DataFrame): DataFrame to save.
      filename (str): Name of the file to save.
      overwrite (bool): Whether to overwrite the file if it exists. Default is True.
      index (bool): Whether to include the index while saving. Default is True.
      delimiter (str, optional): Delimiter character to separate values in the CSV file. Defaults to ','.
  """
    try:
        if os.path.exists(filename) and not overwrite:
            print(
                f"File {filename} already exists. Set overwrite=True to overwrite it.")
            return

        df.to_csv(filename, index=index)
        print(f"Data saved to {filename} successfully.")
    except Exception as e:
        print(f"Error occurred while saving data: {e}")


def do_descriptive_analysis(filtered_data):
    print(stock_name + ' descriptive analysis:')
    print(filtered_data.describe())


# Fit and find the best probability distribution using fitter
def fit_best_distribution(filtered_data, show_best_fit, show_plot):
    distributions_to_fit = [
        'norm', 'expon', 'lognorm', 'gamma', 'beta', 'weibull_min', 'chi2', 'pareto', 'uniform', 't', 'gumbel_r', 'burr',
        'invgauss', 'triang', 'laplace', 'logistic'
    ]

    f = Fitter(filtered_data, distributions=distributions_to_fit, timeout=600)
    f.fit()

    # Plot the best fitting distribution
    f.summary()
    # Print the best distribution
    best_fit = f.get_best()
    best_dist_name = list(best_fit.keys())[0]
    if show_best_fit:
        print("Best fitting distribution:", best_fit)

    if show_plot:
        # Plot the histogram of the data
        plt.figure(figsize=(12, 6))
        plt.hist(filtered_data, bins=30, density=True,
                 alpha=0.6, color='g', label='Data')

        # Plot the best fitting distribution
        x = np.linspace(min(filtered_data), max(filtered_data), 1000)
        dist = getattr(stats, best_dist_name)
        param = best_fit[best_dist_name]

        if best_dist_name == 'uniform':
            pdf_fitted = dist.pdf(x, loc=param['loc'], scale=param['scale'])
        elif best_dist_name == 'beta':
            x_min, x_max = min(filtered_data), max(filtered_data)
            x = np.linspace(x_min, x_max, 1000)
            # For beta distribution, the parameters are 'a', 'b', 'loc', 'scale'
            a, b, loc, scale = param['a'], param['b'], param['loc'], param['scale']
            x_scaled = (x - loc) / scale
            # pdf_fitted = dist.pdf(x_scaled, a, b, loc=loc, scale=scale)
            # Corrected the scaling here
            pdf_fitted = dist.pdf(x_scaled, a, b) / scale
        else:
            pdf_fitted = dist.pdf(x, *param.values())

        plt.plot(x, pdf_fitted, '--', label=f'Best fit: {best_dist_name}')
        plt.title(stock_name + ' Best Fitting Distribution')
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    return best_fit


# detect outliers based on the passed distribution
def detect_outliers(distribution_name, filtered_data, show_outliers, just_upperbound, save_to_file):
    """
    This function identifies outliers based on the provided distribution name and data.

    Args:
    distribution_name (str): Name of the best-fitting distribution.
    filtered_data (pandas.Series): The filtered data for outlier detection.
    """

    distribution = distribution_name
    match distribution:
        case "norm":
            # Identify outliers using the fitted Normal distribution
            normal_params = distributions['Normal'].fit(filtered_data)
            mean = normal_params[0]
            std_dev = normal_params[1]

            # Define outliers as points more than 3 standard deviations from the mean
            outlier_threshold = 1
            lower_bound = mean - outlier_threshold * std_dev
            upper_bound = mean + outlier_threshold * std_dev

            # Identify outliers
            outliers = filtered_data[(filtered_data < lower_bound) | (
                filtered_data > upper_bound)]

        case "expon":
            # Identify outliers using the fitted Exponential distribution
            expon_params = distributions['Exponential'].fit(filtered_data)
            loc = expon_params[0]
            # scale is 1/lambda for Exponential distribution
            scale = expon_params[1]

            # Define outliers as points beyond the 95th percentile
            upper_bound = expon.ppf(0.95, loc=loc, scale=scale)

            # Identify outliers
            outliers = filtered_data[filtered_data > upper_bound]

        case "lognorm":
            # Identify outliers using the fitted Log-Normal distribution
            lognormal_params = distributions['Log-Normal'].fit(filtered_data)
            shape = lognormal_params[0]
            loc = lognormal_params[1]
            scale = lognormal_params[2]

            # Transform the data to the logarithmic scale
            log_transformed_data = np.log(filtered_data)

            # Fit the Normal distribution to the log-transformed data
            lognorm_mean = np.mean(log_transformed_data)
            lognorm_std = np.std(log_transformed_data)

            # Define outliers as points more than 3 standard deviations from the mean
            outlier_threshold = 3
            lower_bound = lognorm_mean - outlier_threshold * lognorm_std
            upper_bound = lognorm_mean + outlier_threshold * lognorm_std

            # Transform the bounds back to the original scale
            lower_bound_exp = np.exp(lower_bound)
            upper_bound_exp = np.exp(upper_bound)

            # Identify outliers in the original data scale
            outliers = filtered_data[(filtered_data < lower_bound_exp) | (
                filtered_data > upper_bound_exp)]

        case "gamma":
            # Identify outliers using the fitted Gamma distribution
            gamma_params = distributions['Gamma'].fit(filtered_data)
            shape = gamma_params[0]
            loc = gamma_params[1]
            scale = gamma_params[2]

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = gamma.ppf(0.95, shape, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "beta":
            # Identify outliers using the fitted Beta distribution
            beta_params = distributions['Beta'].fit(filtered_data)
            a = beta_params[0]
            b = beta_params[1]
            loc = beta_params[2]
            scale = beta_params[3]

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = beta.ppf(0.95, a, b, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "weibull_min":
            # Identify outliers using the fitted Weibull Min distribution
            weibull_min_params = distributions['Weibull_Min'].fit(
                filtered_data)
            c = weibull_min_params[0]
            loc = weibull_min_params[1]
            scale = weibull_min_params[2]

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = weibull_min.ppf(0.95, c, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "chi2":
            # Identify outliers using the fitted Chi-Squared distribution
            chi2_params = distributions['Chi-Squared'].fit(filtered_data)
            if len(chi2_params) == 3:
                dfr = chi2_params[0]
                loc = chi2_params[1]
                scale = chi2_params[2]
            else:
                dfr = chi2_params[0]
                loc = 0
                scale = 1

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = chi2.ppf(0.95, dfr, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "pareto":
            # Identify outliers using the fitted Pareto distribution
            pareto_params = distributions['Pareto'].fit(filtered_data)
            b = pareto_params[0]
            loc = pareto_params[1]
            scale = pareto_params[2]

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = pareto.ppf(0.95, b, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "uniform":
            # Identify outliers using the fitted Uniform distribution
            uniform_params = distributions['Uniform'].fit(filtered_data)
            loc = uniform_params[0]
            scale = uniform_params[1]

            # For a uniform distribution, every point within the range [loc, loc+scale] is considered non-outlier.
            # We'll define outliers as points outside this range.

            # Calculate lower and upper bounds of the uniform distribution
            lower_bound = loc
            upper_bound = loc + scale

            # Identify outliers in the original data scale
            outliers = filtered_data[(filtered_data < lower_bound) | (
                filtered_data > upper_bound)]

        case "t":
            # Identify outliers using the fitted T-Distribution
            t_params = distributions['T-Distribution'].fit(filtered_data)
            dfr = t_params[0]  # Degrees of freedom
            # Location parameter (mean)
            loc = t_params[1] if len(t_params) > 1 else 0
            # Scale parameter (standard deviation)
            scale = t_params[2] if len(t_params) > 2 else 1

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = t.ppf(0.95, dfr, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "gumbel_r":
            # Identify outliers using the fitted Gumbel_R distribution
            gumbel_r_params = distributions['Gumbel_R'].fit(filtered_data)
            loc = gumbel_r_params[0]  # Location parameter
            scale = gumbel_r_params[1]  # Scale parameter

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = gumbel_r.ppf(0.95, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "triang":
            # Identify outliers using the fitted Triangular distribution
            triang_params = distributions['Triangular'].fit(filtered_data)
            c = triang_params[0]  # Lower bound
            loc = triang_params[1]  # Mode
            scale = triang_params[2] - c  # Width

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = triang.ppf(0.95, c, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "laplace":
            # Identify outliers using the fitted Laplace distribution
            laplace_params = distributions['Laplace'].fit(filtered_data)
            loc = laplace_params[0]  # Location parameter
            scale = laplace_params[1]  # Scale parameter

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = laplace.ppf(0.95, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "logistic":
            # Identify outliers using the fitted Logistic distribution
            logistic_params = distributions['Logistic'].fit(filtered_data)
            loc = logistic_params[0]  # Location parameter
            scale = logistic_params[1]  # Scale parameter

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = logistic.ppf(0.95, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "burr":
            # Identify outliers using the fitted Burr distribution
            burr_params = distributions['Burr'].fit(filtered_data)
            alpha = burr_params[0]  # shape parameter
            betaa = burr_params[1]   # shape parameter
            loc = burr_params[2]    # location parameter
            scale = burr_params[3]  # scale parameter

            # Define outliers as points beyond the 95th percentile (upper bound)
            upper_bound = burr.ppf(0.95, alpha, betaa, loc=loc, scale=scale)

            # Identify outliers in the original data scale
            outliers = filtered_data[filtered_data > upper_bound]

        case "IQR":

            # Identify outliers using IQR (Interquartile range) method
            Q1 = filtered_data.quantile(0.25)
            Q3 = filtered_data.quantile(0.75)
            IQR = Q3 - Q1

            # Identify outliers based on 1.5 IQR rule
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)

            # Identify outliers in the filtered DataFrame using original indices
            if not just_upperbound:
                outliers = filtered_data[(filtered_data < lower_bound) | (
                    filtered_data > upper_bound)]
            else:
                outliers = filtered_data[(filtered_data > upper_bound)]
        case _:
            print("Unknown Distribution")

    outliers_df = close_prices.loc[outliers.index]

    # Print outliers information
    if show_outliers:

        print("Number of outliers:", len(outliers_df))

        if len(outliers_df) > 0:
            print("Outliers:")
            # Return the entire DataFrame containing outliers
            print(outliers_df)

    if save_to_file:
        save_dataframe_to_file(
            outliers_df, "C:/Stock Price lists/" + stock_name + "_outliers.csv", overwrite=True, index=True)

    return outliers_df  # Return the DataFrame containing all outliers


# perform adfuller test
def perform_adfuller(prices):
    result = adfuller(prices)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    return result


def perform_differencing(prices):
    prices = prices.diff().dropna()
    return prices


def Perform_Ljung_Box_test(prices, lags=252, return_df=True):

    # Perform Ljung-Box test (replace 'lags' with desired number of lags to test)
    # Test for seasonality (252 lags for yearly seasonality in daily data)
    lb_stats = acorr_ljungbox(prices, lags, return_df)

    # Print test statistic and p-value for each lag
    print("Ljung-Box Test Statistics and p-values:")
    print(lb_stats)

    save_dataframe_to_file(
        lb_stats, "C:/Stock Price lists/" + stock_name + "_lb_stats.csv", overwrite=True, index=False)


def Calculate_correlogram_acf_pacf(prices, nlags=252):
    # Calculate correlogram
    # Calculate lags up to 252 days (one trading year)
    acf_values = acf(prices, nlags)
    pacf_values = pacf(prices, nlags)

    # Plot correlogram
    # plt.figure(figsize=(12, 6))
    # plt.subplot(211)
    # plt.stem(range(len(acf_values)), acf_values)
    # plt.title('Autocorrelation Function')

    # plt.subplot(212)
    # plt.stem(range(len(pacf_values)), pacf_values)
    # plt.title('Partial Autocorrelation Function')

    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(acf(prices, nlags=252))
    plt.title('Autocorrelation')
    plt.subplot(122)
    plt.plot(pacf(prices, nlags=252))
    plt.title('Partial Autocorrelation')
    plt.show()


def draw_Original_Trend(prices, chart_type_title):
    # Plot the original series
    plt.figure(figsize=(12, 6))
    plt.plot(prices)
    plt.title(chart_type_title + ' ' + stock_name + ' Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.show()


def fit_sma(prices, window_size, show_plot=False, print_result=False, save_to_file=False):
    """
    This function calculates the simple moving average (SMA) of a time series.

    Args:
        data: A list of numerical values representing the time series.
        window_size: The number of data points to include in the moving average window.

    Returns:
        A list of the same length as the original data, containing the SMA values for each point.
    """
    # Calculate the window_size-day (for example 100-day) moving average
    sma = prices.rolling(window=window_size).mean()

    if show_plot:
        # Plot the original prices and the moving average
        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Original Prices')
        plt.plot(sma, label=str(window_size) +
                 '-day Moving Average', color='orange')
        plt.title(stock_name + ' Stock Prices with ' +
                  str(window_size) + '-day Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.show()
    if print_result:
        print(
            f"Simple Moving Average (window size {window_size}): {moving_average}")

    if save_to_file:
        save_dataframe_to_file(
            sma, "C:/Stock Price lists/" + stock_name + "_sma.csv", overwrite=True, index=True)

    return sma


def evaluate_vs_baseline_sma(prices, stock_name, window_size, test_size=0.2):
    # Ensure the prices data has a DateTime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("The prices series must be indexed by dates")

    # Split the data into training and test sets
    split_index = int(len(prices) * (1 - test_size))
    train, test = prices[:split_index], prices[split_index:]
    window_size = window_size
    sma = fit_sma(train, window_size, show_plot=False,
                  print_result=False, save_to_file=False)

    # Use the last SMA value from the training period to forecast the test period
    last_sma_value = sma.iloc[-1]
    sma_baseline = pd.Series(last_sma_value, index=test.index)

    # Calculate MSE, RMSE, MAE, MAPE, and R² for the SMA baseline
    mse = mean_squared_error(test, sma_baseline)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, sma_baseline)
    mape = mean_absolute_percentage_error(test, sma_baseline)
    r2 = r2_score(test, sma_baseline)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}%")
    print(f"R-squared: {r2}")

    # Plot the actual prices and the SMA baseline
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Prices')
    plt.plot(test.index, sma_baseline, label='SMA Baseline', linestyle='--')

    plt.title(f'{stock_name} ' +
              str(window_size) + '-day SMA Baseline')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def fit_ema(prices, window_size, show_plot=False, print_result=False, save_to_file=False):
    """
    This function calculates the exponential moving average (EMA) of a time series.

    Args:
        prices: A pandas Series representing the time series.
        window_size: The number of data points to include in the EMA window.

    Returns:
        A pandas Series containing the EMA values for each point.
    """
    # Calculate the window_size-day (for example 100-day) exponential moving average
    ema = prices.ewm(span=window_size, adjust=False).mean()

    if show_plot:
        # Plot the original prices and the moving average
        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Original Prices')
        plt.plot(
            ema, label=f'{window_size}-day Exponential Moving Average', color='orange')
        plt.title(f'{stock_name} ' +
                  str(window_size) + '-day Exponential Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.show()

    if print_result:
        print(f"Exponential Moving Average (window size {window_size}): {ema}")

    if save_to_file:
        ema.to_csv(
            f"C:/Stock Price lists/{stock_name}_ema.csv", index=True)

    return ema


def evaluate_vs_baseline_ema(prices, stock_name, window_size, test_size=0.2):
    # Ensure the prices data has a DateTime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("The prices series must be indexed by dates")

    # Split the data into training and test sets
    split_index = int(len(prices) * (1 - test_size))
    train, test = prices[:split_index], prices[split_index:]
    ema = fit_ema(train, window_size, show_plot=False,
                  print_result=False, save_to_file=False)

    # Use the last EMA value from the training period to forecast the test period
    last_ema_value = ema.iloc[-1]
    ema_baseline = pd.Series(last_ema_value, index=test.index)

    # Calculate MSE and RMSE for the EMA baseline
    mse = mean_squared_error(test, ema_baseline)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, ema_baseline)
    mape = mean_absolute_percentage_error(test, ema_baseline)
    r2 = r2_score(test, ema_baseline)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}%")
    print(f"R-squared: {r2}")

    # Plot the actual prices and the EMA baseline
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Prices')
    plt.plot(test.index, ema_baseline, label='EMA Baseline', linestyle='--')

    plt.title(f'{stock_name} {window_size}-day EMA Baseline')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


class ExponentialSmoothingTimeSeriesPredictor:
    def __init__(self, prices, stock_name, test_size=0.2, seasonal_periods=None, confidence_level=0.95):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("The prices series must be indexed by dates")

        self.prices = prices
        self.stock_name = stock_name
        self.test_size = test_size
        self.seasonal_periods = seasonal_periods
        self.confidence_level = confidence_level

        self.split_index = int(len(prices) * (1 - test_size))
        self.train = prices[:self.split_index]
        self.test = prices[self.split_index:]
        self.model_fit = None

    def fit_model(self):
        try:
            model = ExponentialSmoothing(
                self.train, trend='add', seasonal=None, seasonal_periods=self.seasonal_periods)
            self.model_fit = model.fit()
            print(self.model_fit.summary())
        except Exception as e:
            print("Error in fitting Exponential Smoothing model:", e)

    def forecast(self, steps):
        if self.model_fit is None:
            raise ValueError(
                "Model is not fitted yet. Call fit_model() first.")

        try:
            forecast_values = self.model_fit.forecast(steps)
        except Exception as e:
            print("Error in forecasting:", e)
            return None

        return forecast_values

    def save_dataframe_to_file(self, df, filename, overwrite=True, index=False):
        if overwrite or not os.path.exists(filename):
            df.to_csv(filename, index=index)
        else:
            print(
                f"File {filename} already exists. Set overwrite=True to overwrite it.")

    def plot_forecast(self, forecast_series, lower_conf_int, upper_conf_int):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train, label='Train')
        plt.plot(self.test, label='Test')
        plt.plot(forecast_series, label='Forecast', color='red')
        plt.fill_between(self.test.index, lower_conf_int,
                         upper_conf_int, color='pink', alpha=0.3)
        plt.title(
            f'{self.stock_name} Stock Prices - Train, Test and Forecast (Exponential Smoothing)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.show()

    def predict_and_evaluate(self):
        steps = len(self.test)
        forecast_values = self.forecast(steps)

        if forecast_values is None:
            return

        forecast_series = pd.Series(
            forecast_values.values, index=self.test.index)

        residuals = self.train - self.model_fit.fittedvalues
        residual_std = residuals.std()
        z = 1.96  # For a 95% confidence interval
        lower_conf_int = forecast_values - z * residual_std
        upper_conf_int = forecast_values + z * residual_std

        # Debugging prints to check the values
        print("Forecast values:\n", forecast_values)
        print("Lower confidence interval:\n", lower_conf_int)
        print("Upper confidence interval:\n", upper_conf_int)

        forecast_data = pd.DataFrame({
            'Forecast': forecast_values,
            'Lower_CI': lower_conf_int,
            'Upper_CI': upper_conf_int
        }, index=self.test.index)

        # Ensure the forecast_data does not contain NaNs
        print("Forecast data before saving:\n", forecast_data)

        self.save_dataframe_to_file(
            forecast_data, f"C:/Stock Price lists/{self.stock_name}_Exponential_Smoothing_Predicted.csv", overwrite=True, index=True)

        # Align the indices and remove NaN values from both series
        aligned_test, aligned_forecast = self.test.align(
            forecast_series, join='inner')
        aligned_test = aligned_test.dropna()
        aligned_forecast = aligned_forecast.dropna()

        mse = mean_squared_error(aligned_test, aligned_forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(aligned_test, aligned_forecast)
        mape = mean_absolute_percentage_error(aligned_test, aligned_forecast)
        r2 = r2_score(aligned_test, aligned_forecast)

        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Absolute Percentage Error: {mape}%")
        print(f"R-squared: {r2}")

        self.plot_forecast(forecast_series, lower_conf_int, upper_conf_int)

        return forecast_data

    def compare_with_real_data(self, real_data):
        if not isinstance(real_data.index, pd.DatetimeIndex):
            raise ValueError("The real_data series must be indexed by dates")

        if self.model_fit is None:
            raise ValueError(
                "Model is not fitted yet. Call fit_model() first.")

        steps = len(real_data)
        forecast_values = self.forecast(steps)
        if forecast_values is None:
            return

        forecast_series = pd.Series(
            forecast_values.values, index=real_data.index)
        residuals = real_data - forecast_series

        # Align the indices and remove NaN values from both series
        aligned_real, aligned_forecast = real_data.align(
            forecast_series, join='inner')
        aligned_real = aligned_real.dropna()
        aligned_forecast = aligned_forecast.dropna()

        mse = mean_squared_error(aligned_real, aligned_forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(aligned_real, aligned_forecast)
        mape = mean_absolute_percentage_error(aligned_real, aligned_forecast)
        r2 = r2_score(aligned_real, aligned_forecast)

        print(f"Mean Squared Error against real data: {mse}")
        print(f"Root Mean Squared Error against real data: {rmse}")
        print(f"Mean Absolute Error against real data: {mae}")
        print(f"Mean Absolute Percentage Error: {mape}%")
        print(f"R-squared: {r2}")

        plt.figure(figsize=(12, 6))
        plt.plot(real_data, label='Real Data')
        plt.plot(forecast_series, label='Forecast', color='red')
        plt.title(f'Comparison of Real Data and Forecast (Exponential Smoothing)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.show()


def fit_exponential_Smoothing_forecaster_class(prices, stock_name, forecasting_Duration, test_size=0.2, seasonal_periods=None, confidence_level=0.95, predict_future_value=False):

    # Initialize the predictor
    predictor = ExponentialSmoothingTimeSeriesPredictor(
        prices, stock_name, test_size, seasonal_periods, confidence_level)

    # Fit the model
    predictor.fit_model()

    # Predict and evaluate using the length of the test set
    forecast_data = predictor.predict_and_evaluate()

    if predict_future_value:
        # Predict for a fixed future duration (e.g., 90 days)
        forecast_duration = predictor.forecast(steps=forecasting_Duration)
        if forecast_duration is not None:
            forecast_index_n_days = pd.date_range(
                start=prices.index[-1], periods=forecasting_Duration + 1, freq='D')[1:]
            forecast_series_n_days = pd.Series(
                forecast_duration.values, index=forecast_index_n_days)

        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Original Data')
        plt.plot(forecast_series_n_days, label=str(
            forecasting_Duration) + ' Days Forecast', color='red')
        plt.title(
            f'{stock_name} Stock Prices - ' + str(forecasting_Duration) + ' Days Forecast (Exponential Smoothing)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.show()


def fit_ARIMA(prices, Arima_order, test_size=0.2):
    # Fit ARIMA model to original series (if trend is detected)

    # Ensure the prices data has a DateTime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("The prices series must be indexed by dates")

    # Split the data into training and test sets
    split_index = int(len(prices) * (1 - test_size))
    train, test = prices[:split_index], prices[split_index:]

    # Adjust the order based on ACF/PACF analysis
    # Fit ARIMA model to original series
    try:
        model = ARIMA(train, order=Arima_order)
        model_fit = model.fit()
        print(model_fit.summary())
    except Exception as e:
        print("Error in fitting ARIMA model:", e)
        return

    # Forecast future values
    steps = len(test)
    try:
        forecast = model_fit.get_forecast(steps=steps)
        forecast_values = forecast.predicted_mean
        forecast_conf_int = forecast.conf_int()
    except Exception as e:
        print("Error in forecasting:", e)
        return

    # Create a new index for the forecasted values
    # forecast_index = pd.date_range(start=prices.index[-1], periods=steps+1, freq='B')[1:]
    forecast_series = pd.Series(forecast_values.values, index=test.index)

    # Combine forecasted values and confidence intervals into a single DataFrame
    Complete_forecast_data = pd.DataFrame({
        'Date': test.index,
        'Forecast': forecast_values,
        # Lower bound of confidence interval
        'Lower_CI': forecast_conf_int.iloc[:, 0],
        # Upper bound of confidence interval
        'Upper_CI': forecast_conf_int.iloc[:, 1]
    })
    save_dataframe_to_file(Complete_forecast_data, "C:/Stock Price lists/" +
                           stock_name + "_ARIMA_Predicted.csv", overwrite=True, index=True)

    # Debugging statements to check the forecast series
    print("Train data:")
    print(train.tail())
    print("test data:")
    print(test.head)
    print("Forecast data:")
    print(forecast_series)
    print("Forecast confidence intervals:")
    print(forecast_conf_int)

    # Calculate and print the mean squared error, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)
    mse = mean_squared_error(test, forecast_series)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, forecast_series)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")

    # Plot original, train, test, and forecasted series
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(forecast_series, label='Forecast', color='red')
    plt.fill_between(test.index,
                     forecast_conf_int.iloc[:, 0],
                     forecast_conf_int.iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.title(
        stock_name + ' Stock Prices - Train, Test and Forecast (ARIMA' + str(Arima_order) + ')')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.show()

# Use auto_arima to find the best ARIMA order


def automatic_arima(prices, max_d=3):
    stepwise_fit = auto_arima(prices, start_p=1, start_q=1, max_p=5, max_q=5, max_d=max_d, seasonal=False,
                              trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    print(stepwise_fit.summary())

    # Use the best found order for ARIMA
    best_arima_order = stepwise_fit.order
    print(best_arima_order)


def fit_ARIMA_GARCH(prices, Arima_order, Garch_order=(1, 1), test_size=0.2):
    # Ensure the prices data has a DateTime index and set frequency
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("The prices series must be indexed by dates")

    # Assuming business day frequency, adjust as needed
    prices = prices.asfreq('B')
    prices = prices.ffill()  # Fill missing values if any

    # Split the data into training and test sets
    split_index = int(len(prices) * (1 - test_size))
    train, test = prices[:split_index], prices[split_index:]

    # Fit ARIMA model to original series
    try:
        arima_model = ARIMA(train, order=Arima_order)
        arima_model_fit = arima_model.fit()
        print(arima_model_fit.summary())
    except Exception as e:
        print("Error in fitting ARIMA model:", e)
        return

    # Extract residuals from ARIMA model
    residuals = arima_model_fit.resid

    # Ensure no NaN or infinite values in residuals
    residuals = residuals.replace([np.inf, -np.inf], np.nan).dropna()

    # Fit GARCH model to the residuals
    try:
        garch_model = arch_model(
            residuals, vol='Garch', p=Garch_order[0], q=Garch_order[1])
        garch_model_fit = garch_model.fit(disp="off")
        print(garch_model_fit.summary())
    except Exception as e:
        print("Error in fitting GARCH model:", e)
        return

    # Forecast future values of ARIMA
    steps = len(test)
    try:
        forecast_ARIMA = arima_model_fit.get_forecast(steps=steps)
        forecast_mean_ARIMA = forecast_ARIMA.predicted_mean
        forecast_conf_int_ARIMA = forecast_ARIMA.conf_int()
    except Exception as e:
        print("Error in forecasting:", e)
        return

    # Forecast future values of GARCH
    try:
        forecast_GARCH = garch_model_fit.forecast(horizon=steps)
        forecast_mean_GARCH = forecast_GARCH.mean.iloc[-steps:].values.flatten(
        )
        forecast_variance_GARCH = forecast_GARCH.variance.iloc[-steps:].values.flatten(
        )
    except Exception as e:
        print("Error in forecasting with GARCH model:", e)
        return

    # Ensure forecast has produced the expected results
    if len(forecast_mean_GARCH) == 0 or len(forecast_variance_GARCH) == 0:
        print("Forecast mean or variance is empty.")
        return

    # Ensure the lengths of forecasted series match the length of the test data
    min_length = min(len(forecast_mean_ARIMA),
                     len(forecast_mean_GARCH), len(test))
    forecast_mean_ARIMA = forecast_mean_ARIMA[:min_length]
    forecast_mean_GARCH = forecast_mean_GARCH[:min_length]
    forecast_conf_int_ARIMA = forecast_conf_int_ARIMA[:min_length]
    test = test[:min_length]

    # Create combined forecast series
    combined_forecast_mean = forecast_mean_ARIMA + forecast_mean_GARCH
    combined_forecast_lower_CI = forecast_conf_int_ARIMA.iloc[:, 0] + (
        forecast_mean_GARCH - 1.96 * np.sqrt(forecast_variance_GARCH))
    combined_forecast_upper_CI = forecast_conf_int_ARIMA.iloc[:, 1] + (
        forecast_mean_GARCH + 1.96 * np.sqrt(forecast_variance_GARCH))

    # Ensure the combined forecast and test series have the same index
    combined_forecast_mean.index = test.index
    combined_forecast_lower_CI.index = test.index
    combined_forecast_upper_CI.index = test.index

    # Combine forecasted values and confidence intervals into a single DataFrame
    ARIMA_GARCH_Model_Prediction = pd.DataFrame({
        'Date': test.index,
        'Forecast': combined_forecast_mean,
        'Lower_CI': combined_forecast_lower_CI,
        'Upper_CI': combined_forecast_upper_CI
    })

    # Save forecast data to a CSV file
    ARIMA_GARCH_Model_Prediction.to_csv(
        "C:/Stock Price lists/" + stock_name + "_ARIMA_GARCH_Predicted.csv", index=False)

    # Calculate and print the mean squared error, Root Mean Squared Error (RMSE),Mean Absolute Error (MAE)
    mse = mean_squared_error(test, combined_forecast_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, combined_forecast_mean)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")

    # Plot original, train, test, and forecasted series
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(combined_forecast_mean.index, combined_forecast_mean,
             label='Forecast', color='red')
    plt.fill_between(combined_forecast_mean.index,
                     combined_forecast_lower_CI, combined_forecast_upper_CI, color='gray', alpha=0.3)
    plt.title('Stock Prices - Train, Test and Forecast (ARIMA' +
              str(Arima_order) + '-GARCH' + str(Garch_order) + ')')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.show()


class ARIMAGARCHForecaster:
    def __init__(self, arima_order, garch_order=(1, 1), test_size=0.2):
        self.arima_order = arima_order
        self.garch_order = garch_order
        self.test_size = test_size
        self.arima_model_fit = None
        self.garch_model_fit = None

    def fit(self, prices):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("The prices series must be indexed by dates")

        prices = prices.asfreq('B')
        prices = prices.ffill()

        split_index = int(len(prices) * (1 - self.test_size))
        train, test = prices[:split_index], prices[split_index:]

        try:
            arima_model = ARIMA(train, order=self.arima_order)
            self.arima_model_fit = arima_model.fit()
            print(self.arima_model_fit.summary())
        except Exception as e:
            print("Error in fitting ARIMA model:", e)
            return

        residuals = self.arima_model_fit.resid
        residuals = residuals.replace([np.inf, -np.inf], np.nan).dropna()

        try:
            garch_model = arch_model(
                residuals, vol='Garch', p=self.garch_order[0], q=self.garch_order[1])
            self.garch_model_fit = garch_model.fit(disp="off")
            print(self.garch_model_fit.summary())
        except Exception as e:
            print("Error in fitting GARCH model:", e)
            return

        steps = len(test)
        try:
            forecast_ARIMA = self.arima_model_fit.get_forecast(steps=steps)
            forecast_mean_ARIMA = forecast_ARIMA.predicted_mean
            forecast_conf_int_ARIMA = forecast_ARIMA.conf_int()
        except Exception as e:
            print("Error in forecasting:", e)
            return

        try:
            forecast_GARCH = self.garch_model_fit.forecast(horizon=steps)
            forecast_mean_GARCH = forecast_GARCH.mean.iloc[-steps:].values.flatten(
            )
            forecast_variance_GARCH = forecast_GARCH.variance.iloc[-steps:].values.flatten(
            )
        except Exception as e:
            print("Error in forecasting with GARCH model:", e)
            return

        if len(forecast_mean_GARCH) == 0 or len(forecast_variance_GARCH) == 0:
            print("Forecast mean or variance is empty.")
            return

        min_length = min(len(forecast_mean_ARIMA),
                         len(forecast_mean_GARCH), len(test))
        forecast_mean_ARIMA = forecast_mean_ARIMA[:min_length]
        forecast_mean_GARCH = forecast_mean_GARCH[:min_length]
        forecast_conf_int_ARIMA = forecast_conf_int_ARIMA[:min_length]
        test = test[:min_length]

        combined_forecast_mean = forecast_mean_ARIMA + forecast_mean_GARCH
        combined_forecast_lower_CI = forecast_conf_int_ARIMA.iloc[:, 0] + (
            forecast_mean_GARCH - 1.96 * np.sqrt(forecast_variance_GARCH))
        combined_forecast_upper_CI = forecast_conf_int_ARIMA.iloc[:, 1] + (
            forecast_mean_GARCH + 1.96 * np.sqrt(forecast_variance_GARCH))

        combined_forecast_mean.index = test.index
        combined_forecast_lower_CI.index = test.index
        combined_forecast_upper_CI.index = test.index

        ARIMA_GARCH_Model_Prediction = pd.DataFrame({
            'Date': test.index,
            'Forecast': combined_forecast_mean,
            'Lower_CI': combined_forecast_lower_CI,
            'Upper_CI': combined_forecast_upper_CI
        })

        ARIMA_GARCH_Model_Prediction.to_csv(
            "C:/Stock Price lists/" + stock_name + "_ARIMA_GARCH_Predicted.csv", index=False)

        mse = mean_squared_error(test, combined_forecast_mean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, combined_forecast_mean)
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Absolute Error: {mae}")

        plt.figure(figsize=(12, 6))
        plt.plot(train, label='Train')
        plt.plot(test, label='Test')
        plt.plot(combined_forecast_mean.index, combined_forecast_mean,
                 label='Forecast', color='red')
        plt.fill_between(combined_forecast_mean.index, combined_forecast_lower_CI,
                         combined_forecast_upper_CI, color='gray', alpha=0.3)
        plt.title(stock_name + ' Stock Prices - Train, Test and Forecast (ARIMA' +
                  str(self.arima_order) + '-GARCH' + str(self.garch_order) + ')')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.show()

        return ARIMA_GARCH_Model_Prediction

    def forecast_next_days(self, prices, n_days):
        if self.arima_model_fit is None or self.garch_model_fit is None:
            raise ValueError("Model must be fitted before forecasting.")

        # last_date = prices.index[-1]
        # date_range = pd.bdate_range(
        #     start=last_date, periods=n_days + 1, freq='B')
        # forecast_dates = date_range[1:]

        last_date = prices.index[-1]
        forecast_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=n_days)

        forecast_ARIMA = self.arima_model_fit.get_forecast(steps=n_days)
        forecast_mean_ARIMA = forecast_ARIMA.predicted_mean
        forecast_conf_int_ARIMA = forecast_ARIMA.conf_int()

        forecast_GARCH = self.garch_model_fit.forecast(horizon=n_days)
        forecast_mean_GARCH = forecast_GARCH.mean.iloc[-n_days:].values.flatten(
        )
        forecast_variance_GARCH = forecast_GARCH.variance.iloc[-n_days:].values.flatten(
        )

        combined_forecast_mean = forecast_mean_ARIMA + forecast_mean_GARCH
        combined_forecast_lower_CI = forecast_conf_int_ARIMA.iloc[:, 0] + (
            forecast_mean_GARCH - 1.96 * np.sqrt(forecast_variance_GARCH))
        combined_forecast_upper_CI = forecast_conf_int_ARIMA.iloc[:, 1] + (
            forecast_mean_GARCH + 1.96 * np.sqrt(forecast_variance_GARCH))

        combined_forecast_mean.index = forecast_dates
        combined_forecast_lower_CI.index = forecast_dates
        combined_forecast_upper_CI.index = forecast_dates

        forecast_df = pd.DataFrame({
            'Forecast': combined_forecast_mean,
            'Lower_CI': combined_forecast_lower_CI,
            'Upper_CI': combined_forecast_upper_CI
        })

        forecast_df.to_csv(
            "C:/Stock Price lists/" + stock_name + "_ARIMA_GARCH_Forecast.csv")

        return forecast_df

    def evaluate_forecast(self, actual, forecast, prices):
        forecast_aligned, actual_aligned = forecast.align(
            actual, join='inner', axis=0)

        forecast_rmse = np.sqrt(mean_squared_error(
            actual_aligned, forecast_aligned['Forecast']))
        forecast_mae = mean_absolute_error(
            actual_aligned, forecast_aligned['Forecast'])

        print(f"Forecast RMSE: {forecast_rmse}")
        print(f"Forecast MAE: {forecast_mae}")

        plt.figure(figsize=(12, 6))
        plt.plot(prices.index, prices, label='Original Series')
        plt.plot(forecast.index,
                 forecast['Forecast'], label='Forecast', color='red')
        plt.plot(actual.index, actual, label='Actual', color='orange')
        plt.fill_between(
            forecast.index, forecast['Lower_CI'], forecast['Upper_CI'], color='gray', alpha=0.3)
        plt.title(stock_name + ' Stock Prices - Actual vs Forecast (ARIMA-GARCH)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


# Tuned RNN time series forecaster
class RNNTimeSeriesForecaster:
    def __init__(self, look_back=1, units=50, learning_rate=0.001, epochs=50, batch_size=1, dropout=0, return_sequences=False):
        self.look_back = look_back
        self.units = units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, data):
        X, Y = [], []
        for i in range(len(data) - self.look_back - 1):
            X.append(data[i:(i + self.look_back), 0])
            Y.append(data[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def build_model(self):
        model = Sequential()
        model.add(SimpleRNN(self.units, input_shape=(
            self.look_back, 1), return_sequences=self.return_sequences))
        if self.dropout > 0:
            model.add(Dropout(self.dropout))
        if self.return_sequences:
            model.add(SimpleRNN(self.units))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate), loss='mean_squared_error')
        self.model = model

    def fit(self, prices):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("The prices series must be indexed by dates")

        prices_scaled = self.scaler.fit_transform(prices.values.reshape(-1, 1))

        train_size = int(len(prices_scaled) * 0.8)
        train, test = prices_scaled[:train_size], prices_scaled[train_size:]

        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)

        trainX = np.reshape(trainX, (trainX.shape[0], self.look_back, 1))
        testX = np.reshape(testX, (testX.shape[0], self.look_back, 1))

        self.build_model()
        self.model.fit(trainX, trainY, epochs=self.epochs,
                       batch_size=self.batch_size, verbose=2, validation_data=(testX, testY))

        trainPredict = self.model.predict(trainX)
        testPredict = self.model.predict(testX)

        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY = self.scaler.inverse_transform([trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testY = self.scaler.inverse_transform([testY])

        train_score = mean_squared_error(trainY[0], trainPredict[:, 0])
        test_score = mean_squared_error(testY[0], testPredict[:, 0])

        train_rmse = np.sqrt(train_score)
        test_rmse = np.sqrt(test_score)

        train_mae = mean_absolute_error(trainY[0], trainPredict[:, 0])
        test_mae = mean_absolute_error(testY[0], testPredict[:, 0])

        print(f"Train Mean Squared Error: {train_score}")
        print(f"Test Mean Squared Error: {test_score}")
        print(f"Train Root Mean Squared Error: {train_rmse}")
        print(f"Test Root Mean Squared Error: {test_rmse}")
        print(f"Train Mean Absolute Error: {train_mae}")
        print(f"Test Mean Absolute Error: {test_mae}")

        trainPredictPlot = np.empty_like(prices_scaled)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[self.look_back:len(
            trainPredict) + self.look_back, :] = trainPredict

        testPredictPlot = np.empty_like(prices_scaled)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (self.look_back * 2) +
                        1:len(prices_scaled) - 1, :] = testPredict

        plt.figure(figsize=(12, 6))
        plt.plot(self.scaler.inverse_transform(
            prices_scaled), label='Original Series')
        plt.plot(trainPredictPlot, label='Train Predict', color='green')
        plt.plot(testPredictPlot, label='Test Predict', color='red')
        plt.title(
            stock_name + ' Stock Prices - Original Series and RNN Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def tune_hyperparameters(self, prices, param_dist, n_iter_search=20):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("The prices series must be indexed by dates")

        prices_scaled = self.scaler.fit_transform(prices.values.reshape(-1, 1))

        train_size = int(len(prices_scaled) * 0.8)
        train, test = prices_scaled[:train_size], prices_scaled[train_size:]

        best_params = None
        lowest_val_loss = float('inf')

        param_list = list(ParameterSampler(param_dist, n_iter=n_iter_search))

        for params in param_list:
            self.look_back = params['look_back']
            self.units = params['units']
            self.learning_rate = params['learning_rate']
            self.epochs = params['epochs']
            self.batch_size = params['batch_size']
            self.dropout = params['dropout']
            self.return_sequences = params['return_sequences']

            trainX, trainY = self.create_dataset(train)
            validationX, validationY = self.create_dataset(test)

            trainX = np.reshape(trainX, (trainX.shape[0], self.look_back, 1))
            validationX = np.reshape(
                validationX, (validationX.shape[0], self.look_back, 1))

            self.build_model()
            self.model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size,
                           verbose=0, validation_data=(validationX, validationY))

            val_predict = self.model.predict(validationX)
            val_predict = self.scaler.inverse_transform(val_predict)
            validationY_inverse = self.scaler.inverse_transform([validationY])
            val_loss = mean_squared_error(
                validationY_inverse[0], val_predict[:, 0])

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_params = params

        print(f"Best Hyperparameters: {best_params}")
        print(f"Lowest Validation Loss: {lowest_val_loss}")

        self.look_back = best_params['look_back']
        self.units = best_params['units']
        self.learning_rate = best_params['learning_rate']
        self.epochs = best_params['epochs']
        self.batch_size = best_params['batch_size']
        self.dropout = best_params['dropout']
        self.return_sequences = best_params['return_sequences']


def fit_rnn_forecaster_class(prices):
    # Initialize the forecaster
    forecaster = RNNTimeSeriesForecaster()
    # Fit the model
    forecaster.fit(prices)

    # Tune hyperparameters
    param_dist = {
        'units': randint(20, 100),
        'look_back': randint(1, 10),
        'learning_rate': uniform(0.001, 0.01),
        'epochs': randint(10, 100),
        'batch_size': randint(1, 32),
        'dropout': uniform(0, 0.5),
        'return_sequences': [True, False]
    }
    forecaster.tune_hyperparameters(prices, param_dist, n_iter_search=50)
    # Fit the model with the best hyperparameters found
    forecaster.fit(prices)


class LSTMTimeSeriesForecaster:
    def __init__(self, look_back=1, units=50, learning_rate=0.001, epochs=50, batch_size=1, dropout=0, return_sequences=False):
        self.look_back = look_back
        self.units = units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, data):
        X, Y = [], []
        for i in range(len(data) - self.look_back - 1):
            X.append(data[i:(i + self.look_back), 0])
            Y.append(data[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.units, input_shape=(self.look_back, 1),
                  return_sequences=self.return_sequences))
        if self.dropout > 0:
            model.add(Dropout(self.dropout))
        if self.return_sequences:
            model.add(LSTM(self.units))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate), loss='mean_squared_error')
        self.model = model

    def fit(self, prices):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("The prices series must be indexed by dates")

        prices_scaled = self.scaler.fit_transform(prices.values.reshape(-1, 1))

        train_size = int(len(prices_scaled) * 0.8)
        train, test = prices_scaled[:train_size], prices_scaled[train_size:]

        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)

        trainX = np.reshape(trainX, (trainX.shape[0], self.look_back, 1))
        testX = np.reshape(testX, (testX.shape[0], self.look_back, 1))

        self.build_model()
        self.model.fit(trainX, trainY, epochs=self.epochs,
                       batch_size=self.batch_size, verbose=2, validation_data=(testX, testY))

        trainPredict = self.model.predict(trainX)
        testPredict = self.model.predict(testX)

        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY = self.scaler.inverse_transform([trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testY = self.scaler.inverse_transform([testY])

        train_score = mean_squared_error(trainY[0], trainPredict[:, 0])
        test_score = mean_squared_error(testY[0], testPredict[:, 0])

        train_rmse = np.sqrt(train_score)
        test_rmse = np.sqrt(test_score)

        train_mae = mean_absolute_error(trainY[0], trainPredict[:, 0])
        test_mae = mean_absolute_error(testY[0], testPredict[:, 0])

        print(f"Train Mean Squared Error: {train_score}")
        print(f"Test Mean Squared Error: {test_score}")
        print(f"Train Root Mean Squared Error: {train_rmse}")
        print(f"Test Root Mean Squared Error: {test_rmse}")
        print(f"Train Mean Absolute Error: {train_mae}")
        print(f"Test Mean Absolute Error: {test_mae}")

        trainPredictPlot = np.empty_like(prices_scaled)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[self.look_back:len(
            trainPredict) + self.look_back, :] = trainPredict

        testPredictPlot = np.empty_like(prices_scaled)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (self.look_back * 2) +
                        1:len(prices_scaled) - 1, :] = testPredict

        plt.figure(figsize=(12, 6))
        plt.plot(self.scaler.inverse_transform(
            prices_scaled), label='Original Series')
        plt.plot(trainPredictPlot, label='Train Predict', color='green')
        plt.plot(testPredictPlot, label='Test Predict', color='red')
        plt.title(
            stock_name + ' Stock Prices - Original Series and LSTM Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def tune_hyperparameters(self, prices, param_dist, n_iter_search=20):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("The prices series must be indexed by dates")

        prices_scaled = self.scaler.fit_transform(prices.values.reshape(-1, 1))

        train_size = int(len(prices_scaled) * 0.8)
        train, test = prices_scaled[:train_size], prices_scaled[train_size:]

        best_params = None
        lowest_val_loss = float('inf')

        param_list = list(ParameterSampler(param_dist, n_iter=n_iter_search))

        for params in param_list:
            self.look_back = params['look_back']
            self.units = params['units']
            self.learning_rate = params['learning_rate']
            self.epochs = params['epochs']
            self.batch_size = params['batch_size']
            self.dropout = params['dropout']
            self.return_sequences = params['return_sequences']

            trainX, trainY = self.create_dataset(train)
            validationX, validationY = self.create_dataset(test)

            trainX = np.reshape(trainX, (trainX.shape[0], self.look_back, 1))
            validationX = np.reshape(
                validationX, (validationX.shape[0], self.look_back, 1))

            self.build_model()
            self.model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size,
                           verbose=0, validation_data=(validationX, validationY))

            val_predict = self.model.predict(validationX)
            val_predict = self.scaler.inverse_transform(val_predict)
            validationY_inverse = self.scaler.inverse_transform([validationY])
            val_loss = mean_squared_error(
                validationY_inverse[0], val_predict[:, 0])

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_params = params

        print(f"Best Hyperparameters: {best_params}")
        print(f"Lowest Validation Loss: {lowest_val_loss}")

        self.look_back = best_params['look_back']
        self.units = best_params['units']
        self.learning_rate = best_params['learning_rate']
        self.epochs = best_params['epochs']
        self.batch_size = best_params['batch_size']
        self.dropout = best_params['dropout']
        self.return_sequences = best_params['return_sequences']


def fit_lstm_forecaster_class(prices):

    # Initialize the forecaster
    forecaster = LSTMTimeSeriesForecaster()

    # Fit the model
    forecaster.fit(prices)

    # Tune hyperparameters
    param_dist = {
        'units': randint(20, 100),
        'look_back': randint(1, 10),
        'learning_rate': uniform(0.001, 0.01),
        'epochs': randint(10, 100),
        'batch_size': randint(1, 32),
        'dropout': uniform(0, 0.5),
        'return_sequences': [True, False]
    }
    forecaster.tune_hyperparameters(prices, param_dist, n_iter_search=50)

    # Fit the model with the best hyperparameters found
    forecaster.fit(prices)


class GRUTimeSeriesForecaster:
    def __init__(self, look_back=1, units=50, learning_rate=0.001, epochs=50, batch_size=1, dropout=0, return_sequences=False):
        self.look_back = look_back
        self.units = units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, data):
        X, Y = [], []
        for i in range(len(data) - self.look_back - 1):
            X.append(data[i:(i + self.look_back), 0])
            Y.append(data[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def build_model(self):
        model = Sequential()
        model.add(GRU(self.units, input_shape=(self.look_back, 1),
                  return_sequences=self.return_sequences))
        if self.dropout > 0:
            model.add(Dropout(self.dropout))
        if self.return_sequences:
            model.add(GRU(self.units))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate), loss='mean_squared_error')
        self.model = model

    def fit(self, prices):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("The prices series must be indexed by dates")

        prices_scaled = self.scaler.fit_transform(prices.values.reshape(-1, 1))

        train_size = int(len(prices_scaled) * 0.8)
        train, test = prices_scaled[:train_size], prices_scaled[train_size:]

        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)

        trainX = np.reshape(trainX, (trainX.shape[0], self.look_back, 1))
        testX = np.reshape(testX, (testX.shape[0], self.look_back, 1))

        self.build_model()
        self.model.fit(trainX, trainY, epochs=self.epochs,
                       batch_size=self.batch_size, verbose=2, validation_data=(testX, testY))

        trainPredict = self.model.predict(trainX)
        testPredict = self.model.predict(testX)

        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY = self.scaler.inverse_transform([trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testY = self.scaler.inverse_transform([testY])

        train_score = mean_squared_error(trainY[0], trainPredict[:, 0])
        test_score = mean_squared_error(testY[0], testPredict[:, 0])

        train_rmse = np.sqrt(train_score)
        test_rmse = np.sqrt(test_score)

        train_mae = mean_absolute_error(trainY[0], trainPredict[:, 0])
        test_mae = mean_absolute_error(testY[0], testPredict[:, 0])

        print(f"Train Mean Squared Error: {train_score}")
        print(f"Test Mean Squared Error: {test_score}")
        print(f"Train Root Mean Squared Error: {train_rmse}")
        print(f"Test Root Mean Squared Error: {test_rmse}")
        print(f"Train Mean Absolute Error: {train_mae}")
        print(f"Test Mean Absolute Error: {test_mae}")

        trainPredictPlot = np.empty_like(prices_scaled)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[self.look_back:len(
            trainPredict) + self.look_back, :] = trainPredict

        testPredictPlot = np.empty_like(prices_scaled)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (self.look_back * 2) +
                        1:len(prices_scaled) - 1, :] = testPredict

        plt.figure(figsize=(12, 6))
        plt.plot(self.scaler.inverse_transform(
            prices_scaled), label='Original Series')
        plt.plot(trainPredictPlot, label='Train Predict', color='green')
        plt.plot(testPredictPlot, label='Test Predict', color='red')
        plt.title(
            stock_name + ' Stock Prices - Original Series and GRU Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def tune_hyperparameters(self, prices, param_dist, n_iter_search=20):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("The prices series must be indexed by dates")

        prices_scaled = self.scaler.fit_transform(prices.values.reshape(-1, 1))

        train_size = int(len(prices_scaled) * 0.8)
        train, test = prices_scaled[:train_size], prices_scaled[train_size:]

        best_params = None
        lowest_val_loss = float('inf')

        param_list = list(ParameterSampler(param_dist, n_iter=n_iter_search))

        for params in param_list:
            self.look_back = params['look_back']
            self.units = params['units']
            self.learning_rate = params['learning_rate']
            self.epochs = params['epochs']
            self.batch_size = params['batch_size']
            self.dropout = params['dropout']
            self.return_sequences = params['return_sequences']

            trainX, trainY = self.create_dataset(train)
            validationX, validationY = self.create_dataset(test)

            trainX = np.reshape(trainX, (trainX.shape[0], self.look_back, 1))
            validationX = np.reshape(
                validationX, (validationX.shape[0], self.look_back, 1))

            self.build_model()
            self.model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size,
                           verbose=0, validation_data=(validationX, validationY))

            val_predict = self.model.predict(validationX)
            val_predict = self.scaler.inverse_transform(val_predict)
            validationY_inverse = self.scaler.inverse_transform([validationY])
            val_loss = mean_squared_error(
                validationY_inverse[0], val_predict[:, 0])

            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_params = params

        print(f"Best Hyperparameters: {best_params}")
        print(f"Lowest Validation Loss: {lowest_val_loss}")

        self.look_back = best_params['look_back']
        self.units = best_params['units']
        self.learning_rate = best_params['learning_rate']
        self.epochs = best_params['epochs']
        self.batch_size = best_params['batch_size']
        self.dropout = best_params['dropout']
        self.return_sequences = best_params['return_sequences']

    def set_hyperparameters(self, params):
        self.look_back = params['look_back']
        self.units = params['units']
        self.learning_rate = params['learning_rate']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.dropout = params['dropout']
        self.return_sequences = params['return_sequences']

    def forecast_next_days(self, prices, n_days):
        forecast = []
        last_data = self.scaler.transform(
            prices.values[-self.look_back:].reshape(-1, 1))

        for _ in range(n_days):
            input_data = np.reshape(last_data, (1, self.look_back, 1))
            next_value = self.model.predict(input_data)
            forecast.append(next_value[0, 0])
            last_data = np.append(last_data[1:], next_value)

        forecast = np.array(forecast)
        forecast = self.scaler.inverse_transform(forecast.reshape(-1, 1))

        # Generate date range for forecast
        last_date = prices.index[-1]
        forecast_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=n_days)
        # print(forecast_dates)
        # print(f"Shape of forecast_dates: {forecast_dates.shape}")

        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecast, index=forecast_dates, columns=['Forecast'])

        forecast_df.set_index(forecast_dates, inplace=True)
        forecast_df.index.name = "date"
        # print(forecast_df)

        save_dataframe_to_file(
            forecast_df, "C:/Stock Price lists/" + stock_name + "_GRU_forecast.csv", overwrite=True, index=True)

        return forecast_df

    def evaluate_forecast(self, prices, forecast, actual):
        # Align actual and forecast data by their indices
        forecast_aligned, actual_aligned = forecast.align(
            actual, join='inner', axis=0)

        # Drop NaN values if any
        forecast_aligned = forecast_aligned.dropna()
        actual_aligned = actual_aligned.dropna()

        # Filter out zero values in actual to avoid division by zero in MAPE calculation
        mask = actual_aligned != 0
        actual_filtered = actual_aligned[mask]
        forecast_filtered = forecast_aligned[mask]

        # Compute metrics
        forecast_rmse = np.sqrt(mean_squared_error(
            actual_filtered, forecast_filtered))
        forecast_mae = mean_absolute_error(actual_filtered, forecast_filtered)

        print(f"Forecast RMSE: {forecast_rmse}")
        print(f"Forecast MAE: {forecast_mae}")

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(prices.index, prices, label='Original Series')
        plt.plot(forecast.index, forecast, label='Forecast', color='red')
        plt.plot(actual.index, actual, label='Actual', color='orange')
        plt.title(f'{stock_name} Stock Prices - GRU Model Forecast vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


def fit_gru_forecaster_class(prices, actual_next_prices=None, best_params=None):

    # Initialize the forecaster
    forecaster = GRUTimeSeriesForecaster()

    if best_params:
        forecaster.set_hyperparameters(best_params)
    else:
        # Tune hyperparameters if not provided
        param_dist = {
            'units': randint(20, 100),
            'look_back': randint(1, 10),
            'learning_rate': uniform(0.001, 0.01),
            'epochs': randint(10, 100),
            'batch_size': randint(1, 32),
            'dropout': uniform(0, 0.5),
            'return_sequences': [True, False]
        }
        forecaster.tune_hyperparameters(prices, param_dist, n_iter_search=50)

    # Fit the model with the best hyperparameters found or provided
    forecaster.fit(prices)

    if actual_next_prices is not None:
        # Forecast the next 3 months (approximately 63 business days)
        n_days = 63
        forecast_df = forecaster.forecast_next_days(close_prices, n_days)
        # Evaluate the forecast
        forecaster.evaluate_forecast(prices, forecast_df, actual_next_prices)

# ------------------------------------------------------------------


# Print descriptive analysis of passed variable
# do_descriptive_analysis(close_prices)

# ------------------------------------------------------------------

# Identify the best probability distribution
# best_fit_pd = fit_best_distribution(
#     close_prices, show_best_fit=True, show_plot=True)

# ------------------------------------------------------------------

# detect outliers
# outliers_df = detect_outliers(list(best_fit_pd.keys())[
#     0], close_prices, True, False, True)

# ------------------------------------------------------------------

# check for stationarity without differencing
# Interpret results: A low p-value (<0.05) in the Ljung-Box test suggests seasonality.
# The correlogram can visually show periodic spikes at lags corresponding to the seasonality (e.g., every 252 lags for yearly seasonality in daily data).
# perform_adfuller(close_prices)
# Perform_Ljung_Box_test(close_prices, lags=252)
# Calculate_correlogram_acf_pacf(close_prices, nlags=252)

# ------------------------------------------------------------------

# check for stationarity with differencing
difference_close_prices = perform_differencing(close_prices)
# perform_adfuller(difference_close_prices)
Perform_Ljung_Box_test(difference_close_prices, lags=252)
# Calculate_correlogram_acf_pacf(difference_close_prices, nlags=252)

# ------------------------------------------------------------------


# inspect original trends to identify trends or cyclicality
# fit simple moving average to smooth fluctuation and then plot it with original trends
# draw_Original_Trend(close_prices, '')
# fit_sma(close_prices, window_size=90, show_plot=True,
#         print_result=False, save_to_file=False)

# ------------------------------------------------------------------

# to evaluate all models with simple moving average and exponential moving average as a baseline
# evaluate_vs_baseline_sma(close_prices, stock_name,
#                          window_size=90, test_size=0.2)
# evaluate_vs_baseline_ema(close_prices, stock_name,
#                          window_size=90, test_size=0.2)
# ------------------------------------------------------------------

# fit Exponential Smoothing time series model
# fit_exponential_Smoothing_forecaster_class(
#     close_prices, stock_name, forecasting_Duration=90, test_size=0.2, seasonal_periods=None, confidence_level=0.95, predict_future_value=False)

# ------------------------------------------------------------------

# determine best arima configuration
# fit ARIMA time series model
# automatic_arima(close_prices, max_d=3)
# fit_ARIMA(close_prices, (2, 2, 2), test_size=0.2)

# ------------------------------------------------------------------


# Fit ARIMA-GARCH model
# fit_ARIMA_GARCH(close_prices, Arima_order=(2, 2, 2),
#                 Garch_order=(2, 2), test_size=0.2)


# arima_garch_forecaster = ARIMAGARCHForecaster(
#     arima_order=(2, 2, 2), garch_order=(2, 2))
# arima_garch_forecaster.fit(close_prices)
# forecast_df = arima_garch_forecaster.forecast_next_days(
#     close_prices, n_days=63)
# arima_garch_forecaster.evaluate_forecast(
#     actual_next_prices, forecast_df, close_prices)

# ------------------------------------------------------------------

# fit adjusted RNN time series model
# fit_rnn_forecaster_class(close_prices)

# ------------------------------------------------------------------

# fit adjusted LSTM time series model
# fit_lstm_forecaster_class(close_prices)

# ------------------------------------------------------------------

# Call the function when we don't have tuned parameters and actual next prices
# fit_gru_forecaster_class(close_prices)

# Define the best hyperparameters for Alphabet
# best_params = {
#     'batch_size': 24,
#     'dropout': 0.0010595028648908156,
#     'epochs': 91,
#     'learning_rate': 0.0065272063812476955,
#     'look_back': 4,
#     'return_sequences': False,
#     'units': 34
# }

# Define the best hyperparameters for Microsoft
# best_params = {
#     'batch_size': 16,
#     'dropout': 0.044295524080964044,
#     'epochs': 73,
#     'learning_rate': 0.010301134772215333,
#     'look_back': 5,
#     'return_sequences': False,
#     'units': 40
# }

# Call the function without best hyperparameters to trigger tuning
# fit_gru_forecaster_class(prices, actual_next_prices)

# Call the function when we don't have the actual next prices
# fit_gru_forecaster_class(close_prices, best_params=best_params)


# Call the function with the best hyperparameters
# fit_gru_forecaster_class(close_prices, actual_next_prices, best_params)
