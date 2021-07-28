import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_heat_map(df):
    '''Plot the correlation heatmap of a DataFrame'''

    corrs = df.corr()
    # Set the default matplotlib figure size:
    fig, ax = plt.subplots(figsize=(11, 7))
    # Generate a mask for the upper triangle (taken from seaborn example gallery)
    mask = np.zeros_like(corrs, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Plot the heatmap with seaborn.
    # Assign the matplotlib axis the function returns. This will let us resize the labels.
    ax = sns.heatmap(corrs, mask=mask, annot=True)
    # Resize the labels.
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=12, rotation=90)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=12, rotation=0)
    # If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.
    plt.show()


def plot_learning_curves(train_score, test_score, train_sizes):
    '''plots the learning curves of a model w.r.t list of training sizes'''

    # Plotting the learning curves
    plt.plot(train_sizes, train_score, label= 'Training score')
    plt.plot(train_sizes, test_score, label= 'Test score')
    plt.ylabel('r2 score', fontsize= 14)
    plt.xlabel('Training set size', fontsize= 14)
    plt.title('Learning curves', fontsize= 18, y= 1.03)
    plt.legend()
    plt.show()


def plot_forecast(forecast, train, test, upper=None, lower=None):
    '''plots ARIMA forecast result and confidence int
    usage: forecast, std_err, confidence_int = arima.forecast(len(test), alpha = 0.05)
    '''
    is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
    # Prepare plot series
    fc_series = pd.Series(forecast, index=test.index)
    lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
    upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

    # Plot
    plt.figure(figsize=(10,4), dpi=100)
    plt.plot(train, label='training', color='black')
    plt.plot(test, label='actual', color='black', ls='--')
    plt.plot(fc_series, label='forecast', color='orange')
    if is_confidence_int:
        plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8);
    plt.show();


