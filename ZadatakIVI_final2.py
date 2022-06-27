# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 08:48:18 2022

@author: korisnik
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False

fname = os.path.dirname(os.path.realpath(__file__))

from window import WindowGenerator


def main():

    df_load, df_wday, df_whour = import_data()
    df_cloud, df_wind, df_temp_full = separate_hour(df_whour)
    df_load_08, df_ready = prepare_halfFeature(df_temp_full, df_load)
    df_data = prepared_dataset(df_wday, df_ready, df_load_08)
    plot_fft(df_data)
    num_features, train_df, val_df, test_df = split_normalize_data(df_data)

    OUT_STEPS = 24
    multi_window = WindowGenerator(
        input_width=24, label_width=OUT_STEPS, shift=OUT_STEPS,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["Load"])

    multi_val_performance = {}
    multi_performance = {}

    # RNN model
    multi_lstm_model = tf.keras.Sequential(
        [
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(
                OUT_STEPS * num_features,
                kernel_initializer=tf.initializers.zeros()
            ),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([OUT_STEPS, num_features]),
        ]
    )

    history = compile_and_fit(multi_lstm_model, multi_window)

    multi_val_performance["LSTM"] = multi_lstm_model.evaluate(multi_window.val)
    multi_performance["LSTM"] = multi_lstm_model.evaluate(multi_window.test,
                                                          verbose=0)
    multi_window.plot(multi_lstm_model)

    # Performance
    x = np.arange(len(multi_performance))
    width = 0.3

    metric_name = "mean_absolute_error"
    metric_index = multi_lstm_model.metrics_names.index(metric_name)
    val_mae = [v[metric_index] for v in multi_val_performance.values()]
    test_mae = [v[metric_index] for v in multi_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label="Validation")
    plt.bar(x + 0.17, test_mae, width, label="Test")
    plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
    plt.ylabel("MAE (average over all times and outputs)")
    _ = plt.legend()

    for name, value in multi_performance.items():
        print(f"{name:8s}: {value[1]:0.4f}")

    for name, value in multi_val_performance.items():
        print(f"{name:8s}: {value[1]:0.4f}")


# Function definition
def import_data():
    # global df_load
    df_load = pd.read_csv(fname + "\\EMS\\EMS_Load.csv", delimiter=";",
                          header=0)
    # global df_wday
    df_wday = pd.read_csv(fname + "\\EMS\\EMS_Weather_Daily.csv",
                          delimiter=";", header=0)
    # global df_whour
    df_whour = pd.read_csv(fname + "\\EMS\\EMS_Weather_Hourly.csv",
                           delimiter=";", header=0)

    return df_load, df_wday, df_whour


def separate_hour(df_whour):
    # global df_cloud
    df_cloud = df_whour[df_whour["WeatherType"] == "Cloud"].drop("WeatherType",
                                                                 axis=1)
    # global df_wind
    df_wind = df_whour[df_whour["WeatherType"] == "Wind"].drop("WeatherType",
                                                               axis=1)
    # global df_temp_full
    df_temp_full = df_whour[df_whour["WeatherType"] == "Temperature"].drop(
        "WeatherType", axis=1)

    return df_cloud, df_wind, df_temp_full


def prepare_halfFeature(df_temp_full, df_load):
    df_temp = df_temp_full[df_temp_full["Timestamp"] >
                           "2013-04-14 23:00:00.000"]

    df_load_idx = df_load.set_index("Timestamp")
    df_load_idx.index = pd.to_datetime(df_load_idx.index)

    df_temp_idx = df_temp.set_index("Timestamp")
    df_temp_idx.index = pd.to_datetime(df_temp_idx.index)

    difference_load = pd.date_range(start="2013-04-15",
                                    end="2018-06-30").difference(
                                        df_load_idx.index)
    difference_temp = pd.date_range(start="2013-04-15",
                                    end="2018-06-30").difference(
                                        df_temp_idx.index)

    # Separate data for day 08.01.2018
    # global df_load_08
    df_load_08 = df_load.loc["41496":"41519"]

    # missing day 09.01.2018, all 12 samples get last value from 08.01
    df_resample_load = df_load_idx.resample("H").ffill()
    # add one missing sample with value from previous sample
    df_resample_temp = df_temp_idx.resample("H").ffill()

    # global df_ready
    df_ready = pd.concat([df_resample_load, df_resample_temp], axis=1).rename(
        {"WeatherValue": "Temp"}, axis=1
    )

    return df_load_08, df_ready 


# prepare data from file fajla 'Weather Daily'
def prepared_dataset(df_wday, df_ready, df_load_08):
    df_avgTemp_full = df_wday[df_wday["WeatherType"] ==
                              "Average temperature"].drop("WeatherType",
                                                          axis=1)
    df_maxTemp_full = df_wday[df_wday["WeatherType"] ==
                              "Max temperature"].drop("WeatherType", axis=1)
    df_minTemp_full = df_wday[df_wday["WeatherType"] ==
                              "Min temperature"].drop("WeatherType", axis=1)
    df_avgWind_full = df_wday[df_wday["WeatherType"] ==
                              "Avg Wind"].drop("WeatherType", axis=1)

    df_avgTemp = df_avgTemp_full[
        df_avgTemp_full["Timestamp"] > "2013-04-14 23:00:00.000"]
    df_maxTemp = df_maxTemp_full[
        df_maxTemp_full["Timestamp"] > "2013-04-14 23:00:00.000"]
    df_minTemp = df_minTemp_full[
        df_minTemp_full["Timestamp"] > "2013-04-14 23:00:00.000"]
    df_avgWind = df_avgWind_full[
        df_avgWind_full["Timestamp"] > "2013-04-14 23:00:00.000"]

# parametar 'closed' in function ffill() doesn't work properly, so it's my way
    df_add = pd.DataFrame({"Timestamp": ["2018-07-01 00:00:00.000"],
                           "WeatherValue": [0]})
    df_avgWind = df_avgWind.append(df_add, ignore_index=True)
    df_maxTemp = df_maxTemp.append(df_add, ignore_index=True)
    df_avgTemp = df_avgTemp.append(df_add, ignore_index=True)
    df_minTemp = df_minTemp.append(df_add, ignore_index=True)

    df_avgTemp_idx = df_avgTemp.set_index("Timestamp")
    df_avgTemp_idx.index = pd.to_datetime(df_avgTemp_idx.index)

    df_avgWind_idx = df_avgWind.set_index("Timestamp")
    df_avgWind_idx.index = pd.to_datetime(df_avgWind_idx.index)

    df_maxTemp_idx = df_maxTemp.set_index("Timestamp")
    df_maxTemp_idx.index = pd.to_datetime(df_maxTemp_idx.index)

    df_minTemp_idx = df_minTemp.set_index("Timestamp")
    df_minTemp_idx.index = pd.to_datetime(df_minTemp_idx.index)

    difference_avgTemp = pd.date_range(start="2013-04-15",
                                       end="2018-06-30").difference(
                                           df_avgTemp_idx.index)
    difference_avgWind = pd.date_range(start="2013-04-15",
                                       end="2018-06-30").difference(
                                           df_avgWind_idx.index)
    difference_maxTemp = pd.date_range(start="2013-04-15",
                                       end="2018-06-30").difference(
                                           df_maxTemp_idx.index)
    difference_maxTemp = pd.date_range(start="2013-04-15",
                                       end="2018-06-30").difference(
                                           df_minTemp_idx.index)

    df_resample_avgTemp = df_avgTemp_idx.resample("H").ffill()
    df_resample_avgWind = df_avgWind_idx.resample("H").ffill()
    df_resample_minTemp = df_minTemp_idx.resample("H").ffill()
    df_resample_maxTemp = df_maxTemp_idx.resample("H").ffill()

    df_resample_avgTemp = df_resample_avgTemp[:-1]
    df_resample_avgWind = df_resample_avgWind[:-1]
    df_resample_minTemp = df_resample_minTemp[:-1]
    df_resample_maxTemp = df_resample_maxTemp[:-1]

    df_ready_all = pd.concat([df_ready, df_resample_avgTemp], axis=1).rename(
        {"WeatherValue": "avg Temp"}, axis=1)
    df_ready_all = pd.concat([df_ready_all, df_resample_avgWind],
                             axis=1).rename({"WeatherValue": "avg Wind"},
                                            axis=1)
    df_ready_all = pd.concat([df_ready_all, df_resample_maxTemp],
                             axis=1).rename({"WeatherValue": "max Temp"},
                                            axis=1)
    df_ready_all = pd.concat([df_ready_all, df_resample_minTemp],
                             axis=1).rename({"WeatherValue": "min Temp"},
                                            axis=1)

    # global df_data
    df_data = df_ready_all.reset_index()

    # Data for day 09.01 is equal with data for day 08.01.2018
    df_data.loc[41520:41543, "Load"] = df_load_08["Load"].values

    # Prepare dataset for forecasting

    date_time = pd.to_datetime(df_data.pop("Timestamp"),
                               format="%d.%m.%Y %H:%M:%S")

    df_data.describe().transpose()

    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = (365.2425) * day

    df_data["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df_data["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df_data["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df_data["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    return df_data


# Examining of features periodicality
def plot_fft(df_data):
    fft = tf.signal.rfft(df_data['Load'])
    f_per_dataset = np.arange(0, len(fft))

    n_samples_h = len(df_data['Load'])
    hours_per_year = 24*365.2524
    years_per_dataset = n_samples_h/(hours_per_year)

    f_per_year = f_per_dataset/years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 20000000)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')

    return


# Split data and prepare train, validations and test set
def split_normalize_data(df_data):
    column_indices = {name: i for i, name in enumerate(df_data.columns)}

    n = len(df_data)
    # global train_df
    # global val_df
    # global test_df
    train_df = df_data[0: int(n * 0.7)]
    val_df = df_data[int(n * 0.7): int(n * 0.9)]
    test_df = df_data[int(n * 0.9):]

    # global num_features
    num_features = df_data.shape[1]

    # Normalize data
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (df_data - train_mean) / train_std
    df_std = df_std.melt(var_name="Column", value_name="Normalized")
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x="Column", y="Normalized", data=df_std)
    _ = ax.set_xticklabels(df_data.keys(), rotation=90)

    return num_features, train_df, val_df, test_df


def compile_and_fit(model, window, patience=2):
    MAX_EPOCHS = 20
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()],
    )
    # global history
    history = model.fit(
        window.train,
        epochs=MAX_EPOCHS,
        validation_data=window.val,
        callbacks=[early_stopping],
    )
    return history


if __name__ == "__main__":
    main()
