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

df_load = pd.read_csv(fname + "\\EMS\\EMS_Load.csv", delimiter=";",
                      header=0)
df_wday = pd.read_csv(fname + "\\EMS\\EMS_Weather_Daily.csv", delimiter=";",
                      header=0)
df_whour = pd.read_csv(fname + "\\EMS\\EMS_Weather_Hourly.csv", delimiter=";",
                       header=0)


unique_values_daily = df_wday["WeatherType"].unique()
unique_values_hourly = df_whour["WeatherType"].unique()

count_type_daily = df_wday["WeatherType"].value_counts()
count_type_hourly = df_whour["WeatherType"].value_counts()

df_cloud = df_whour[df_whour["WeatherType"] == "Cloud"].drop("WeatherType",
                                                             axis=1)
df_wind = df_whour[df_whour["WeatherType"] == "Wind"].drop("WeatherType",
                                                           axis=1)
df_temp_full = df_whour[df_whour["WeatherType"] == "Temperature"].drop(
    "WeatherType", axis=1
)

stats_cloud = df_cloud.describe().transpose()
stats_wind = df_wind.describe().transpose()
stats_load = df_load.describe().transpose()
stats_temp = df_temp_full.describe().transpose()

df_temp = df_temp_full[df_temp_full["Timestamp"] > "2013-04-14 23:00:00.000"]

df_load_idx = df_load.set_index("Timestamp")
df_load_idx.index = pd.to_datetime(df_load_idx.index)

df_temp_idx = df_temp.set_index("Timestamp")
df_temp_idx.index = pd.to_datetime(df_temp_idx.index)


print(pd.date_range(start="2013-04-15", end="2018-06-30").difference(
    df_load_idx.index))
print(pd.date_range(start="2013-04-15", end="2018-06-30").difference(
    df_temp_idx.index))


# Separate data for day 08.01.2018
df_load_08 = df_load.loc["41496":"41519"]

# missing day 09.01.2018, all 12 samples get last value from 08.01
df_resample_load = df_load_idx.resample("H").ffill()
# add one missing sample with value from previous sample
df_resample_temp = df_temp_idx.resample("H").ffill()

df_ready = pd.concat([df_resample_load, df_resample_temp], axis=1).rename(
    {"WeatherValue": "Temp"}, axis=1
)

# prepare data from file fajla 'Weather Daily'

df_avgTemp_full = df_wday[df_wday["WeatherType"] ==
                          "Average temperature"].drop("WeatherType", axis=1)
df_maxTemp_full = df_wday[df_wday["WeatherType"] == "Max temperature"].drop(
    "WeatherType", axis=1
)
df_minTemp_full = df_wday[df_wday["WeatherType"] == "Min temperature"].drop(
    "WeatherType", axis=1
)
df_avgWind_full = df_wday[df_wday["WeatherType"] == "Avg Wind"].drop(
    "WeatherType", axis=1
)


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

print(
    pd.date_range(start="2013-04-15", end="2018-06-30").difference(
        df_avgTemp_idx.index)
)
print(
    pd.date_range(start="2013-04-15", end="2018-06-30").difference(
        df_avgWind_idx.index)
)
print(
    pd.date_range(start="2013-04-15", end="2018-06-30").difference(
        df_maxTemp_idx.index)
)
print(
    pd.date_range(start="2013-04-15", end="2018-06-30").difference(
        df_minTemp_idx.index)
)

df_resample_avgTemp = df_avgTemp_idx.resample("H").ffill()
df_resample_avgWind = df_avgWind_idx.resample("H").ffill()
df_resample_minTemp = df_minTemp_idx.resample("H").ffill()
df_resample_maxTemp = df_maxTemp_idx.resample("H").ffill()

df_resample_avgTemp = df_resample_avgTemp[:-1]
df_resample_avgWind = df_resample_avgWind[:-1]
df_resample_minTemp = df_resample_minTemp[:-1]
df_resample_maxTemp = df_resample_maxTemp[:-1]

df_ready_all = pd.concat([df_ready, df_resample_avgTemp], axis=1).rename(
    {"WeatherValue": "avg Temp"}, axis=1
)
df_ready_all = pd.concat([df_ready_all, df_resample_avgWind], axis=1).rename(
    {"WeatherValue": "avg Wind"}, axis=1
)
df_ready_all = pd.concat([df_ready_all, df_resample_maxTemp], axis=1).rename(
    {"WeatherValue": "max Temp"}, axis=1
)
df_ready_all = pd.concat([df_ready_all, df_resample_minTemp], axis=1).rename(
    {"WeatherValue": "min Temp"}, axis=1
)


df_data = df_ready_all.reset_index()

# Data for day 09.01 is equal with data for day 08.01.2018
df_data.loc[41520:41543, "Load"] = df_load_08["Load"].values

# Prepare dataset for forecasting

date_time = pd.to_datetime(df_data.pop("Timestamp"),
                           format="%d.%m.%Y %H:%M:%S")
stats = df_data.describe().transpose()

timestamp_s = date_time.map(pd.Timestamp.timestamp)

day = 24 * 60 * 60
year = (365.2425) * day

df_data["Day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
df_data["Day cos"] = np.cos(timestamp_s * (2 * np.pi / day))
df_data["Year sin"] = np.sin(timestamp_s * (2 * np.pi / year))
df_data["Year cos"] = np.cos(timestamp_s * (2 * np.pi / year))

# Examining of features periodicality
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

# Split data and prepare train, validations and test set
column_indices = {name: i for i, name in enumerate(df_data.columns)}

n = len(df_data)
train_df = df_data[0: int(n * 0.7)]
val_df = df_data[int(n * 0.7): int(n * 0.9)]
test_df = df_data[int(n * 0.9):]

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


""" Model prepare """


# Indexing and offsets
class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(
            train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col="Load", max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col,
                                                                 None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue
            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )
            if n == 0:
                plt.legend()
        plt.xlabel("Time [h]")

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        window.train,
        epochs=MAX_EPOCHS,
        validation_data=window.val,
        callbacks=[early_stopping],
    )
    return history


""" Multi-step model """

OUT_STEPS = 24
multi_window = WindowGenerator(
    input_width=24, label_width=OUT_STEPS, shift=OUT_STEPS,
    label_columns=["Load"]
)

multi_window.plot()
multi_window


multi_val_performance = {}
multi_performance = {}


# Multi linear model
multi_linear_model = tf.keras.Sequential(
    [
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(
            OUT_STEPS * num_features,
            kernel_initializer=tf.initializers.zeros()
        ),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ]
)


history = compile_and_fit(multi_linear_model, multi_window)

multi_val_performance["Linear"] = multi_linear_model.evaluate(multi_window.val)
multi_performance["Linear"] = multi_linear_model.evaluate(multi_window.test,
                                                          verbose=0)

multi_window.plot(multi_linear_model)


# Multi dense model
multi_dense_model = tf.keras.Sequential(
    [
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, dense_units]
        tf.keras.layers.Dense(512, activation="relu"),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(
            OUT_STEPS * num_features,
            kernel_initializer=tf.initializers.zeros()
        ),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ]
)

history = compile_and_fit(multi_dense_model, multi_window)

multi_val_performance["Dense"] = multi_dense_model.evaluate(multi_window.val)
multi_performance["Dense"] = multi_dense_model.evaluate(multi_window.test,
                                                        verbose=0)

multi_window.plot(multi_dense_model)


# CNN model

CONV_WIDTH = 12
multi_conv_model = tf.keras.Sequential(
    [
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation="relu",
                               kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(
            OUT_STEPS * num_features,
            kernel_initializer=tf.initializers.zeros()
        ),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features]),
    ]
)

history = compile_and_fit(multi_conv_model, multi_window)


multi_val_performance["Conv"] = multi_conv_model.evaluate(multi_window.val)
multi_performance["Conv"] = multi_conv_model.evaluate(multi_window.test,
                                                      verbose=0)
multi_window.plot(multi_conv_model)


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

""" Autoregresive RNN model """


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

prediction, state = feedback_model.warmup(multi_window.example[0])
prediction.shape

# FeedBack.call = call

print(
    "Output shape (batch, time, features): ",
    feedback_model(multi_window.example[0]).shape,
)

history = compile_and_fit(feedback_model, multi_window)


multi_val_performance["AR LSTM"] = feedback_model.evaluate(multi_window.val)
multi_performance["AR LSTM"] = feedback_model.evaluate(multi_window.test,
                                                       verbose=0)
multi_window.plot(feedback_model)


# Performance
x = np.arange(len(multi_performance))
width = 0.3

metric_name = "mean_absolute_error"
metric_index = feedback_model.metrics_names.index("mean_absolute_error")
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label="Validation")
plt.bar(x + 0.17, test_mae, width, label="Test")
plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
plt.ylabel("MAE (average over all times and outputs)")
_ = plt.legend()

for name, value in multi_performance.items():
    print(f"{name:8s}: {value[1]:0.4f}")
