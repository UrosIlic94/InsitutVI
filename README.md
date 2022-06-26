#  Load forecasting 24 hours ahead
At the beginning of the file, the libraries used in the code are imported and
the graphics parameters are set.

Then a path is set to the folder containing the data folder and the file
containing the WindowGenerator class. After that the WindowGenerator class has
been imported.

The main function is defined and the functions necessary for creating an
adequate dataset are called within it.

Then a class with the necessary methods for preparing, dividing, displaying data
and creating a dataset suitable for model training was used and the input
parameters for creating data windows were set. The input width is set
to 24 input data (24h) and the same for output data. Also, the necessary data
containing training DF, validation DF and test DF were forwarded.

After that, the RNN model was formed according to the instructions from the tutorial.

This model is trained with the compile and fit function, using a previously prepared data window. The plot that is generated shows the efficiency of the created model by showing the actual and predicted values. The second graph shows the performance of the created model, which is also printed in the workspace.

# FUNCTIONS DEFINITIONS

# Function import_data
The import_data function imports data from csv files into DataFrames (DF).
A separate DF is created for each of the three files.

# Function separete_hour
Within the separete_hour function, the data from the Weather_Hourly file is divided
into separate DFs containing data on temperature, cloudiness and wind.
Cloud and wind data have a lot of missing data and the time period of the first
available values is from the end of 2014. Due to the unclear way of approximation of the missing values, but also due to the known facts that clouds and wind directly affect the temperature, which is known throughout  the time period, I decided not to consider this data further in this version of data preparation for model training.

# Function prepare_halfFeature
Since the goal is to predict consumption for the future, load data are very
important data. The prepare_halfFeature function is used to prepare the first and second features (load and temperature). Because the load data starts from April 15, 2013, the temperature data is reduced to that time interval.
After that, data were prepared for the use of the "difference" function,
which showed that the consumption data lacked measurements for January 9, 2018.
Temperature data lacks only one hour value.

All missing hourly timestamps are filled with the resample () function with the
selection of the parameter H, and the values are filled with the last measured
value using the ffill() function. This is an acceptable approximation in the case of temperature because the temperature does not change significantly from hour to hour.

For consumption, I chose to fill in the missing values for 09.01 with the identical
measured on 08.01. This approximation is not the best possible solution,
but it is satisfactory considering that it is known that there is a periodicity
of load on a daily basis.

# Function prepared_dataset
Data from another file, 'Weather_Daily', was prepared in a similar way.
Since the data are given on a daily basis, using the resample () function were created of timestamps with a period of one hour for the entire observed time period. The new hourly values are the same for all selections during the same day. There was no measurement in this file for 3 days, which was also solved as in the previous file.

A small problem arose with the use of the ffill() function. By default, the ffill() function does not fill values for the last available date, but ends with the penultimate one. This is changeable with the 'closed' parameter, but in my case it didn't work the way I wanted, so I added another date before using the resample() function and then deleted the date at the end after that.

A summary Data Frame has been prepared, in which the above-mentioned consumption approximation for the missing January 9, 2018 was finally performed.

Summary Data Frame has added values, which will give the model an insight into the daily and annual periodicity of the load change.

# Function plot_fft
Confirmation that the annual and daily periodicity are most pronounced is visible on the basis of graphs, which are obtained by using the plot_fft function. What I noticed on this plot is the third largest peak, which is at a frequency of less than one day.
Assuming that this is the data of an industrial consumer, this peak is logical and indicates the periodicity within the working hours of this industrial plant. For some future work, it would probably be interesting to check the impact of this periodicity on the results that the model produces.

# Function split_normalize_data
The split_normalize_data function divides the data into training, validation and test data sets and normalizes them.

# Function compile_and_fit
Within this function, the value of epochs for model training and other parameters necessary for compiling and fitting models are defined.
