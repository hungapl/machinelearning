!conda install pandas
!conda install scipy

import os
import pandas as pd
import pickle
from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter

current_dir = os.path.realpath(".") + "/"

ts = pd.read_csv(current_dir + 'timeseriesdata.csv', parse_dates=[1])
ts.info() # This shows that 'time' column is in datetime64 nanoseconds precision
ts.head(10)
ts = ts.dropna()

# Sort by time
ts = ts.sort_values('time')
# Express time as epoc
ts['epoc'] = ts['time'].astype(int)/1e9 # Since precision is ns
# Get time passed from previous observation
ts['time_diff_in_s'] = ts['epoc'] - ts['epoc'].shift(1)
start_time = ts['epoc'].iloc[0]
# Extract first hour of observations
ts_first_hour = ts[ts['epoc'].between(start_time, start_time + 3600)]

# Persist data as pickle file
ts.to_pickle(current_dir + 'timeseriesdata.pkl')
# Load data from pickle file
with open(current_dir + 'timeseriesdata.pkl', 'rb') as f:
    ts1 = pickle.load(f)

ts1.info()

# Plot time series using bokeh
p = figure(plot_width=1200, plot_height=900, x_axis_type="datetime", title='Example time series data', tools='wheel_zoom', toolbar_location='above')
p.title.text_font_size = '14pt'
p.line(ts.time, ts.value)
p.xaxis.axis_label = "Time"
p.xaxis.formatter=DatetimeTickFormatter(hours=["%H:%M"])
p.yaxis.axis_label = "Value"
show(p)