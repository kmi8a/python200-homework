import numpy as np
import pandas as pd
from prefect import task, flow

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name='values')

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {'mean': series.mean(), 'median': series.median(), 'std': series.std(), 'mode': series.mode()[0]}

@flow
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    return summarize_data(cleaned_series)


if __name__=='__main__':
    result = data_pipeline(arr)
    print(result)

## 1. Why might Prefect be more overhead than it is worth here?
## The complexity that prefect is capable of doesn't really add anything valuable to to this data pipeline, not neeeded at this level of complexity. 
## 2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
## A possible scenario would be when you need to run this pipeline automatically on a set schedule, prefect gives you that functionality withouth writing a custom scheduler.
## A second possible scenario would be when you need to track sucess, failure and retry states.