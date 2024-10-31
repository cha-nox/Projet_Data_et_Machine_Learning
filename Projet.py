from utils import addInfo
from pandas import DataFrame, read_csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_data_frame = read_csv(data_url, sep = "\s+", skiprows = 22, header = None)
    data = np.hstack([raw_data_frame.values[::2, :], raw_data_frame.values[1::2, :2]])
    target = raw_data_frame.values[1::2, 2]
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    boston = DataFrame(data = data,columns = columns)
    boston.head()