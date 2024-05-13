import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path

DATADIR = "../data"

# Load the data
def readCSV(fname="data.csv"):
    data = pd.read_csv(path.join(DATADIR, fname))
    return data


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def videoToPeaks(fname: str):
    pass

def processCSV():
    pass

def processData():
    pass