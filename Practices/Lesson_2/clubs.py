import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


# csv_file = "clubs.csv"
# df = pd.read_csv(csv_file)
# leagues = pd.unique(df["domestic_competition_id"])
# ukr_l = df[df["domestic_competition_id"] == "UKR1"]
# ukr_l = ukr_l.dropna(axis=1)

a = np.random.randint(-4000, 4000, size=100)
data = pd.Series(a)
