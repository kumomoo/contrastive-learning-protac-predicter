import pandas as pd
import pickle

df = pd.read_csv("data/protac.csv")
names = df["Compound ID"].tolist()
with open("data/name.pkl", "wb") as f:
    pickle.dump(names, f)
