import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

url = "https://uk.wikipedia.org/wiki/Народжуваність_в_Україні"

page_html = requests.get(url=url).text
tabs = pd.read_html(page_html, header=0)
df = tabs[-12]

# Task 1.1
df.head(3)

# Task 1.2
df_shape = df.shape

# Task 1.4
df_dtypes = df.dtypes

# Task 1.6
columns = df.columns
total_missing_data = np.sum(pd.isnull(df[columns[1:]]))

# Task 1.7
df = df.drop([27])

# Task 1.8 Замініть відсутні дані в стовпцях середніми значеннями цих стовпців (метод fillna)
values = {
    "1950": df["1950"].mean(),
    "1960": df["1960"].mean(),
    "1970": df["1970"].mean(),
    "2014": df["2014"].mean(),
}
df = df.fillna(value=values)

# Task 1.9 Отримайте список регіонів, де рівень народжуваності у 2019 році був вищим за середній по Україні
y_2014_mean = df["2014"].mean()
result = df[df["2014"] > y_2014_mean][["регіон", "2014"]]

# Task 1.10 У якому регіоні була найвища народжуваність у 2014 році?
max_2014 = df["2014"].max()
max_2014_region = df[df["2014"] == max_2014][["регіон"]]

# Task 1.11 Побудуйте стовпчикову діаграму народжуваності по регіонах у 2019 році
year_14 = df["2014"]
regions = df["регіон"]


def clean_region(regs: list):
    result = []
    for reg in regs:
        if reg == "Автономна Республіка Крим":
            reg = "APK"
            result.append(reg)
            continue
        if reg == "Севастополь (міськрада)":
            reg = "Севастополь"
            result.append(reg)
            continue
        name = reg.split(" ")[-1]
        if name == "область":
            name = "о."
            reg = f"{reg.split(' ')[0]} {name}"
            result.append(reg)
        else:
            result.append(reg)
    return result


regions = clean_region(df["регіон"])


def rgb_colors(arg: int):
    counter = 0
    result = []
    while counter < arg:
        random_color = (
            round(random.random(), 1),
            round(random.random(), 1),
            round(random.random(), 1),
        )
        result.append(random_color)
        counter += 1
    return result


colors = rgb_colors(len(regions))

# plt.figure(figsize=(10, 6))
# plt.bar(regions, year_14, color=colors)
# plt.xticks(rotation=90)
# plt.tick_params(axis="both", labelsize=5)
# plt.title("Birth Rate in Ukraine in 2014", fontsize=10)
# plt.xlabel("Regions", fontsize="small", color="midnightblue")
# plt.ylabel("Birth Rate", fontsize="small", color="midnightblue")
# plt.show()
# plt.close("all")


# Chart 1
# plt.pie(
#     year_14,
#     labels=regions,
#     shadow=True,
#     autopct="%.2f%%",
#     pctdistance=1.15,
#     labeldistance=1.35,
# )

# plt.show()
# plt.close("all")

# Chart 2
df = tabs[-12].drop([27])
# zak = df.loc[(df["регіон"] == "Закарпатська область")]
# lviv = df.loc[(df["регіон"] == "Львівська область")]

a = df.groupby("регіон")[df.columns[1:]].agg("max")
print(a.index)
print(a.loc["Закарпатська область"])
