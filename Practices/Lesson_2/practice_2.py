import time
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


csv_file = "2017_jun_final.csv"

# Task 2.1
df = pd.read_csv(csv_file, header=0)

# Task 2.2 Прочитайте отриману таблицю, використовуючи метод head
df.head()

# Task 2.3 Визначте розмір таблиці за допомогою методу shape
df.shape

# Task 2.4 Визначте типи всіх стовпців за допомогою dataframe.dtypes
df.dtypes

# Task 2.5 Порахуйте, яка частка пропусків міститься в кожній колонці (використовуйте методи isnull та sum)
columns = df.columns
total_missing_data = np.sum(pd.isnull(df[columns[1:]]))


# Task 2.6 Видаліть усі стовпці з пропусками, крім стовпця "Мова програмування"
def del_nan_columns(df: DataFrame) -> DataFrame:
    counter = 1
    while counter < len(columns):
        empty_fields = np.sum(pd.isnull(df[columns[counter]]))
        if empty_fields > 0 and columns[counter] != "Язык.программирования":
            df = df.drop([columns[counter]], axis=1)
        counter += 1

    return df


df = del_nan_columns(df)

# Task 2.7 Знову порахуйте, яка частка пропусків міститься в кожній колонці і переконайтеся, що залишився тільки стовпець "Мова.програмування"
columns = df.columns
total_missing_data = np.sum(pd.isnull(df[columns[1:]]))

# Task 2.8 Видаліть усі рядки у вихідній таблиці за допомогою методу dropna
df = df.dropna()

# Task 2.9 Визначте новий розмір таблиці за допомогою методу shape
df.shape

# Task 2.10 Створіть нову таблицю python_data, в якій будуть тільки рядки зі спеціалістами, які вказали мову програмування Python
py_series = df["Язык.программирования"] == "Python"
python_data = df.loc[py_series]  # Передається або int або Series

# Task 2.11 Визначте розмір таблиці python_data за допомогою методу shape
python_data.shape

# Task 2.12 Використовуючи метод groupby, виконайте групування за стовпчиком "Посада"
dev_dfgby = python_data.groupby("Должность")

# Task 2.13 Створіть новий DataFrame, де для згрупованих даних за стовпчиком "Посада",
# виконайте агрегацію даних за допомогою методу agg і
# знайдіть мінімальне та максимальне значення у стовпчику "Зарплата.в.місяць"
min_max_sal = dev_dfgby["Зарплата.в.месяц"].agg(["min", "max"])


# Task 2.14 Створіть функцію fill_avg_salary,
# яка повертатиме середнє значення заробітної плати на місяць.
# # Використовуйте її для методу apply та створіть новий стовпчик "avg"
def fill_avg_salary(row):
    row["avg"] = row[["min", "max"]].mean()
    row["position"] = row.name
    return row


df1 = min_max_sal.apply(fill_avg_salary, axis=1)
df1 = df1[["position", "min", "max", "avg"]]

# Task 2.15 Створіть описову статистику за допомогою методу describe для нового стовпчика.
df1["avg"].describe()

# Task 2.16 Збережіть отриману таблицю в CSV файл
df1.to_csv("py_sal.csv", index=False)

# Chart 1
plt.close("all")
# df = pd.read_csv(csv_file, header=0)
# comp_type = df.groupby(["Тип.компании", "Город"])["Город"].count()
# kyiv = comp_type[:, "Киев"]
# kharkiv = comp_type[:, "Харьков"]

# plt.plot(kyiv, "bo", label="Kyiv")
# plt.plot(kharkiv, "ro", label="Kharkiv")
# plt.xlabel("Type", fontsize="small", color="midnightblue")
# plt.ylabel("Quantity", fontsize="small", color="midnightblue")
# plt.legend()
# plt.grid()
# plt.show()
# plt.close("all")

# Chart 2
# df = pd.read_csv(csv_file, header=0)
# sal_sex = df.groupby(["Пол", "Город"])["Зарплата.в.месяц"].agg("mean")
# f_sal = sal_sex["женский"]
# m_sal = sal_sex["мужской"]

# plt.plot(f_sal, "b-o", label="Women's average salary")
# plt.plot(m_sal, "r-o", label="Men's average salary")
# plt.xticks(rotation=90)
# plt.xlabel("City", fontsize="small", color="midnightblue")
# plt.ylabel("Salary", fontsize="small", color="midnightblue")
# plt.title("Women's average salary vs Men's in Ukraine")
# plt.legend()
# plt.grid()
# plt.show()
# plt.close("all")

# Chart 3
# df = pd.read_csv(csv_file, header=0)
# eng_sal = df.groupby(["Уровень.английского"])["Зарплата.в.месяц"].agg(
#     ["min", "max", "mean"]
# )

# data = eng_sal["mean"]
# labels = eng_sal.index

# plt.bar(
#     labels,
#     data,
#     color=["b", "r", "y", "g", "c"],
# )

# plt.xlabel("Рівень англійської", fontsize="small", color="midnightblue")
# plt.ylabel("Зарплата", fontsize="small", color="midnightblue")
# plt.xticks(rotation=90)
# plt.title("Вплив знанная англійської мови на зарплату", fontsize=15)
# plt.show()
# plt.close("all")

# Chart 4
df = pd.read_csv(csv_file, header=0)
# group_data = df.groupby(["Должность", "Пол"])["Зарплата.в.месяц"].agg("max")
# a = group_data[:, "женский"]
# poss = a.index.values
# sal = a.values
# dfq = pd.concat(
#     [pd.DataFrame(poss, columns=["Position"]), pd.DataFrame(sal, columns=["Salary"])],
#     axis=1,
# )

# Chart 5
# df = df.loc[df["Должность"] == "DevOps"]
# df = df.groupby(["exp"])["salary"].agg(["min", "max"])

# exp = df.index.values
# min_sal = df.values[:, 0]
# max_sal = df.values[:, 1]

# plt.plot(exp, min_sal)
# plt.plot(exp, max_sal)
# plt.show()
