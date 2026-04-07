# PRACTICAL 1 ARIMA MANUAL MODEL
import pandas as p
import numpy as n
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
u = ""
d = p.read_csv(u)
d["Date"] = p.to_datetime(d["Date"])
d = d.rename(columns={"Consumption": "Production"}) if "Consumption" in d.columns else d
d = d.set_index("Date")
ts = d["Production"].interpolate()
h = 365
tr = ts[:-h]
te = ts[-h:]
m = ARIMA(tr, order=(1,1,1)).fit()
pr = m.forecast(steps=h)
r = n.sqrt(mean_squared_error(te, pr))
print("RMSE:", round(r,2))
plt.plot(te.index, te, "r")
plt.plot(te.index, pr, "--")
plt.show()



# PRACTICAL 1 ARIMA AUTO MODEL
import pandas as p
import numpy as n
import itertools as it
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
u = ""
d = p.read_csv(u)
d["Date"] = p.to_datetime(d["Date"])
d = d.rename(columns={"Consumption": "Production"}) if "Consumption" in d.columns else d
d = d.set_index("Date")
ts = d["Production"].interpolate()
h = 365
tr = ts[:-h]
te = ts[-h:]
ba, bo, bm = 1e9, None, None
for o in it.product(range(3), repeat=3):
    try:
        m = ARIMA(tr, order=o).fit()
        if m.aic < ba:
            ba, bo, bm = m.aic, o, m
    except:
        pass
pr = bm.forecast(steps=h)
r = n.sqrt(mean_squared_error(te, pr))
print("Best Order:", bo)
print("RMSE:", round(r,2))
plt.plot(te.index, te, "r")
plt.plot(te.index, pr, "--")
plt.show()



# PRACTICAL 2 LINEAR REGRESSION MODEL
import pandas as p
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
u = ""
d = p.read_csv(u, parse_dates=["Month"], index_col="Month")
d.columns = ["P"]
a = d.copy()
a["L1"] = a["P"].shift(1)
a = a.dropna()
s = int(len(a) * 0.8)
tr = a.iloc[:s]
te = a.iloc[s:]
x1 = tr[["L1"]]
y1 = tr["P"]
x2 = te[["L1"]]
y2 = te["P"]
m = LinearRegression()
m.fit(x1, y1)
pr = m.predict(x2)
r = sqrt(mean_squared_error(y2, pr))
print("Linear Regression RMSE:", round(r, 2))
plt.plot(d.index, d["P"], label="Actual", color="black")
plt.plot(te.index, pr, label="Linear Regression", color="orange")
plt.legend()
plt.show()



# PRACTICAL 2 RANDOM FOREST MODEL
import pandas as p
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
u = ""
d = p.read_csv(u, parse_dates=["Month"], index_col="Month")
d.columns = ["P"]
a = d.copy()
for i in range(1, 4):
    a[f"L{i}"] = a["P"].shift(i)
a = a.dropna(
s = int(len(a) * 0.8)
tr = a.iloc[:s]
te = a.iloc[s:]
x1 = tr[["L1", "L2", "L3"]]
y1 = tr["P"]
x2 = te[["L1", "L2", "L3"]]
y2 = te["P"]
m = RandomForestRegressor(n_estimators=100, random_state=42)
m.fit(x1, y1)
pr = m.predict(x2)
r = sqrt(mean_squared_error(y2, pr))
print("Random Forest RMSE:", round(r, 2))
plt.plot(d.index, d["P"], label="Actual", color="black")
plt.plot(te.index, pr, label="Random Forest", color="green")
plt.legend()
plt.show()



# PRACTICAL 2 SARIMA MODEL
import pandas as p
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
u = ""
d = p.read_csv(u, parse_dates=["Month"], index_col="Month")
d.columns = ["P"]
s = int(len(d) * 0.8)
tr = d.iloc[:s]
te = d.iloc[s:]
m = SARIMAX(tr, order=(1,1,1), seasonal_order=(1,1,1,12))
f = m.fit(disp=False)
pr = f.forecast(steps=len(te))
r = sqrt(mean_squared_error(te, pr))
print("SARIMA RMSE:", round(r, 2))
plt.plot(d.index, d["P"], label="Actual", color="black")
plt.plot(te.index, pr, label="SARIMA", color="red")
plt.legend()
plt.show()



# PRACTICAL 3 LDA TOPIC MODELING
import pandas as p
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
u = ""
d = p.read_json(u)
txt = d["content"].astype(str)
v = CountVectorizer(stop_words="english", max_df=0.95, min_df=2)
x = v.fit_transform(txt)
m = LatentDirichletAllocation(n_components=3, random_state=42)
m.fit(x)
f = v.get_feature_names_out()
for i, t in enumerate(m.components_):
    w = [f[j] for j in t.argsort()[-10:][::-1]]
    print("Topic", i+1, ":", ", ".join(w))
pr = m.transform(x)
r = p.DataFrame(pr, columns=["Science", "Arts", "Commerce"])
r["Topic"] = r.idxmax(axis=1)
print(r.head())



# PRACTICAL 4 TEXT CLASSIFICATION USING LOGISTIC REGRESSION
import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
u = ""
d = p.read_table(u, header=None, names=["y", "x"])
v = TfidfVectorizer(stop_words="english")
x = v.fit_transform(d["x"])
y = d["y"]
x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=42)
m = LogisticRegression(max_iter=1000)
m.fit(x1, y1)
p1 = m.predict(x2)
print("Accuracy:", round(accuracy_score(y2, p1) * 100, 2), "%")
print(classification_report(y2, p1))



# PRACTICAL 5 ARTIFICIAL NEURAL NETWORK MODEL
import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
u = ""
d = p.read_csv(u)
x = d.drop("Outcome", axis=1)
y = d["Outcome"]
x1, x2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=42)
s = StandardScaler()
x1 = s.fit_transform(x1)
x2 = s.transform(x2)
m = Sequential([
    Dense(12, activation="relu", input_shape=(x1.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
m.fit(x1, y1, epochs=50, batch_size=10, verbose=0)
l, a = m.evaluate(x2, y2, verbose=0)
p1 = (m.predict(x2) > 0.5).astype(int)
print("Accuracy:", round(a * 100, 2), "%")
print(confusion_matrix(y2, p1))
print(classification_report(y2, p1))
