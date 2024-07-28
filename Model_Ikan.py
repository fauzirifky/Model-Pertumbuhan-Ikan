import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import streamlit as st
from sklearn.linear_model import LinearRegression
import pymysql



# Connect to the database
# Accessing secrets
db_host = st.secrets["database"]["DB_HOST"]
db_user = st.secrets["database"]["DB_USER"]
db_password = st.secrets["database"]["DB_PASSWORD"]
db_name = st.secrets["database"]["DB_NAME"]
#

connection = pymysql.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    db=db_name
)

# Query to fetch data
query = "SELECT * FROM DataMasCipta"

# Load data into a pandas DataFrame
data = pd.read_sql(query, connection)
print(data)
# Close the connection
connection.close()


st.title("Modeling the Impact of Brassica Colocation on the Growth Dynamics of Anabas")
# data = pd.read_excel("Data Ikan Mas Cipta.xlsx")
def model(t, a, b, c, d):
    return a/(1 + np.exp(-b*(t+c))) + d



weeks = np.arange(1,9)
pl = []
masses = []
for p in data["P"].unique():
    temp = data[data["P"] == p]
    mass_temp = {}
    mass_mean = {}
    for w in weeks:

        mass_temp[w] = temp[temp["Minggu"]==w]["Berat Individu"]
        mass_mean[w] = temp[temp["Minggu"]==w]["Berat Individu"].mean()


    pl.append(mass_temp)
    masses.append(mass_mean)
print(pl)
params, covariances = {}, {}
growth = {}
for i in range(len(pl)):
    initial_guess = [1, 1, 0.1, 1]  # Initial guess for the parameters
    y_data = np.array(list(masses[i].values()), dtype=float)
    params[i], covariances[i] = curve_fit(model, weeks, y_data, p0=initial_guess, maxfev=10000)
    growth[i] = np.diff(y_data)

t_weeks = np.linspace(weeks[0], weeks[-1],20)
fig, axes = plt.subplots(2,2, figsize=(8,6))
axes = axes.flatten()
for i in range(len(pl)):
    axes[i].set_title("Experiment number-"+str(i+1))
    axes[i].boxplot(pl[i].values())
    axes[i].plot(weeks, masses[i].values(), label = 'average')
    axes[i].plot(t_weeks, model(t_weeks, *params[i]), label='logistic model')
    axes[i].set_xticklabels(pl[i].keys())
    axes[i].legend()
st.write("## Weekly masses by using Logistic model")
st.pyplot(fig)
fig, ax = plt.subplots( figsize=(8,4))

colors = ['tab:red', 'tab:blue', 'tab:green', 'purple']
for i in range(len(pl)):
    # Use weeks[1:] for the linear regression
    weeks_trimmed = weeks[1:].reshape(-1, 1)  # Reshape for sklearn
    growth_trimmed = growth[i]  # Assuming growth data matches the length

    # Create and fit the model
    model = LinearRegression()
    model.fit(weeks_trimmed, growth_trimmed)
    growth_pred = model.predict(weeks_trimmed)
    ax.plot(weeks_trimmed, growth_pred, '--',color=colors[i])
    ax.plot(weeks[1:],growth[i], '-o', color=colors[i], label="Experiment number-"+str(i+1))
ax.legend()
st.write("## Weekly growth by using Linear Regression")
st.pyplot(fig)
plt.show()
