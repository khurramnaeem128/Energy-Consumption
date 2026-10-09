# 1. Import Library and Load Dataset
import pandas as pd

df = pd.read_csv("Energy_consumption.csv")

# 2. Explore the Dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# 3. Check Missing Values and Duplicates
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.isnull().sum())

# 4. Feature Engineering - Extract Time Features
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.day
df["Month"] = df["Timestamp"].dt.month
df["Year"] = df["Timestamp"].dt.year

# 5. Analyze Correlation
print(df[["Timestamp", "Hour", "Day", "Month", "Year"]].head())
print(df.corr(numeric_only=True))

# 6. Convert Categorical Features into Numerical Features
df = pd.get_dummies(
    df,
    columns=["HVACUsage", "LightingUsage", "DayOfWeek", "Holiday"],
    drop_first=True
)
print(df.head())
print(df.columns)
print(df.head().to_string())

# 7. Define Features (X) and Target (y)
X = df.drop(columns=["EnergyConsumption", "Timestamp"])
y = df["EnergyConsumption"]

X_train = X.iloc[:800]
X_test = X.iloc[800:]

y_train = y.iloc[:800]
y_test = y.iloc[800:]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# 9. Train Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# 10. Make Predictions
y_pred = model.predict(X_test)

# 11. Compare Actual vs Predicted Values
X_test_5 = X_test.iloc[:5]
y_test_5 = y_test.iloc[:5]
y_pred_5 = model.predict(X_test_5)

comparison = pd.DataFrame({
    "Timestamp": df["Timestamp"].iloc[800:805].values,
    "Actual": y_test_5.values,
    "Predicted": y_pred_5
})

print(comparison)

# 12. Evaluate Model Performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

# 13. Visualize Actual vs Predicted Energy Consumption
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption")

plt.show()

# 14. Average energy consumption by hour

hour_energy = df.groupby("Hour")["EnergyConsumption"].mean()

print(hour_energy)

plt.plot(hour_energy.index, hour_energy.values)

plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Consumption")
plt.title("Average Energy Consumption by Hour")

plt.show()

# 15. Average energy consumption by day of week

df["DayOfWeek"] = df["Timestamp"].dt.day_name()

day_energy = df.groupby("DayOfWeek")["EnergyConsumption"].mean()

print(day_energy)

plt.bar(day_energy.index, day_energy.values)

plt.xlabel("Day of Week")
plt.ylabel("Average Energy Consumption")
plt.title("Average Energy Consumption by Day of Week")

plt.xticks(rotation=45)

plt.show()

from sklearn.ensemble import RandomForestRegressor

# 16. Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# 17. Train the model
rf_model.fit(X_train, y_train)

# 18.  Make predictions
rf_pred = rf_model.predict(X_test)

# 19. Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest MAE:", rf_mae)
print("Random Forest MSE:", rf_mse)
print("Random Forest R2:", rf_r2)

# Display Successful Prediction Message
print("successfully predicted energy consumption")

