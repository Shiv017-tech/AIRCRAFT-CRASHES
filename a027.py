# Project: Fatalities from people on board (modern era, 1990 onwards)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
# STEP 1: Load the data file (it must be in the same folder)
df = pd.read_csv(r"C:\Users\Shivraj\OneDrive\Desktop\dt\archive\Airplane_Crashes_and_Fatalities_Since_1908.csv")
print("Rows and columns:", df.shape)
# STEP 2: Clean the data
# Keep only crashes from 1990 onwards
df["Year"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.year
df = df[df["Year"] >= 1990].copy()
df = df.dropna(subset=["Aboard", "Fatalities"])
# STEP 3: Choose input X and output y
X = df[["Aboard"]] # double brackets: X must be a table
y = df["Fatalities"]
# STEP 4: Split: 80% to train, 20% to test
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42)
# STEP 5: Train the model (find the best straight line)
model = LinearRegression()
model.fit(X_train, y_train)
# STEP 6: Test on the hidden 20%
y_pred = model.predict(X_test)
print("R2 score :", round(r2_score(y_test, y_pred), 3))
print("Average error:", round(mean_absolute_error(y_test, y_pred), 2), "people")
print("Slope m :", round(model.coef_[0], 4))
print("Intercept c :", round(model.intercept_, 2))
# STEP 7: Predict a new value and draw the graph
new = pd.DataFrame({"Aboard": [150]})
print("Predicted fatalities with 150 on board:", round(model.predict(new)[0], 2), "people")
line_x = pd.DataFrame({"Aboard": sorted(X["Aboard"])})
plt.scatter(X_test["Aboard"], y_test, s=12, alpha=0.6, label="Actual (test data)")
plt.plot(line_x["Aboard"], model.predict(line_x), color="red", label="Regression line")
plt.xlabel("People on board")
plt.ylabel("Fatalities")
plt.title("Fatalities from people on board (modern era, 1990 onwards)")
plt.legend()
plt.show()