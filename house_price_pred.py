from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
import pandas as pd

data = [["West", "Excellent", 5500], ["Central", "Good", 4500],
        ["North", "Fair", 1700], ["Slum", "Poor", 600],["Upstate", "Good", 3800]]

df = pd.DataFrame(data, columns = ["Location", "Property Condition", "Price"])

rank = ["Poor", "Fair", "Good", "Excellent"]
property_encoder = OrdinalEncoder(categories = [rank]).fit(df[["Property Condition"]])
Neig_encoder = OneHotEncoder().fit(df[["Location"]])

property_value = property_encoder.transform(df[["Property Condition"]])
df[["Property Rank"]] = property_value
Neighbourhood = Neig_encoder.transform(df[["Location"]])
df[["Locale"]] = Neighbourhood

x = property_value
y = df["Price"].values

df.drop(columns = ["Location", "Property Condition"], inplace = True)

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict([[0],[1],[2],[3]])

print(f"The value of the slope is: {model.coef_[0]:.2f}")
print(f"The value of the Y-Intercept is: {model.intercept_:2f}")

print(f"""The model predicts the following prices for 'Poor', 'Fair', 'Good'
and 'Excellent' property conditions: """)
for condition, price in zip(rank, y_pred):
    print(f"- {condition}: ${price:.2f}")
