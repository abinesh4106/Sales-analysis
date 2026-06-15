import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading
df = pd.read_csv("sales_data.csv")

print("\nDataset Preview")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# Feature Engineering
df["Revenue"] = df["Units_Sold"] * df["Unit_Price"]

print("\nRevenue Added")
print(df.head())

# Statistical Analysis
revenue_array = np.array(df["Revenue"])

print("\nNumPy Statistics")
print("Total Revenue:", np.sum(revenue_array))
print("Average Revenue:", np.mean(revenue_array))
print("Maximum Revenue:", np.max(revenue_array))
print("Minimum Revenue:", np.min(revenue_array))
print("Standard Deviation:", np.std(revenue_array))

# Sales Analysis
product_sales = (
    df.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

category_sales = (
    df.groupby("Category")["Revenue"]
    .sum()
)

region_sales = (
    df.groupby("Region")["Revenue"]
    .sum()
)

print("\nProduct Sales")
print(product_sales)

print("\nCategory Sales")
print(category_sales)

print("\nRegion Sales")
print(region_sales)

# Visualizations
plt.figure(figsize=(8, 5))
sns.barplot(
    x=product_sales.index,
    y=product_sales.values
)
plt.title("Revenue by Product")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.show()

plt.figure(figsize=(6, 5))
plt.pie(
    category_sales,
    labels=category_sales.index,
    autopct="%1.1f%%"
)
plt.title("Category Revenue Share")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=region_sales.index,
    y=region_sales.values
)
plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(
    df["Revenue"],
    bins=10,
    kde=True
)
plt.title("Revenue Distribution")
plt.show()

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(6, 4))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.show()

# Business Insights
best_product = product_sales.idxmax()

print("\nBest Selling Product:")
print(best_product)

print("\nProject Completed Successfully")
