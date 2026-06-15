import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sales_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

df["Revenue"] = (
    df["Units_Sold"] *
    df["Unit_Price"] *
    (1 - df["Discount"] / 100)
)

df["Month"] = df["Date"].dt.month_name()
df["Quarter"] = df["Date"].dt.quarter
df["Day"] = df["Date"].dt.day_name()

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

revenue = np.array(df["Revenue"])

print("\nNumPy Analysis")
print("Total Revenue:", np.sum(revenue))
print("Average Revenue:", np.mean(revenue))
print("Maximum Revenue:", np.max(revenue))
print("Minimum Revenue:", np.min(revenue))
print("Standard Deviation:", np.std(revenue))

product_sales = (
    df.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

category_sales = (
    df.groupby("Category")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

region_sales = (
    df.groupby("Region")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

monthly_sales = (
    df.groupby("Month")["Revenue"]
    .sum()
)

payment_sales = (
    df.groupby("Payment_Method")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop Products")
print(product_sales)

print("\nCategory Sales")
print(category_sales)

print("\nRegion Sales")
print(region_sales)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=product_sales.index,
    y=product_sales.values
)
plt.title("Revenue by Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=region_sales.index,
    y=region_sales.values
)
plt.title("Revenue by Region")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(
    category_sales.values,
    labels=category_sales.index,
    autopct="%1.1f%%"
)
plt.title("Category Contribution")
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(
    x=monthly_sales.index,
    y=monthly_sales.values,
    marker="o"
)
plt.title("Monthly Revenue Trend")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=payment_sales.index,
    y=payment_sales.values
)
plt.title("Revenue by Payment Method")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(
    df["Revenue"],
    bins=20,
    kde=True
)
plt.title("Revenue Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(
    x=df["Revenue"]
)
plt.title("Revenue Outliers")
plt.show()

numeric_cols = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 6))
sns.heatmap(
    numeric_cols.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

best_product = product_sales.idxmax()
best_region = region_sales.idxmax()

print("\nBusiness Insights")
print(f"Best Product: {best_product}")
print(f"Best Region: {best_region}")
print(f"Total Revenue: {df['Revenue'].sum():,.2f}")
print(f"Average Order Revenue: {df['Revenue'].mean():,.2f}")
