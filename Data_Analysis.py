import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('new_data.csv')
print(df.head())
print("Rows and columns:", df.shape)

print(df.describe())
print(df.dtypes)
print("Missing values:\n", df.isnull().sum())
print("Unique products:", df['Product'].unique())
print("Unique regions:", df['Region'].unique())

df['Revenue'] = df['Revenue'].fillna(df['Units Sold'] * df['Unit Price'])
df['Revenue'].fillna(df['Revenue'].mean(), inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
df.drop_duplicates(inplace=True)

revenue_by_product = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
print(revenue_by_product)
monthly_revenue = df.groupby('Month')['Revenue'].sum()
print(monthly_revenue)
regional_sales = df.groupby('Region')['Revenue'].sum()
print(regional_sales)

revenue_by_product.plot(kind='bar', title='Total Revenue by Product', ylabel='Revenue', xlabel='Product')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

monthly_revenue.plot(kind='line', title='Monthly Revenue Trend', marker='o')
plt.ylabel('Revenue')
plt.xlabel('Month')
plt.tight_layout()
plt.show()

regional_sales.plot(kind='pie', autopct='%1.1f%%', title='Revenue by Region')
plt.ylabel('')
plt.tight_layout()
plt.show()

plt.hist(df['Units Sold'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Units Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
