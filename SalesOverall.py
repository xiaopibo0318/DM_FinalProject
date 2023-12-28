import pandas as pd
import matplotlib.pyplot as plt

# Reload the CSV file since the code execution state was reset
file_path = 'stringData.csv'
nintendo_data = pd.read_csv(file_path)

# Group the data by publisher and sum up the global sales
publisher_sales = nintendo_data.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10)

# Create the plot for the top 10 publishers
plt.figure(figsize=(12, 6))
publisher_sales.plot(kind='bar', color='green')
plt.title('Top 10 Video Game Publishers by Global Sales')
plt.xlabel('Publisher')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
publisher_sales_file = 'top_10_publishers_sales.png'
plt.savefig(publisher_sales_file)

publisher_sales_file, plt.show()