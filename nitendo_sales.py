import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the CSV file
file_path = 'stringData.csv'
nintendo_data = pd.read_csv(file_path)




# Ensure the 'Year' column is in a proper format
nintendo_data['Year'] = nintendo_data['Year'].dropna().astype(int)

# Group the data by year and sum up the global sales
yearly_sales = nintendo_data.groupby('Year')['Global_Sales'].sum().reset_index()

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Year', y='Global_Sales', data=yearly_sales, color='blue')
plt.title('Nintendo Global Sales by Year')
plt.xlabel('Year')
plt.ylabel('Global Sales (in millions)')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt_file = 'nintendo_sales_by_year.png'
plt.savefig(plt_file)

plt_file, plt.show()
