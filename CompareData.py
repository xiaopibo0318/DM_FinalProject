import pandas as pd

data1 = pd.read_csv("standard_answer.csv")
data2 = pd.read_csv("40871223H_answer.csv")

# Merge the predicted dataset with the standard answer dataset based on 'Rank'
merged_data = pd.merge(data1[['Rank', 'Global_Sales']], data2   [['Rank', 'Global_Sales']], on='Rank', suffixes=('_predicted', '_standard'))

# Calculate the percentage similarity
# Similarity can be measured as 1 - (absolute error / actual value), averaged over all data points
merged_data['similarity'] = 1 - (abs(merged_data['Global_Sales_predicted'] - merged_data['Global_Sales_standard']) / merged_data['Global_Sales_standard'])
average_similarity = merged_data['similarity'].mean() * 100  # Convert to percentage

print(f"相似度:{average_similarity}")