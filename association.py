from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
# Create a list of transactions (each transaction is a list of games released on a platform)

file_path = 'stringData.csv'
nintendo_data = pd.read_csv(file_path)

transactions = nintendo_data.groupby('Platform')['Name'].apply(list).tolist()

# Initialize the TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

# Convert the encoded data into a DataFrame
df = pd.DataFrame(te_ary, columns=te.columns_)

# Applying Apriori algorithm to find frequent item sets
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# Generating the association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Display the top 20 association rules
print(rules.head(20))