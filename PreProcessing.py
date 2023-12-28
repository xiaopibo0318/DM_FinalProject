import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = 'updated_vgsales.csv'
data = pd.read_csv(file_path)

print(data.shape)
missing_values = data.isna().sum()
print(missing_values)
data['Year'].fillna(2006, inplace=True)
data['Publisher'].fillna("Unknown", inplace=True)
data_encoded = data.copy()
data_encoded.to_csv("stringData.csv", index=False)
# label_encoder = LabelEncoder()
# columns_to_encode = ['Name', 'Platform', 'Genre', 'Publisher']

# for column in columns_to_encode:
#     data_encoded[column] = label_encoder.fit_transform(data_encoded[column].astype(str))

# data_encoded.to_csv("newPredictGlobal.csv", index=False)
