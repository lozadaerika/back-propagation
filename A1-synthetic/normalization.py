import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# read the csv file
name='A1-synthetic/A1-synthetic.txt'

df= pd.read_csv(name,sep='\t')
print(df.head())

print(df.describe())

numerical_columns = df.columns

#Normalization
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)

print(df_normalized.head())

output_file_name=name.split(".")[0]+'-normalized.csv'
df_normalized.to_csv(output_file_name,sep=',', index=False,header=None)

