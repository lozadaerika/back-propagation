import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# read the csv file
name='A1-synthetic/A1-synthetic.txt'

df= pd.read_csv(name,sep='\t')
print(df.head())

print(df.describe())

# Extract the numerical columns for normalization
numerical_columns = df.columns

#Normalization
#scaler = MinMaxScaler()
#df_normalized = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)

df_processed=df.copy()
# Data normalization
# Min-Max Scaling
df_processed.iloc[:, 0]= (df.iloc[:,0] - df.iloc[:,0].min()) / (df.iloc[:,0].max() - df.iloc[:,0].min())
df_processed.iloc[:, 9]= (df.iloc[:,9] - df.iloc[:,9].min()) / (df.iloc[:,9].max() - df.iloc[:,9].min())

print(df_processed.head())

output_file_name=name.split(".")[0]+'-normalized.csv'
df_processed.to_csv(output_file_name,sep=',', index=False,header=None)

