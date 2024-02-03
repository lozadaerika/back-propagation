import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# read the csv file
name='A1-personalized/A1-energy.txt'

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
df_processed.iloc[:, 1]= (df.iloc[:,1] - df.iloc[:,1].min()) / (df.iloc[:,1].max() - df.iloc[:,1].min())
df_processed.iloc[:, 2]= (df.iloc[:,2] - df.iloc[:,2].min()) / (df.iloc[:,2].max() - df.iloc[:,2].min())
df_processed.iloc[:, 3]= (df.iloc[:,3] - df.iloc[:,3].min()) / (df.iloc[:,3].max() - df.iloc[:,3].min())

print(df_processed.head())

output_file_name=name.split(".")[0]+'-normalized.csv'
df_processed.to_csv(output_file_name,sep=',', index=False,header=None)

