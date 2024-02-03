import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# read the csv file
name='A1-turbine/A1-turbine.txt'

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
df_processed.iloc[:, 4]= (df.iloc[:,4] - df.iloc[:,4].min()) / (df.iloc[:,4].max() - df.iloc[:,4].min())
# Standardization
df_processed.iloc[:, 1]= (df.iloc[:,1] - df.iloc[:,1].mean()) / df.iloc[:,1].std()
df_processed.iloc[:, 2]= (df.iloc[:,2] - df.iloc[:,2].mean()) / df.iloc[:,2].std()
df_processed.iloc[:, 3]= (df.iloc[:,3] - df.iloc[:,3].mean()) / df.iloc[:,3].std()

print(df_processed.head())

output_file_name=name.split(".")[0]+'-normalized.csv'
df_processed.to_csv(output_file_name,sep=',', index=False,header=None)

