import pandas as pd

df=pd.read_csv(r"E:\py_projects\pythonProject\Anomaly-Transformer-main\dataset\NIP_KPI\data_train.csv")
df = df.values[:, 2]
df = pd.DataFrame(df)
df.to_pickle(r"E:\py_projects\计网\OmniAnomaly-master\data\KPI_train.pkl")