import pandas as pd
path = "/Users/myounes/Documents/Code/thesis/data/data_1file.csv" 


data_frame = pd.read_csv(path, delimiter=',')
print(data_frame.iloc[0, 0])
