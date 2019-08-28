import pandas



from pandas import DataFrame


df = DataFrame([['111','222','333','444','555'],['666','777','888','999','000'],['1','2','3','4','5']])

export_csv = df.to_csv (r'C:\Users\Awangardowy Kaloryfe\Desktop\export_dataframe.csv', index = None, header=False)

print (df)