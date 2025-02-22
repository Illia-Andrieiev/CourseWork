print("hello world")
from DataPreparation import DataPreparation
from DataVisualisation import DataVisualisation
import pandas as pd

# download data
url = 'https://auronum.co.uk/wp-content/uploads/2024/09/Auronum-Historic-Gold-Price-Data-5.xlsx'
save_path = 'data/Auronum-Historic-Gold-Price-Data-5.xlsx'
dp = DataPreparation()
# dp.download_file(url, save_path)

# read xlsx as pandas dataframe
df = dp.read_xlsx_to_dataframe(save_path)

# Grouping data by month and calculating the average
monthly_avg = df.resample('ME').mean()
monthly_avg.reset_index(inplace=True)

# Grouping data by year and calculating the average
yearly_avg = df.resample('YE').mean()
yearly_avg.reset_index(inplace=True)

#print("Средние значения за месяц:")
#print(monthly_avg)

#print("\n Средние значения за год:")
#print(yearly_avg)

dv = DataVisualisation()
dv.plot_dataframe(monthly_avg, 'date', 'USD', title='monthly_avg', x_label='date', y_label='USD')
dv.plot_dataframe(yearly_avg, 'date', 'USD', title='yearly_avg', x_label='date', y_label='USD')

