print("start")
from DataPreparation import DataPreparation
print("start")
from DataVisualisation import DataVisualisation
print("start")
import ParametricalIdentification as pi
print("start")
import pandas as pd
import numpy as np
print("start")

# download data
url = 'https://auronum.co.uk/wp-content/uploads/2024/09/Auronum-Historic-Gold-Price-Data-5.xlsx'
save_path = 'data/Auronum-Historic-Gold-Price-Data-5.xlsx'
dp = DataPreparation()
# dp.download_file(url, save_path)

# read xlsx as pandas dataframe
df = dp.read_xlsx_to_dataframe(save_path)
debt = dp.read_csv_to_dataframe("data/HstDebt_17900101_20240930.csv")
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
debt.reset_index(inplace=True)
debt_plot = dv.plot_dataframe(debt,'date', 'USD')
debt_plot.show()
w = input()

y_year = np.array(yearly_avg['USD'])
y_month = np.array(monthly_avg['USD'])
x_month = np.arange(1, len(y_month)+1)
x_year = np.arange(1, len(y_year)+1)
x_year_k = np.arange(1, len(y_year)+1 +25)


# how many years add
years_to_add = 25

# last date in column
last_date = yearly_avg['date'].iloc[-1]

# add new dates
new_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=years_to_add, freq='YE')
new_data = pd.DataFrame({'date': new_dates})

# unite old dates and new one
yearly_avg_extended = pd.concat([yearly_avg, new_data]).reset_index(drop=True)


#polynomial
a,b,c = pi.general_least_squares_fit(x_year, y_year, pi.polynomial_func, [0.02,3,160], 'polynomial', epsilon=1e-5, is_print=False)
polynomial_y_year = pi.polynomial_func(x_year_k,a,b,c)

a,b,c = pi.general_least_squares_fit(x_month, y_month, pi.polynomial_func, [1,1,1], 'polynomial', epsilon=1e-5, is_print=False)
polynomial_y_month = pi.polynomial_func(x_month,a,b,c)

#exponent
a,b,c = [52.445,0.0656,98.73]  #general_least_squares_fit(x_year, y_year, exponent_func,[52.445,0.0656,98.73] , 'exponent', epsilon=1e-5, is_print=True)
exponent_y_year = pi.exponential_func(x_year_k,a,b,c)

a,b,c = pi.general_least_squares_fit(x_month, y_month, pi.exponential_func,  [52.445,0.0656,98.73], 'exponent', epsilon=1e-5, is_print=False)
exponent_y_month = pi.exponential_func(x_month,a,b,c)

#linear
a,b = pi.general_least_squares_fit(x_year, y_year, pi.linear_func, [28,-150], 'linear', epsilon=1e-5, is_print=False)
linear_y_year = pi.linear_func(x_year_k,a,b)

a,b = pi.general_least_squares_fit(x_month, y_month, pi.linear_func, [1,-1], 'linear', epsilon=1e-5, is_print=False)
linear_y_month = pi.linear_func(x_month,a,b)




yearly_avg_extended['linear'] = linear_y_year
monthly_avg['linear'] = linear_y_month
yearly_avg_extended['exponent'] = exponent_y_year
monthly_avg['exponent'] = exponent_y_month
yearly_avg_extended['polynomial'] = polynomial_y_year
monthly_avg['polynomial'] = polynomial_y_month

y_plt = dv.plot_dataframe(yearly_avg_extended, 'date', 'USD', else_y_cols=['linear','exponent','polynomial'], title='yearly_avg', x_label='date', y_label='USD', is_use_locator=True)
m_plt = dv.plot_dataframe(monthly_avg, 'date', 'USD', else_y_cols=['linear','exponent','polynomial'], title='monthly_avg', x_label='date', y_label='USD', is_use_locator=True)
y_plt.show()
m_plt.show()