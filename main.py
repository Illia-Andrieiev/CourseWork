print("hello world")
from DataPreparation import DataPreparation
# Пример использования функции
url = 'https://auronum.co.uk/wp-content/uploads/2024/09/Auronum-Historic-Gold-Price-Data-5.xlsx'
save_path = 'data/Auronum-Historic-Gold-Price-Data-5.xlsx'
dp = DataPreparation()
dp.download_file(url, save_path)
