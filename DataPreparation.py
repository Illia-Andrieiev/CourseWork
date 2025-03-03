import pandas as pd
import requests
import os
class DataPreparation:
    def download_file(self, url, save_path):
        # create directory if not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # download file by url
        response = requests.get(url)
        
        # if responded successfully
        if response.status_code == 200:
            # write file
            with open(save_path, 'wb') as file:
                file.write(response.content)   
        else:
            print(f"Cannot download file. Status: {response.status_code}")

    def read_xlsx_to_dataframe(self, file_path):
        try:
            df = pd.read_excel(file_path, usecols=[4,5])
            df.columns = ["date","USD"]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            print(f"error: {e}")

            
    def read_csv_to_dataframe(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.columns = ["date","USD"]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            print(f"error: {e}")

