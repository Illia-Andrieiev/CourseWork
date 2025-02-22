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
