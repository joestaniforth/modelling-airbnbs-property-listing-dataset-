import boto3
import os
import requests
import pandas as pd

def download_images(bucket:str, uuid_list:list) -> None:
    s3 = boto3.client('s3')
    #not scalable, list_objects_v2 returns max 1000 items
    response = s3.list_objects_v2(
        Bucket = bucket,
        Prefix = 'unstructured/'
    )
    file_list = [metadata['Key'] for metadata in response['Content']]
    # if bucket public
    for key in file_list:
        folder_name = key.split('/')[2]
        file_name = key.split('/')[3]
        if not os.path.isdir(f'data\\images\\{folder_name}'):
            os.mkdir(f'data\\images\\{folder_name}')
        response = requests.get(f'https://{bucket}.s3.amazonaws.com/{key}')
        with open(f'data\\images\\{folder_name}\\{file_name}', 'wb') as file:
            file.write(response.content)
    # if not
    for key in file_list:
        folder_name = key.split('/')[2]
        file_name = key.split('/')[3]
        s3.download_file(bucket,key,f'data\\images\\{folder_name}\\{file_name}')
        


if __name__ == '__main__':
    df = pd.read_csv('data\\clean_tabular_data')
    download_images(bucket = '?', uuid_list = df['ID'])