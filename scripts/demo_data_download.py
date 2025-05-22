# download the demo data from the google drive 
import os
import gdown
import zipfile

def download_from_google_drive(url, output):
    gdown.download(url, output, quiet=False)

def unzip_file(file_path, output_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

if __name__ == '__main__':
    data_url = "https://drive.google.com/uc?id=11xg5FuBjSn1pXqR8vJwfoxN1ypY_Xntv"
    ckpt_url = "https://drive.google.com/uc?id=18Urpigfm6hAVr0tFdy7-ImcuBl9qvwa3"

    # if not os.path.exists('data/blimp_filtered'):
    download_from_google_drive(data_url, 'data.zip')
    unzip_file('data.zip', '.')
    os.remove('data.zip')
    print('data downloaded')
    # else:
    #     print('data already exists')

    # if not os.path.exists('ckpt/evo_exp_full_a'):
    download_from_google_drive(ckpt_url, 'ckpt.zip')
    unzip_file('ckpt.zip', '.')
    os.remove('ckpt.zip')
    print('ckpt downloaded')
    # else:
    #     print('ckpt already exists')

