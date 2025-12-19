import os
import requests
import zipfile
import pandas as pd
from tqdm import tqdm

class NatureDataDownloader:
    """Nature COVID-19移动性数据集下载器"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self._create_directories()
        
        self.dataset_url = "https://figshare.com/ndownloader/articles/12656165/versions/9"
        self.dataset_info = {
            "title": "Mobility, exposure, and epidemiological timelines of COVID-19 infections in China outside Hubei Province",
            "doi": "10.1038/s41597-021-00844-8",
            "files": [
                "case_data.csv",
                "mobility_data.csv", 
                "exposure_data.csv",
                "symptom_data.csv"
            ]
        }
    
    def _create_directories(self):
        """创建数据目录"""
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, "temporal_networks"), exist_ok=True)
    
    def download_dataset(self):
        """下载数据集"""
        print("开始下载Nature COVID-19数据集...")
        
        zip_path = os.path.join(self.raw_dir, "covid_mobility_data.zip")
        
        response = requests.get(self.dataset_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as file, tqdm(
            desc="下载进度",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        print(f"数据集已下载到: {zip_path}")
        return zip_path
    
    def extract_dataset(self, zip_path):
        """解压数据集"""
        print("解压数据集...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        
        print("解压完成")
        
        extracted_files = os.listdir(self.raw_dir)
        print(f"解压的文件: {extracted_files}")
        
        return extracted_files
    
    def load_case_data(self):
        """加载病例数据"""
        case_file = os.path.join(self.raw_dir, "case_data.csv")
        if os.path.exists(case_file):
            df = pd.read_csv(case_file)
            print(f"病例数据: {len(df)} 条记录")
            return df
        else:
            print("病例数据文件不存在")
            return None
    
    def load_mobility_data(self):
        """加载移动性数据"""
        mobility_file = os.path.join(self.raw_dir, "mobility_data.csv")
        if os.path.exists(mobility_file):
            df = pd.read_csv(mobility_file)
            print(f"移动性数据: {len(df)} 条记录")
            return df
        else:
            print("移动性数据文件不存在")
            return None
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return self.dataset_info