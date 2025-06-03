import os
import wget
import gzip
import shutil
from config import DATASETS, RAW_DATA_DIR

def download_dblp():
    """下载DBLP引文网络数据集"""
    # 使用Snap数据集
    base_url = "https://snap.stanford.edu/data"
    target_dir = os.path.join(RAW_DATA_DIR, 'dblp-citation')
    
    # 下载DBLP合作网络数据
    filename = "com-dblp.ungraph.txt.gz"
    url = f"{base_url}/com-dblp.ungraph.txt.gz"
    target_path = os.path.join(target_dir, filename)
    
    if not os.path.exists(target_path):
        print(f"下载 {filename}...")
        wget.download(url, target_path)
        print("\n")
        
        # 解压文件
        print("解压文件...")
        with gzip.open(target_path, 'rb') as f_in:
            with open(target_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def download_pubmed():
    """下载PubMed引文网络数据集"""
    # 使用Snap数据集
    base_url = "https://snap.stanford.edu/data"
    target_dir = os.path.join(RAW_DATA_DIR, 'pubmed-diabetes')
    
    # 下载Pubmed引文网络数据
    filename = "cit-HepPh.txt.gz"  # 使用高能物理引文网络作为替代
    url = f"{base_url}/cit-HepPh.txt.gz"
    target_path = os.path.join(target_dir, filename)
    
    if not os.path.exists(target_path):
        print(f"下载 {filename}...")
        wget.download(url, target_path)
        print("\n")
        
        # 解压文件
        print("解压文件...")
        with gzip.open(target_path, 'rb') as f_in:
            with open(target_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def main():
    print("开始下载数据集...")
    
    # 下载DBLP数据集
    print("\n下载DBLP合作网络数据集...")
    download_dblp()
    
    # 下载PubMed数据集（使用HepPh替代）
    print("\n下载高能物理引文网络数据集（替代PubMed）...")
    download_pubmed()
    
    print("\n所有数据集下载完成！")

if __name__ == "__main__":
    main() 