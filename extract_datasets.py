import os
import gzip
import shutil
from config import DATASETS, RAW_DATA_DIR

def extract_file(gz_path):
    """解压gzip文件"""
    if not os.path.exists(gz_path):
        print(f"文件不存在: {gz_path}")
        return False
    
    output_path = gz_path[:-3]  # 移除.gz后缀
    print(f"解压 {os.path.basename(gz_path)}...")
    
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"解压完成: {output_path}")
        return True
    except Exception as e:
        print(f"解压失败: {str(e)}")
        return False

def main():
    # 解压DBLP数据集
    dblp_gz = os.path.join(RAW_DATA_DIR, 'dblp-citation', 'com-dblp.ungraph.txt.gz')
    if extract_file(dblp_gz):
        print("DBLP数据集解压成功")
    
    # 解压HEP-PH数据集
    hepph_gz = os.path.join(RAW_DATA_DIR, 'pubmed-diabetes', 'cit-HepPh.txt.gz')
    if extract_file(hepph_gz):
        print("HEP-PH数据集解压成功")

if __name__ == "__main__":
    main() 