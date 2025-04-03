import os
import pydicom
from pydicom.errors import InvalidDicomError
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义源目录 (fixed) 和目标目录 (moving)
source_dir = 'images/fixed' 
# target_dir = 'images/week0_CT' # 旧的目标
target_dir = 'images/moving' 

def get_patient_id_from_source(directory):
    """递归查找源目录及其子目录，从第一个找到的有效 DICOM 文件中读取 PatientID"""
    logging.info(f"开始在目录 '{directory}' 及其子目录中查找 PatientID...")
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.dcm'):
                filepath = os.path.join(root, filename)
                logging.debug(f"尝试读取文件: {filepath}")
                try:
                    # 只读取必要的元数据，不需要整个像素数据
                    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                    # 检查是否存在 PatientID 标签
                    if 'PatientID' in ds:
                        patient_id = ds.PatientID
                        logging.info(f"从文件 '{filepath}' 读取到 PatientID: {patient_id}")
                        return patient_id
                    else:
                         logging.debug(f"文件 '{filepath}' 没有 PatientID 标签。")
                except InvalidDicomError:
                    logging.debug(f"文件 '{filepath}' 不是有效的 DICOM 文件，跳过。")
                except Exception as e:
                    logging.warning(f"读取文件 '{filepath}' 时发生错误: {e}")
    logging.error(f"错误: 在目录 '{directory}' 及其子目录中未找到包含 PatientID 的有效 DICOM 文件。")
    return None

def update_dicom_patient_id(target_directory, new_patient_id):
    """递归更新目标目录及其子目录中所有 DICOM 文件的 PatientID"""
    if new_patient_id is None:
        logging.error("错误: 未提供有效的 PatientID，无法更新。")
        return 0, 0 # 返回 (检查的总数, 更新的总数)

    total_checked = 0
    total_updated = 0
    logging.info(f"开始在目录 '{target_directory}' 及其子目录中更新 PatientID 为 '{new_patient_id}'...")

    for root, _, files in os.walk(target_directory):
        for filename in files:
            if filename.lower().endswith('.dcm'):
                filepath = os.path.join(root, filename)
                total_checked += 1
                logging.debug(f"处理文件: {filepath}")
                try:
                    ds = pydicom.dcmread(filepath)
                    original_id = ds.PatientID if 'PatientID' in ds else '不存在'
                    
                    # 检查 PatientID 是否需要更新
                    if hasattr(ds, 'PatientID') and ds.PatientID == new_patient_id:
                         logging.info(f"文件 '{filepath}' 的 PatientID ('{original_id}') 已是目标 ID，跳过。")
                         continue # 如果ID相同，则跳过
                         
                    logging.info(f"更新文件: {filepath} (原 PatientID: {original_id})")

                    # 更新 PatientID
                    ds.PatientID = new_patient_id

                    # 保存更改
                    ds.save_as(filepath)
                    logging.debug(f"  -> 已保存更改")
                    total_updated += 1

                except InvalidDicomError:
                    logging.warning(f"文件 '{filepath}' 不是有效的 DICOM 文件，跳过。")
                except Exception as e:
                    logging.error(f"处理文件 '{filepath}' 时发生错误: {e}")
                    
    return total_checked, total_updated

if __name__ == "__main__":
    # 1. 从 fixed 目录获取 PatientID
    logging.info(f"步骤 1: 从源目录 '{source_dir}' 获取 PatientID")
    new_id = get_patient_id_from_source(source_dir)

    # 2. 更新 moving 目录中的文件
    if new_id:
        logging.info(f"\n步骤 2: 开始更新目标目录 '{target_dir}' 的 PatientID 为 '{new_id}'")
        checked_count, updated_count = update_dicom_patient_id(target_dir, new_id)
        logging.info(f"\n处理完成。共检查 {checked_count} 个 .dcm 文件，成功更新 {updated_count} 个。")
    else:
         logging.error("未能从源目录获取 PatientID，无法执行更新。")
