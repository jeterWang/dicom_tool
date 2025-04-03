import os
import pydicom
from pydicom.errors import InvalidDicomError

# 定义源目录和目标目录
source_dir = 'images/week4_CT'
target_dir = 'images/week0_CT'

def get_patient_id_from_source(directory):
    """从源目录的第一个 DICOM 文件中读取 PatientID"""
    for filename in os.listdir(directory):
        if filename.lower().endswith('.dcm'):
            filepath = os.path.join(directory, filename)
            try:
                ds = pydicom.dcmread(filepath)
                patient_id = ds.PatientID
                print(f"从文件 {filename} 读取到 PatientID: {patient_id}")
                return patient_id
            except InvalidDicomError:
                print(f"警告: 文件 {filename} 不是有效的 DICOM 文件，跳过。")
            except AttributeError:
                print(f"警告: 文件 {filename} 没有 PatientID 标签，跳过。")
            except Exception as e:
                print(f"读取文件 {filename} 时发生错误: {e}")
    print(f"错误: 在目录 {directory} 中未找到有效的 DICOM 文件或 PatientID。")
    return None

def update_dicom_patient_id(target_directory, new_patient_id):
    """更新目标目录中所有 DICOM 文件的 PatientID"""
    if new_patient_id is None:
        print("错误: 未提供有效的 PatientID，无法更新。")
        return

    count = 0
    updated_count = 0
    for filename in os.listdir(target_directory):
        if filename.lower().endswith('.dcm'):
            filepath = os.path.join(target_directory, filename)
            count += 1
            try:
                ds = pydicom.dcmread(filepath)
                original_id = ds.PatientID if 'PatientID' in ds else '不存在'
                print(f"处理文件: {filename} (原 PatientID: {original_id})")

                # 更新 PatientID
                ds.PatientID = new_patient_id

                # 保存更改
                ds.save_as(filepath)
                print(f"  -> 已更新 PatientID 为: {new_patient_id}")
                updated_count += 1

            except InvalidDicomError:
                print(f"  -> 警告: 文件 {filename} 不是有效的 DICOM 文件，跳过。")
            except Exception as e:
                print(f"  -> 处理文件 {filename} 时发生错误: {e}")

    print(f"\n处理完成。共检查 {count} 个 .dcm 文件，成功更新 {updated_count} 个。")

if __name__ == "__main__":
    # 1. 从 week4_CT 获取 PatientID
    new_id = get_patient_id_from_source(source_dir)

    # 2. 更新 week0_CT 中的文件
    if new_id:
        update_dicom_patient_id(target_dir, new_id)
