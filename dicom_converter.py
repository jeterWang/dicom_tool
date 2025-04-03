import os
import pydicom
import nibabel as nib
import numpy as np
import dicom2nifti
import dicom2nifti.settings as settings
from rt_utils import RTStructBuilder
import logging
import traceback

# --- 配置 ---
# 禁用 dicom2nifti 详细输出 (可选)
settings.disable_validate_slice_increment()
# settings.disable_validate_orthogonal()
# settings.disable_validate_slice_thickness()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义基础路径
INPUT_BASE_DIR = 'images'
OUTPUT_BASE_DIR = 'nifti_output'

# 定义要处理的数据集配置
# 每个字典代表一个数据集 (例如 'fixed', 'moving')
DATASET_CONFIGS = [
    {
        'name': 'fixed',
        'dicom_image_dir': os.path.join(INPUT_BASE_DIR, 'fixed'),
        'dicom_contour_dir': os.path.join(INPUT_BASE_DIR, 'fixed', 'contour'),
        'nifti_image_output_dir': os.path.join(OUTPUT_BASE_DIR, 'fixed'),
        'nifti_image_filename': 'img.nii.gz',
        'nifti_mask_filename': 'contour_mask.nii.gz'
    },
    {
        'name': 'moving',
        'dicom_image_dir': os.path.join(INPUT_BASE_DIR, 'moving'),
        'dicom_contour_dir': os.path.join(INPUT_BASE_DIR, 'moving', 'contour'),
        'nifti_image_output_dir': os.path.join(OUTPUT_BASE_DIR, 'moving'),
        'nifti_image_filename': 'img.nii.gz',
        'nifti_mask_filename': 'contour_mask.nii.gz'
    },
    {
        'name': 'moving_bio',
        'dicom_image_dir': os.path.join(INPUT_BASE_DIR, 'moving_bio'),
        'dicom_contour_dir': None,
        'nifti_image_output_dir': os.path.join(OUTPUT_BASE_DIR, 'moving_bio'),
        'nifti_image_filename': 'img.nii.gz',
        'nifti_mask_filename': None
    }
]

# --- 辅助函数 ---

def convert_image_series(dicom_dir, output_dir, desired_filename):
    """将 DICOM 图像序列转换为 NIfTI 文件，并重命名。"""
    logging.info(f"--- 开始处理图像序列: {dicom_dir} ---")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"期望文件名: {desired_filename}")

    if not os.path.isdir(dicom_dir):
        logging.error(f"错误: 图像目录 {dicom_dir} 不存在，跳过。")
        return None # 返回 None 表示失败

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    target_filepath = os.path.join(output_dir, desired_filename)

    try:
        # 记录转换前的文件列表
        files_before = set(os.listdir(output_dir))

        # 执行转换 (不进行重定向)
        dicom2nifti.convert_directory(dicom_dir, output_dir, compression=True, reorient=False)
        logging.info(f"dicom2nifti 转换完成: {dicom_dir} -> {output_dir}")

        # 记录转换后的文件列表
        files_after = set(os.listdir(output_dir))

        # 找出新增的文件
        new_files = files_after - files_before
        nifti_files_generated = [f for f in new_files if f.endswith('.nii.gz') or f.endswith('.nii')] # 允许 .nii

        # 重命名逻辑
        if len(nifti_files_generated) == 1:
            generated_filename = nifti_files_generated[0]
            generated_filepath = os.path.join(output_dir, generated_filename)

            if generated_filepath == target_filepath:
                logging.info(f"生成的文件名已是目标名称: {target_filepath}")
                return target_filepath # 返回成功生成的文件路径
            else:
                if os.path.exists(target_filepath):
                    logging.warning(f"警告：目标文件 {target_filepath} 已存在，将被覆盖。")
                    try:
                        os.remove(target_filepath)
                    except OSError as e:
                         logging.error(f"错误：无法删除已存在的目标文件 {target_filepath}: {e}")
                         return None # 返回 None 表示失败

                try:
                    os.rename(generated_filepath, target_filepath)
                    logging.info(f"已将 {generated_filepath} 重命名为 {target_filepath}")
                    return target_filepath # 返回成功生成的文件路径
                except OSError as e:
                     logging.error(f"错误：重命名文件 {generated_filepath} 到 {target_filepath} 失败: {e}")
                     return None # 返回 None 表示失败

        elif len(nifti_files_generated) > 1:
            logging.warning(f"警告：在目录 {output_dir} 中检测到多个新生成的 NIfTI 文件，无法自动重命名: {nifti_files_generated}")
            # 尝试查找是否已存在期望的文件名
            if os.path.exists(target_filepath):
                 logging.warning(f"警告：虽然生成了多个文件，但目标文件 {target_filepath} 已存在。假设使用此文件。")
                 return target_filepath
            return None # 无法确定，返回失败

        else: # len(nifti_files_generated) == 0
            if os.path.exists(target_filepath):
                logging.info(f"未找到新生成的NIfTI文件，但目标文件 {target_filepath} 已存在。假设转换成功。")
                return target_filepath
            else:
                logging.error(f"错误：在目录 {output_dir} 中未找到新生成的 NIfTI 文件，且目标文件 {target_filepath} 不存在。转换失败。")
                return None

    except Exception as e:
        logging.error(f"处理图像目录 {dicom_dir} 时发生严重错误: {e}")
        traceback.print_exc()
        return None

def convert_contour_mask(rtstruct_dir, dicom_image_ref_dir, reference_nifti_path, output_mask_path):
    """将 DICOM RTSTRUCT 转换为 NIfTI 掩码文件，使用特定变换。"""
    logging.info(f"--- 开始处理轮廓: {rtstruct_dir} ---")
    logging.info(f"参考 NIfTI 图像: {reference_nifti_path}")
    logging.info(f"输出掩码路径: {output_mask_path}")

    if not os.path.isdir(rtstruct_dir):
        logging.error(f"错误: 轮廓目录 {rtstruct_dir} 不存在，跳过。")
        return False

    if not os.path.exists(reference_nifti_path):
        logging.error(f"错误: 参考 NIfTI 图像 {reference_nifti_path} 不存在，跳过轮廓转换。")
        return False

    # 查找 RTSTRUCT 文件
    rtstruct_file = None
    for filename in os.listdir(rtstruct_dir):
        filepath = os.path.join(rtstruct_dir, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.dcm'):
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                if hasattr(dcm, 'Modality') and dcm.Modality == 'RTSTRUCT':
                    if rtstruct_file is not None:
                        logging.warning(f"警告: 目录 {rtstruct_dir} 找到多个 RTSTRUCT 文件，使用第一个: {rtstruct_file}")
                    else:
                        rtstruct_file = filepath
                        logging.info(f"找到 RTSTRUCT 文件: {rtstruct_file}")
            except Exception as e:
                logging.warning(f"读取文件 {filepath} 时出错: {e}")

    if rtstruct_file is None:
        logging.error(f"错误: 在目录 {rtstruct_dir} 未找到有效的 DICOM RTSTRUCT 文件，跳过。")
        return False

    try:
        # 1. 加载 RTSTRUCT 和参考图像
        rtstruct = RTStructBuilder.create_from(dicom_series_path=dicom_image_ref_dir,
                                               rt_struct_path=rtstruct_file)
        reference_image = nib.load(reference_nifti_path)

        # 2. 获取 ROI 名称并合并掩码
        roi_names = rtstruct.get_roi_names()
        logging.info(f"RTSTRUCT 中的 ROI: {roi_names}")
        if not roi_names:
            logging.warning(f"警告：文件 {rtstruct_file} 中无 ROI，生成空掩码。")
            base_mask_data = np.zeros(reference_image.shape, dtype=np.uint8)
        else:
            combined_mask_data_bool = np.zeros(reference_image.shape, dtype=bool)
            logging.info(f"合并 ROI 掩码: {roi_names}")
            for roi_name in roi_names:
                try:
                    roi_mask = rtstruct.get_roi_mask_by_name(roi_name)
                    if roi_mask.shape != reference_image.shape:
                        logging.warning(f"警告：ROI '{roi_name}' 形状 {roi_mask.shape} 与参考图像 {reference_image.shape} 不匹配，跳过。")
                        continue
                    combined_mask_data_bool |= roi_mask
                    logging.info(f"  合并了 ROI '{roi_name}'")
                except Exception as roi_e:
                     logging.error(f"  获取或合并 ROI '{roi_name}' 时出错: {roi_e}")
            base_mask_data = combined_mask_data_bool.astype(np.uint8)
            logging.info("ROI 掩码合并完成。")

        # 3. 执行最终的方向变换 (perm102, flip1)
        logging.info("执行最终方向变换 (perm102, flip1)...")
        final_mask_data = np.transpose(base_mask_data, axes=(1, 0, 2))
        

        # 4. 检查最终形状
        if final_mask_data.shape != reference_image.shape:
             logging.error(f"错误：最终掩码形状 {final_mask_data.shape} 与参考图像 {reference_image.shape} 不匹配！无法保存。")
             return False

        # 5. 创建并保存 NIfTI 文件
        mask_nifti = nib.Nifti1Image(final_mask_data, reference_image.affine, reference_image.header)
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True) # 确保输出目录存在
        nib.save(mask_nifti, output_mask_path)
        logging.info(f"成功保存方向正确的掩码: {output_mask_path}")
        return True

    except Exception as e:
        logging.error(f"处理 RTSTRUCT 文件 {rtstruct_file} 时发生严重错误: {e}")
        traceback.print_exc()
        return False

# --- 主程序 ---
if __name__ == "__main__":
    logging.info("===== 开始 DICOM 到 NIfTI 转换任务 =====")

    # 确保整体输出目录存在
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    successful_datasets = 0
    failed_datasets = 0

    for config in DATASET_CONFIGS:
        dataset_name = config['name']
        logging.info(f"===== 处理数据集: {dataset_name} =====")

        # 1. 转换图像序列
        image_output_path = os.path.join(config['nifti_image_output_dir'], config['nifti_image_filename'])
        generated_image_path = convert_image_series(
            dicom_dir=config['dicom_image_dir'],
            output_dir=config['nifti_image_output_dir'],
            desired_filename=config['nifti_image_filename']
        )

        # 2. 如果图像转换成功，则转换轮廓
        mask_conversion_success = False
        contour_dir = config.get('dicom_contour_dir')
        mask_filename = config.get('nifti_mask_filename')

        if generated_image_path and contour_dir and mask_filename:
            logging.info(f"图像转换成功: {generated_image_path}，开始处理轮廓...")
            mask_output_path = os.path.join(config['nifti_image_output_dir'], mask_filename)
            mask_conversion_success = convert_contour_mask(
                rtstruct_dir=contour_dir,
                dicom_image_ref_dir=config['dicom_image_dir'], # RTStructBuilder 仍需原始图像目录
                reference_nifti_path=generated_image_path,     # 使用刚生成的 NIfTI 做参考
                output_mask_path=mask_output_path
            )
            if mask_conversion_success:
                logging.info(f"轮廓掩码转换成功: {mask_output_path}")
            else:
                logging.error(f"轮廓掩码转换失败: {contour_dir}")
        else:
            logging.error(f"图像转换失败，跳过 {dataset_name} 的轮廓转换。")
            if generated_image_path and not contour_dir:
                mask_conversion_success = True # 没有轮廓需要转换，图像成功即可
                logging.info(f"数据集 {dataset_name} 只包含图像，图像转换成功。")

        # 判断成功的条件
        dataset_success = (generated_image_path is not None) and (mask_conversion_success if contour_dir else True)

        if dataset_success:
            successful_datasets += 1
            logging.info(f"===== 数据集 {dataset_name} 处理成功 =====")
        else:
            failed_datasets += 1
            logging.error(f"===== 数据集 {dataset_name} 处理失败 =====")


    logging.info("===== 转换任务总结 =====")
    logging.info(f"成功处理数据集: {successful_datasets}")
    logging.info(f"失败处理数据集: {failed_datasets}")
    logging.info("==========================") 