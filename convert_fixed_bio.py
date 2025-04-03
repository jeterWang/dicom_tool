import os
import SimpleITK as sitk
import logging
import traceback
import numpy as np

# --- 配置 ---
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义输入和输出路径
INPUT_DICOM_DIR = 'images/fixed_bio'
REFERENCE_NIFTI_PATH = 'nifti_output/fixed/img.nii.gz' # 参考空间
OUTPUT_NIFTI_DIR = 'nifti_output/fixed_bio'
OUTPUT_NIFTI_FILENAME = 'img.nii.gz'
OUTPUT_NIFTI_PATH = os.path.join(OUTPUT_NIFTI_DIR, OUTPUT_NIFTI_FILENAME)

# 插值方法
INTERPOLATOR = sitk.sitkLinear

# --- 主程序 ---
if __name__ == "__main__":
    logging.info(f"===== 开始转换 DICOM 序列 '{INPUT_DICOM_DIR}' 并重采样到参考空间 =====")

    # 1. 检查输入和参考文件/目录是否存在
    if not os.path.isdir(INPUT_DICOM_DIR):
        logging.error(f"错误: 输入 DICOM 目录不存在: {INPUT_DICOM_DIR}")
        exit(1)
    if not os.path.exists(REFERENCE_NIFTI_PATH):
        logging.error(f"错误: 参考 NIfTI 文件不存在: {REFERENCE_NIFTI_PATH}")
        exit(1)

    # 2. 创建输出目录
    os.makedirs(OUTPUT_NIFTI_DIR, exist_ok=True)

    try:
        # 3. 读取参考 NIfTI 图像以获取空间信息
        logging.info(f"加载参考 NIfTI 图像: {REFERENCE_NIFTI_PATH}")
        reference_image = sitk.ReadImage(REFERENCE_NIFTI_PATH)
        logging.info(f"参考图像信息: Size={reference_image.GetSize()}, Spacing={reference_image.GetSpacing()}, Origin={reference_image.GetOrigin()}, Direction={reference_image.GetDirection()}")

        # 4. 读取输入的 DICOM 序列
        logging.info(f"读取 DICOM 序列从目录: {INPUT_DICOM_DIR}")
        reader = sitk.ImageSeriesReader()
        try:
            dicom_names = reader.GetGDCMSeriesFileNames(INPUT_DICOM_DIR)
            if not dicom_names:
                 logging.error(f"错误：在目录 {INPUT_DICOM_DIR} 中未找到 DICOM 文件或序列。")
                 exit(1)
            reader.SetFileNames(dicom_names)
            input_image = reader.Execute()
        except RuntimeError as read_err:
            logging.error(f"读取 DICOM 序列时出错: {read_err}")
            # 尝试读取为单个文件（如果目录只有一个文件且非序列）
            if len(os.listdir(INPUT_DICOM_DIR)) == 1 and os.listdir(INPUT_DICOM_DIR)[0].lower().endswith('.dcm'):
                 single_file = os.path.join(INPUT_DICOM_DIR, os.listdir(INPUT_DICOM_DIR)[0])
                 logging.warning(f"尝试读取为单个 DICOM 文件: {single_file}")
                 input_image = sitk.ReadImage(single_file)
            else:
                 raise read_err # 如果不是单个文件或读取仍失败，重新抛出错误
        logging.info(f"成功读取输入 DICOM 图像: Size={input_image.GetSize()}, Spacing={input_image.GetSpacing()}, Origin={input_image.GetOrigin()}, Direction={input_image.GetDirection()}")
        
        # 5. 配置并执行重采样
        logging.info("开始重采样...")
        resample_filter = sitk.ResampleImageFilter()
        
        # 使用参考图像设置输出空间
        resample_filter.SetReferenceImage(reference_image)
        
        # 设置插值器
        resample_filter.SetInterpolator(INTERPOLATOR)
        
        # 使用身份变换 (假设在同一空间)
        resample_filter.SetTransform(sitk.Transform()) 
        
        # 设置默认像素值 (可选, 通常设为背景值, 如 0)
        # 尝试获取输入图像像素类型的最小值，如果失败则用0
        try: 
             default_value = sitk.GetImageFromArray(np.array(0, input_image.GetPixelID().lower())).GetPixelIDValue()
        except:
             default_value = 0
        resample_filter.SetDefaultPixelValue(default_value)
        logging.info(f"设置默认像素值为: {default_value}")
        
        # 执行重采样
        resampled_image = resample_filter.Execute(input_image)
        logging.info(f"重采样完成。输出图像信息: Size={resampled_image.GetSize()}, Spacing={resampled_image.GetSpacing()}, Origin={resampled_image.GetOrigin()}, Direction={resampled_image.GetDirection()}")

        # 6. 保存结果
        logging.info(f"保存重采样后的 NIfTI 文件到: {OUTPUT_NIFTI_PATH}")
        sitk.WriteImage(resampled_image, OUTPUT_NIFTI_PATH)

        logging.info(f"===== 转换和重采样成功完成 =====")

    except Exception as e:
        logging.error("处理过程中发生错误:")
        traceback.print_exc()
        logging.error("===== 转换和重采样失败 =====") 