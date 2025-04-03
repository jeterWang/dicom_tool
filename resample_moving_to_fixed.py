import os
import SimpleITK as sitk
import logging
import traceback
import numpy as np # 确保导入 numpy

# --- 配置 ---
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义输入和输出路径
NIFTI_BASE_DIR = 'output/nifti'
OUTPUT_RESAMPLED_DIR = 'output/nifti'

FIXED_IMAGE_PATH = os.path.join(NIFTI_BASE_DIR, 'fixed', 'img.nii.gz')
MOVING_IMAGE_PATH = os.path.join(NIFTI_BASE_DIR, 'moving', 'img.nii.gz')
MOVING_MASK_PATH = os.path.join(NIFTI_BASE_DIR, 'moving', 'contour_mask.nii.gz')
MOVING_BIO_IMAGE_PATH = os.path.join(NIFTI_BASE_DIR, 'moving_bio', 'img.nii.gz') # 新增

OUTPUT_MOVING_IMAGE_RESAMPLED_PATH = os.path.join(OUTPUT_RESAMPLED_DIR, 'moving', 'img_resampled.nii.gz')
OUTPUT_MOVING_MASK_RESAMPLED_PATH = os.path.join(OUTPUT_RESAMPLED_DIR, 'moving', 'contour_mask_resampled.nii.gz')
OUTPUT_MOVING_BIO_IMAGE_RESAMPLED_PATH = os.path.join(OUTPUT_RESAMPLED_DIR, 'moving_bio', 'img_resampled.nii.gz') # 新增

# 定义插值方法
IMAGE_INTERPOLATOR = sitk.sitkLinear # 图像使用线性插值
MASK_INTERPOLATOR = sitk.sitkNearestNeighbor # 掩码使用最近邻插值

# --- 辅助函数 ---

def get_physical_center(image):
    """计算 SimpleITK 图像的物理中心坐标。"""
    size = np.array(image.GetSize())
    index = (size - 1) / 2.0
    physical_center = image.TransformContinuousIndexToPhysicalPoint(index)
    return physical_center

def resample_image_to_centered_grid(input_image, reference_image_for_grid, center_physical_point, interpolator, output_pixel_type=None):
    """将 input_image 重采样到一个新的网格，该网格属性来自 reference_image_for_grid,
       但其物理中心位于 center_physical_point。"""
    resample_filter = sitk.ResampleImageFilter()

    # --- 计算新的网格原点 --- 
    ref_size = np.array(reference_image_for_grid.GetSize())
    ref_spacing = np.array(reference_image_for_grid.GetSpacing())
    ref_direction = np.array(reference_image_for_grid.GetDirection()).reshape(reference_image_for_grid.GetDimension(), -1)
    ref_center_index = (ref_size - 1) / 2.0
    
    # 计算参考网格中心索引对应的物理坐标偏移 (相对于其原点)
    # physical_offset = ref_spacing * ref_direction * ref_center_index 这是一个简化，需要矩阵乘法
    # 使用 TransformContinuousIndexToPhysicalPoint 计算更准确
    ref_grid_center_phys_if_origin_zero = reference_image_for_grid.TransformContinuousIndexToPhysicalPoint(ref_center_index)
    ref_origin_to_center_vector = np.array(ref_grid_center_phys_if_origin_zero) - np.array(reference_image_for_grid.GetOrigin())
    
    # 新原点 = 期望的物理中心 - 从原点到中心索引的物理向量
    new_origin = np.array(center_physical_point) - ref_origin_to_center_vector
    # --- 结束计算新原点 ---

    # 设置新的网格属性
    resample_filter.SetSize(reference_image_for_grid.GetSize())           # 尺寸来自参考
    resample_filter.SetOutputSpacing(reference_image_for_grid.GetSpacing())     # 间距来自参考
    resample_filter.SetOutputDirection(reference_image_for_grid.GetDirection()) # 方向来自参考
    resample_filter.SetOutputOrigin(new_origin.tolist())                     # 使用计算出的新原点

    # 设置插值方法
    resample_filter.SetInterpolator(interpolator)

    # 设置默认像素值
    resample_filter.SetDefaultPixelValue(input_image.GetPixelIDValue())

    # 设置输出像素类型
    if output_pixel_type is not None:
        resample_filter.SetOutputPixelType(output_pixel_type)
    else:
        resample_filter.SetOutputPixelType(input_image.GetPixelID())

    # --- 使用身份变换 --- 
    # 因为我们已经调整了输出网格的原点来对齐中心
    resample_filter.SetTransform(sitk.Transform()) 
    # --- 结束修改 ---

    # 执行重采样
    resampled_image = resample_filter.Execute(input_image)
    return resampled_image

# --- 主程序 ---
if __name__ == "__main__":
    logging.info("===== 开始将 Moving 图像/掩码重采样到以 Moving 中心定位的 Fixed 空间网格 =====")

    # 检查输入文件是否存在
    input_files_ok = True
    for f_path in [FIXED_IMAGE_PATH, MOVING_IMAGE_PATH, MOVING_MASK_PATH, MOVING_BIO_IMAGE_PATH]:
        if not os.path.exists(f_path):
            logging.error(f"错误: 输入文件不存在: {f_path}")
            input_files_ok = False
    
    if not input_files_ok:
        logging.error("输入文件检查失败，无法继续。请确保 output/nifti 目录中有正确的 img.nii.gz 和 contour_mask.nii.gz 文件。")
        exit(1) # 退出脚本
        
    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_MOVING_IMAGE_RESAMPLED_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_MOVING_MASK_RESAMPLED_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_MOVING_BIO_IMAGE_RESAMPLED_PATH), exist_ok=True) # 新增

    try:
        # 1. 加载图像和掩码
        logging.info(f"加载 Fixed 图像: {FIXED_IMAGE_PATH}")
        fixed_image = sitk.ReadImage(FIXED_IMAGE_PATH)
        logging.info(f"加载 Moving 图像: {MOVING_IMAGE_PATH}")
        moving_image = sitk.ReadImage(MOVING_IMAGE_PATH)
        logging.info(f"加载 Moving 掩码: {MOVING_MASK_PATH}")
        moving_mask = sitk.ReadImage(MOVING_MASK_PATH)
        logging.info(f"加载 Moving Bio 图像: {MOVING_BIO_IMAGE_PATH}") # 新增
        moving_bio_image = sitk.ReadImage(MOVING_BIO_IMAGE_PATH) # 新增
        
        # --- 计算 Moving 和 Moving Bio 图像的物理中心 --- 
        logging.info("计算 Moving 图像物理中心...")
        moving_center = get_physical_center(moving_image)
        logging.info(f"Moving 中心 (物理坐标): {moving_center}")
        logging.info("计算 Moving Bio 图像物理中心...") # 新增
        moving_bio_center = get_physical_center(moving_bio_image) # 新增
        logging.info(f"Moving Bio 中心 (物理坐标): {moving_bio_center}") # 新增
        # --- 结束计算中心 --- 
        
        # --- 可选: 检查掩码类型，确保是整数类型 ---
        if moving_mask.GetPixelIDValue() not in [sitk.sitkUInt8, sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkInt16, sitk.sitkUInt32, sitk.sitkInt32, sitk.sitkUInt64, sitk.sitkInt64]:
             logging.warning(f"警告：Moving 掩码像素类型 ({moving_mask.GetPixelIDValue()}) 不是标准整数类型。如果遇到问题，请考虑预处理掩码。")
        # 确保输出掩码为 uint8
        output_mask_pixel_type = sitk.sitkUInt8
        # --- 结束检查 ---

        # 2. 重采样 Moving 图像到以其自身为中心的 Fixed 网格
        logging.info("重采样 Moving 图像 (到中心对齐的 Fixed 网格)...")
        # --- 修改：调用新的重采样函数 --- 
        moving_image_resampled = resample_image_to_centered_grid(
            input_image=moving_image,
            reference_image_for_grid=fixed_image, # 使用 fixed 定义网格属性
            center_physical_point=moving_center,  # 网格中心对齐 moving 中心
            interpolator=IMAGE_INTERPOLATOR
        )
        # --- 结束修改 ---

        # 3. 重采样 Moving 掩码到以其自身为中心的 Fixed 网格
        logging.info("重采样 Moving 掩码 (到中心对齐的 Fixed 网格)...")
        # --- 修改：调用新的重采样函数 --- 
        moving_mask_resampled = resample_image_to_centered_grid(
            input_image=moving_mask,
            reference_image_for_grid=fixed_image, # 使用 fixed 定义网格属性
            center_physical_point=moving_center,  # 网格中心对齐 moving 中心
            interpolator=MASK_INTERPOLATOR,
            output_pixel_type=output_mask_pixel_type
        )
        # --- 结束修改 ---
        
        # --- 新增：重采样 Moving Bio 图像 --- 
        logging.info("重采样 Moving Bio 图像 (到中心对齐的 Fixed 网格)...")
        moving_bio_image_resampled = resample_image_to_centered_grid(
            input_image=moving_bio_image,
            reference_image_for_grid=fixed_image, # 使用 fixed 定义网格属性
            center_physical_point=moving_bio_center, # 网格中心对齐 moving_bio 中心
            interpolator=IMAGE_INTERPOLATOR
        )
        # --- 结束新增 ---

        # 4. 保存重采样后的文件
        logging.info(f"保存重采样后的 Moving 图像到: {OUTPUT_MOVING_IMAGE_RESAMPLED_PATH}")
        sitk.WriteImage(moving_image_resampled, OUTPUT_MOVING_IMAGE_RESAMPLED_PATH)

        logging.info(f"保存重采样后的 Moving 掩码到: {OUTPUT_MOVING_MASK_RESAMPLED_PATH}")
        sitk.WriteImage(moving_mask_resampled, OUTPUT_MOVING_MASK_RESAMPLED_PATH)
        
        logging.info(f"保存重采样后的 Moving Bio 图像到: {OUTPUT_MOVING_BIO_IMAGE_RESAMPLED_PATH}") # 新增
        sitk.WriteImage(moving_bio_image_resampled, OUTPUT_MOVING_BIO_IMAGE_RESAMPLED_PATH) # 新增

        logging.info("===== 重采样完成 =====")
        
        # --- 新增：删除原始的 moving 和 moving_bio 文件 --- 
        files_to_delete = [
            MOVING_IMAGE_PATH,
            MOVING_MASK_PATH,
            MOVING_BIO_IMAGE_PATH
        ]
        logging.info("开始删除原始 NIfTI 文件...")
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"  已删除: {file_path}")
                else:
                     logging.warning(f"  文件不存在，无法删除: {file_path}")
            except OSError as delete_err:
                logging.error(f"  删除文件时出错: {file_path} - {delete_err}")
        logging.info("原始文件删除完成 (如果存在)。")
        # --- 结束新增 ---

    except Exception as e:
        logging.error("重采样过程中发生错误:")
        traceback.print_exc()
        logging.error("===== 重采样失败 =====") 