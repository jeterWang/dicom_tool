import os
import pydicom
import SimpleITK as sitk
import numpy as np
import logging
import traceback
from pydicom.uid import generate_uid
from pydicom.sequence import Sequence
from pydicom.dataset import Dataset, FileMetaDataset
import datetime
from rt_utils import RTStructBuilder
from rt_utils.rtstruct import RTStruct
import itertools # 确保导入 itertools

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 输入路径 (来自之前的脚本输出)
NIFTI_INPUT_DIR = 'output/nifti'
RESAMPLED_MOVING_IMAGE_PATH = os.path.join(NIFTI_INPUT_DIR, 'moving', 'img_resampled.nii.gz')
RESAMPLED_MOVING_MASK_PATH = os.path.join(NIFTI_INPUT_DIR, 'moving', 'contour_mask_resampled.nii.gz')

# 原始 DICOM 模板路径
ORIGINAL_DICOM_IMAGE_DIR = 'images/moving/images' # 主要用于获取非空间元数据

# 新 DICOM 输出路径
DICOM_OUTPUT_DIR = 'output/dicom_from_nifti'
NEW_DICOM_IMAGE_OUTPUT_DIR = os.path.join(DICOM_OUTPUT_DIR, 'moving', 'images')
NEW_RTSTRUCT_OUTPUT_PATH = os.path.join(DICOM_OUTPUT_DIR, 'moving', 'rtstruct.dcm') # 新增 RTSTRUCT 输出路径

# 掩码灰度值
# MASK_FOREGROUND_VALUE = 1000 # 不再需要，因为我们直接用布尔掩码

# --- 辅助函数 ---

# --- 修改：添加空间参数 --- 
def create_dicom_dataset(
    slice_data,
    template_ds,
    series_instance_uid,
    sop_instance_uid,
    instance_number,
    image_position_patient,
    image_orientation_patient,
    pixel_spacing,
    slice_thickness
):
# --- 结束修改 ---
    """根据模板创建并填充新的 DICOM 数据集，使用计算出的空间信息。"""
    new_ds = pydicom.dcmread(template_ds.filename, force=True) # 重新读取以获取完整副本
    
    # 确保文件元信息存在且被复制 (逻辑不变)
    if not hasattr(new_ds, 'file_meta') or new_ds.file_meta is None:
         new_ds.file_meta = FileMetaDataset()
         new_ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
         new_ds.file_meta.MediaStorageSOPClassUID = template_ds.SOPClassUID 
         logging.warning("模板缺少 file_meta，已创建基本元数据。")
    else:
        new_ds.file_meta = FileMetaDataset(template_ds.file_meta)

    # -- 更新关键元数据 (UIDs, InstanceNumber) (逻辑不变) --
    new_ds.SOPInstanceUID = sop_instance_uid
    new_ds.SeriesInstanceUID = series_instance_uid
    new_ds.InstanceNumber = instance_number
    new_ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid 

    # -- 更新像素数据相关元数据 (逻辑不变) --
    new_ds.Rows = slice_data.shape[0]
    new_ds.Columns = slice_data.shape[1]
    if slice_data.dtype == np.uint8 or slice_data.dtype == np.uint16:
        new_ds.PixelRepresentation = 0
        bits_allocated = slice_data.itemsize * 8
    elif slice_data.dtype == np.int8 or slice_data.dtype == np.int16:
        new_ds.PixelRepresentation = 1
        bits_allocated = slice_data.itemsize * 8
    else:
        logging.warning(f"输入切片数据类型为 {slice_data.dtype}，将尝试转换为 int16。")
        slice_data = slice_data.astype(np.int16)
        new_ds.PixelRepresentation = 1
        bits_allocated = 16
    new_ds.BitsAllocated = bits_allocated
    new_ds.BitsStored = bits_allocated
    new_ds.HighBit = bits_allocated - 1
    if 'ImageType' in new_ds:
         if isinstance(new_ds.ImageType, str):
              if 'DERIVED' not in new_ds.ImageType:
                   new_ds.ImageType = [new_ds.ImageType, 'DERIVED', 'SECONDARY']
         elif isinstance(new_ds.ImageType, list):
             if 'DERIVED' not in new_ds.ImageType:
                  new_ds.ImageType.extend(['DERIVED', 'SECONDARY'])
    else:
         new_ds.ImageType = ['DERIVED', 'SECONDARY']
    new_ds.PhotometricInterpretation = "MONOCHROME2"
    new_ds.RescaleSlope = '1'
    new_ds.RescaleIntercept = '0'

    # --- 修改：使用传入的空间参数覆盖模板值 --- 
    new_ds.ImagePositionPatient = [f"{v:.6f}" for v in image_position_patient] # [x, y, z]
    new_ds.ImageOrientationPatient = [f"{v:.6f}" for v in image_orientation_patient] # [Rx, Ry, Rz, Cx, Cy, Cz]
    new_ds.PixelSpacing = [f"{v:.6f}" for v in pixel_spacing] # [RowSpacing, ColSpacing]
    new_ds.SliceThickness = f"{slice_thickness:.6f}" 
    # 尝试计算 SliceLocation (沿法向量的位置)
    try:
         slice_normal = np.cross(np.array(image_orientation_patient[0:3]), np.array(image_orientation_patient[3:6]))
         slice_location = np.dot(np.array(image_position_patient), slice_normal)
         new_ds.SliceLocation = f"{slice_location:.6f}"
    except Exception as sl_err:
         logging.warning(f"计算 SliceLocation 时出错: {sl_err}")
         if 'SliceLocation' in new_ds: del new_ds.SliceLocation # 移除模板值以防冲突
    # --- 结束修改 ---

    # --- 修改：移除可能冲突的 Window Center/Width (让计算值优先) ---
    if 'WindowCenter' in new_ds: del new_ds.WindowCenter
    if 'WindowWidth' in new_ds: del new_ds.WindowWidth
    # --- 结束修改 ---
    
    # --- 新增：根据 slice_data 计算并设置 Window Center/Width --- 
    try:
        finite_data = slice_data[np.isfinite(slice_data)]
        if finite_data.size > 0: 
             p5 = np.percentile(finite_data, 5)
             p95 = np.percentile(finite_data, 95)
             wc = (p95 + p5) / 2.0
             ww = max(p95 - p5, 1)
             new_ds.WindowCenter = f"{wc:.1f}" 
             new_ds.WindowWidth = f"{ww:.1f}" 
             logging.debug(f"根据数据计算并设置 WC={new_ds.WindowCenter}, WW={new_ds.WindowWidth}")
        else:
             logging.warning("切片中无有效数据计算窗位。")
    except Exception as wcww_e:
         logging.error(f"计算或设置 WindowCenter/Width 时出错: {wcww_e}")
    # --- 结束新增 ---
    
    # -- 写入像素数据 -- (逻辑不变)
    new_ds.PixelData = slice_data.tobytes()
    
    # -- 添加必要的 DICOM 前导码 -- (逻辑不变)
    new_ds.preamble = b"\0" * 128

    return new_ds

# --- 主程序 ---
if __name__ == "__main__":
    logging.info("===== 开始将 NIfTI 转换回 DICOM 序列 (图像) 和 RTSTRUCT (轮廓) =====") # 更新标题

    # 1. 检查输入文件是否存在
    input_files_ok = True
    # --- 修改：现在只需要检查图像NIfTI, 掩码NIfTI, 和原始DICOM目录 ---
    for f_path in [RESAMPLED_MOVING_IMAGE_PATH, RESAMPLED_MOVING_MASK_PATH, ORIGINAL_DICOM_IMAGE_DIR]:
    # --- 结束修改 ---
        if not os.path.exists(f_path):
            logging.error(f"错误: 输入路径不存在: {f_path}")
            input_files_ok = False
    if not os.path.isdir(ORIGINAL_DICOM_IMAGE_DIR):
        logging.error(f"错误: 原始 DICOM 目录不是有效目录: {ORIGINAL_DICOM_IMAGE_DIR}")
        input_files_ok = False
        
    if not input_files_ok:
        logging.error("输入检查失败，无法继续。")
        exit(1)
        
    # 2. 创建输出目录
    os.makedirs(NEW_DICOM_IMAGE_OUTPUT_DIR, exist_ok=True)
    # os.makedirs(NEW_DICOM_MASK_OUTPUT_DIR, exist_ok=True) # 不再需要掩码序列目录
    os.makedirs(os.path.dirname(NEW_RTSTRUCT_OUTPUT_PATH), exist_ok=True) # 确保 RTSTRUCT 目录存在

    try:
        # 3. 加载 NIfTI 图像和掩码 (使用 SimpleITK)
        logging.info(f"加载 NIfTI 图像: {RESAMPLED_MOVING_IMAGE_PATH}")
        sitk_image = sitk.ReadImage(RESAMPLED_MOVING_IMAGE_PATH)
        image_array = sitk.GetArrayFromImage(sitk_image) # Shape: (Z, Y, X)
        logging.info(f"  图像数组形状: {image_array.shape}, 数据类型: {image_array.dtype}")

        logging.info(f"加载 NIfTI 掩码: {RESAMPLED_MOVING_MASK_PATH}")
        sitk_mask = sitk.ReadImage(RESAMPLED_MOVING_MASK_PATH)
        mask_array = sitk.GetArrayFromImage(sitk_mask) # Shape: (Z, Y, X)
        logging.info(f"  掩码数组形状: {mask_array.shape}, 数据类型: {mask_array.dtype}")

        if image_array.shape[0] != mask_array.shape[0]:
             logging.error("NIfTI 图像和掩码切片数不一致！")
             exit(1)
        num_slices = image_array.shape[0]

        # 4. 获取第一个 DICOM 模板文件 (逻辑不变)
        logging.info(f"读取原始 DICOM 文件名从: {ORIGINAL_DICOM_IMAGE_DIR}")
        reader = sitk.ImageSeriesReader()
        try:
            original_dicom_filenames = reader.GetGDCMSeriesFileNames(ORIGINAL_DICOM_IMAGE_DIR)
        except RuntimeError as e:
             logging.error(f"无法从 {ORIGINAL_DICOM_IMAGE_DIR} 读取 DICOM 序列文件名: {e}")
             exit(1)
        if not original_dicom_filenames:
            logging.error(f"在模板目录 {ORIGINAL_DICOM_IMAGE_DIR} 中未找到 DICOM 文件！")
            exit(1)
        first_template_filepath = original_dicom_filenames[0]
        logging.info(f"找到 {len(original_dicom_filenames)} 个原始 DICOM 文件，将只使用第一个文件 '{first_template_filepath}' 作为模板。")
        try:
             template_ds_for_all = pydicom.dcmread(first_template_filepath, force=True)
             target_dtype = np.dtype(template_ds_for_all.pixel_array.dtype)
             logging.info(f"将使用模板 DICOM 的数据类型: {target_dtype}")
        except Exception as e:
             logging.error(f"无法读取通用 DICOM 模板 '{first_template_filepath}': {e}")
             exit(1)

        # --- 新增：从 NIfTI 提取通用空间信息 --- 
        spacing = sitk_image.GetSpacing() # (Sx, Sy, Sz)
        direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
        # DICOM PixelSpacing 是 [RowSpacing, ColSpacing]
        # 通常对应 NIfTI Y 轴和 X 轴间距
        pixel_spacing = [spacing[1], spacing[0]] 
        slice_thickness = spacing[2]
        # DICOM ImageOrientationPatient 是 [Rx, Ry, Rz, Cx, Cy, Cz]
        # 通常对应 NIfTI X 和 Y 方向向量 (前两列)
        iop = direction[:, 0].tolist() + direction[:, 1].tolist() 
        logging.info(f"从 NIfTI 提取: PixelSpacing={pixel_spacing}, SliceThickness={slice_thickness}, IOP={iop}")
        # --- 结束新增 ---
        
        # 5. 处理图像 NIfTI -> DICOM
        logging.info("--- 开始转换 NIfTI 图像到 DICOM --- ")
        new_image_series_uid = generate_uid() 
        logging.info(f"新图像序列的 SeriesInstanceUID: {new_image_series_uid}")
        image_conversion_successful = True # 假设成功
        for i in range(num_slices):
            instance_number = i + 1
            slice_data = image_array[i, :, :].astype(target_dtype) 
            output_filename = f"slice_{instance_number:04d}.dcm"
            output_filepath = os.path.join(NEW_DICOM_IMAGE_OUTPUT_DIR, output_filename)
            new_sop_instance_uid = generate_uid()
            
            # --- 新增：计算当前切片的 ImagePositionPatient --- 
            # NIfTI/SITK index is (x, y, z), NumPy is (z, y, x)
            # 我们需要 NIfTI 的 (0, 0, i) 对应的物理坐标
            # SimpleITK 的 GetArrayFromImage 是 ZYX 顺序，所以 NumPy 的第 i slice 对应 SITK 的第 i Z-index
            # 我们需要 NIfTI (0,0,i) 的物理坐标, 假设 NIfTI index 是 (x,y,z) 顺序
            # 因此，需要转换 NumPy 索引 (i, 0, 0) -> SITK 索引 (0, 0, i)
            # 注意 SimpleITK 的 TransformIndexToPhysicalPoint 期望 (x, y, z) 顺序的索引
            index_ijk = [0, 0, i] # 假设切片第一个像素的 NIfTI 索引
            ipp = sitk_image.TransformIndexToPhysicalPoint(index_ijk)
            logging.debug(f"图像切片 {instance_number}: Index={index_ijk} -> IPP={ipp}")
            # --- 结束新增 ---
            
            logging.debug(f"处理图像切片 {instance_number}/{num_slices}, SOP UID: {new_sop_instance_uid}")
            try:
                 # --- 修改：传递空间参数 --- 
                 new_ds = create_dicom_dataset(
                     slice_data,
                     template_ds_for_all,
                     new_image_series_uid,
                     new_sop_instance_uid,
                     instance_number,
                     ipp, iop, pixel_spacing, slice_thickness # 传递计算出的空间信息
                 )
                 # --- 结束修改 ---
                 new_ds.save_as(output_filepath, enforce_file_format=True)
            except Exception as e:
                 logging.error(f"保存图像切片 {instance_number} 到 {output_filepath} 时出错: {e}")
                 traceback.print_exc()
                 image_conversion_successful = False # 标记失败
                 # 这里可以选择是 break 还是 continue，取决于是否要尝试保存剩余切片
                 break # 如果一个切片失败，可能整个序列都有问题，终止图像转换
        
        if image_conversion_successful:
             logging.info("NIfTI 图像到 DICOM 序列转换完成。")
        else:
             logging.error("NIfTI 图像到 DICOM 序列转换过程中发生错误，后续 RTSTRUCT 可能无法生成。")

        # --- 新增：第 6 节：从 NIfTI 掩码生成 RTSTRUCT (使用 NIfTI 作为参考) --- 
        logging.info("--- 开始从 NIfTI 掩码生成 DICOM RTSTRUCT (关联到新生成的 DICOM 序列) --- ")
        if not image_conversion_successful:
            logging.warning("由于图像转换失败，跳过 RTSTRUCT 生成。")
        elif np.sum(mask_array) == 0: 
             logging.warning("NIfTI 掩码为空 (全零)，不生成 RTSTRUCT 文件。")
        else:
            # 先创建 RTStructBuilder 实例
            rtstruct = None
            try:
                # logging.info(f"创建 RTStructBuilder，使用 NIfTI 图像作为参考: {RESAMPLED_MOVING_IMAGE_PATH}") # 错误的方式
                # rtstruct = RTStructBuilder.create_from(imaging_path=RESAMPLED_MOVING_IMAGE_PATH) # 错误的方式
                # --- 修改：恢复使用 create_new --- 
                logging.info(f"创建 RTStructBuilder，引用新生成的 DICOM 序列: {NEW_DICOM_IMAGE_OUTPUT_DIR}")
                rtstruct = RTStructBuilder.create_new(dicom_series_path=NEW_DICOM_IMAGE_OUTPUT_DIR)
                # --- 结束修改 ---
            except ImportError:
                 logging.error("错误: 无法导入 rt-utils。请确保已安装： pip install rt-utils")
                 traceback.print_exc()
            except Exception as builder_e:
                 logging.error(f"创建 RTStructBuilder 时发生错误: {builder_e}")
                 traceback.print_exc()

            if rtstruct:
                # 获取只包含值为 1 的布尔掩码 (ZYX 顺序)
                logging.info("提取掩码中值为 1 的像素...")
                mask_array_bool = (mask_array == 1)
                
                # --- 新增：检查布尔掩码是否为空 ---
                if not np.any(mask_array_bool):
                    logging.warning("布尔掩码 (mask_array == 1) 中不包含任何 True 值！跳过 ROI 添加。")
                else:
                    logging.info(f"布尔掩码包含 {np.sum(mask_array_bool)} 个 True 体素。继续准备 ROI。")
                    # --- 恢复：8 种可能性逻辑 --- 
                    # --- 修改：只尝试 YXZ (Row, Col, Slice) 及其 Row/Col 翻转 (共 4 种) --- 
                    
                    # 1. 准备基础的 YXZ (Row, Col, Slice) 置换掩码
                    mask_yxz = None
                    # mask_xyz = None # 不再需要 XYZ
                    try:
                        axes_yxz = (1, 2, 0) # YXZ 对应 Row, Col, Slice
                        logging.info(f"准备基础 YXZ (Row, Col, Slice) 置换掩码 {axes_yxz}...")
                        mask_yxz = np.transpose(mask_array_bool, axes=axes_yxz)
                        logging.debug(f"    YXZ (Row, Col, Slice) 形状: {mask_yxz.shape}")
                    except Exception as e_prep_yxz:
                        logging.error(f"准备 YXZ 掩码时出错: {e_prep_yxz}")
                    
                    # try:
                    #     axes_xyz = (2, 1, 0)
                    #     logging.info(f"准备基础 XYZ 置换掩码 {axes_xyz}...")
                    #     mask_xyz = np.transpose(mask_array_bool, axes=axes_xyz)
                    #     logging.debug(f"    XYZ 形状: {mask_xyz.shape}")
                    # except Exception as e_prep_xyz:
                    #     logging.error(f"准备 XYZ 掩码时出错: {e_prep_xyz}")

                    # 2. 尝试添加所有可能的 ROI (基础 + 翻转)
                    # --- 修改：只尝试 YXZ 基础及其 Row/Col 翻转 --- 
                    roi_attempts = []
                    if mask_yxz is not None:
                        roi_attempts.extend([
                            ("Contour_YXZ_Base", mask_yxz),
                            # ("Contour_YXZ_FlipRow", np.flip(mask_yxz, axis=0)), # YXZ 的 Row 轴是 axis 0
                            # ("Contour_YXZ_FlipCol", np.flip(mask_yxz, axis=1)), # YXZ 的 Col 轴是 axis 1
                            # ("Contour_YXZ_FlipRowCol", np.flip(np.flip(mask_yxz, axis=0), axis=1)), # 双重翻转
                            # ("Contour_YXZ_flipZ", np.flip(mask_yxz, axis=2)), # 不再包含 Z 翻转
                        ])
                    # if mask_xyz is not None:
                    #      roi_attempts.extend([
                    #         ("Contour_XYZ", mask_xyz),
                    #         ("Contour_XYZ_flipX", np.flip(mask_xyz, axis=0)), # XYZ 的 X 轴是 axis 0
                    #         ("Contour_XYZ_flipY", np.flip(mask_xyz, axis=1)), # XYZ 的 Y 轴是 axis 1
                    #         ("Contour_XYZ_flipZ", np.flip(mask_xyz, axis=2)), # XYZ 的 Z 轴是 axis 2
                    #     ])

                    logging.info(f"尝试添加 {len(roi_attempts)} 种可能的 ROI (基于 YXZ 及其 Row/Col 翻转)...")
                    for roi_name, mask_to_add in roi_attempts:
                        try:
                            logging.info(f"  尝试添加 ROI: {roi_name}")
                            rtstruct.add_roi(mask=mask_to_add, name=roi_name)
                            logging.info(f"    成功添加 ROI: {roi_name}")
                        except RTStruct.ROIException as roi_exception:
                            # 维度错误理论上不应再发生，因为基础是 YXZ
                            logging.error(f"    添加 ROI '{roi_name}' 时发生 ROIException: {roi_exception}") 
                            # traceback.print_exc() # 可能过于冗长，先注释掉
                        except Exception as e:
                            logging.error(f"    添加 ROI '{roi_name}' 时发生意外错误: {e}")
                            traceback.print_exc()
                    # --- 结束恢复逻辑 --- 
                # --- 结束新增的 else 块 --- 

                # --- 移除旧的 YXZ/XYZ 单独尝试逻辑 ---
                         
                # 保存包含所有成功添加的 ROI 的 RTStruct 文件
                try:
                     logging.info(f"保存 RTStruct 文件到: {NEW_RTSTRUCT_OUTPUT_PATH}")
                     rtstruct.save(NEW_RTSTRUCT_OUTPUT_PATH)
                     logging.info("RTStruct 文件保存成功。")
                except Exception as save_e:
                     logging.error(f"保存最终 RTStruct 文件时出错: {save_e}")
                     traceback.print_exc()
        # --- 结束新增 (RTSTRUCT 生成逻辑) ---

        logging.info("===== NIfTI 到 DICOM 转换任务完成 =====")

    except Exception as e:
        logging.error("处理过程中发生严重错误:")
        traceback.print_exc()
        logging.error("===== NIfTI 到 DICOM 转换失败 =====")