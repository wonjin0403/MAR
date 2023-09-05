import numpy as np
import pydicom as dcm
import os

def pearson_correlation_coeff(x: np.array, y: np.array)-> float:
    std_x = np.std(x)
    std_y = np.std(y)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    vx = (x - mean_x) / (std_x + 0.0001)
    vy = (y - mean_y) / (std_y + 0.0001)

    return np.mean(vx * vy)

def save_as_dicom_test(output_, test_save_path:str, imgName:str, input_= None, target_= None) -> None:
    if not os.path.exists("%s/test_dcm" % (test_save_path)):
        os.makedirs("%s/test_dcm" % (test_save_path))
        print("%s/test_dcm" % (test_save_path))
    save_path_dicom = "%s/test_dcm/%s" % (test_save_path, imgName[:-4])
    input_ = ((input_ + 1) / 2 * 4095) if input_ is not None else None
    target_ = ((target_ + 1) / 2 * 4095) if target_ is not None else None
    output_ = ((output_ + 1) / 2 * 4095)
    total = np.concatenate([input_, target_, output_], axis=2) if input_ is not None else output_
    # try:    
    # dcm_info = dcm.dcmread("/app/home/jhk22/MAR/data/raw_data/new_data_220109/SNUH_RO_HN_Metal/Non-OMAR(Anony)/%s/%s.dcm" % (imgName[:5], imgName[:-4]), force=True)
    dcm_info = dcm.dcmread("/app/home/jhk22/MAR/data/raw_data/new_data_220109/SNUH_RO_HN_Metal/Non-OMAR(Anony)/ID001/ID001_NonOMAR_005.dcm", force=True)
    new_check = total.astype(np.uint16)
    new_check = new_check.reshape(total.shape[1], total.shape[2])
    dcm_info.Rows = new_check.shape[0]
    dcm_info.Columns = new_check.shape[1]
    dcm_info.PixelData = new_check.tobytes()
    ### output insertion to DCM pixel data
    dcm_info.PixelData = new_check.tobytes()
    dcm_info.save_as(save_path_dicom + '.dcm')
    # except:
    #     print("check data_path", "/app/home/jhk22/MAR/data/raw_data/new_data_220109/SNUH_RO_HN_Metal/Non-OMAR(Anony)/%s/%s.dcm" % (imgName[:5], imgName[:-4]))

    
def save_as_dicom_infer(output_, test_save_path:str, imgName:str, input_= None, target_= None) -> None:
    if not os.path.exists("%s/infer_dcm" % (test_save_path)):
        os.makedirs("%s/infer_dcm" % (test_save_path))
        print("%s/infer_dcm" % (test_save_path))
    save_path_dicom = "%s/infer_dcm/%s" % (test_save_path, imgName[:-4])
    input_ = ((input_ + 1) / 2 * 4095) if input_ is not None else None
    target_ = ((target_ + 1) / 2 * 4095) if target_ is not None else None
    output_ = ((output_ + 1) / 2 * 4095)
    total = np.concatenate([input_, target_, output_], axis=2) if input_ is not None else output_
    # try:    
    dcm_info = dcm.dcmread("/app/home/jhk22/MAR/data/raw_data/new_data_220109/SNUH_RO_HN_Metal/Non-OMAR(Anony)/%s/%s.dcm" % (imgName[:5], imgName[:-4]), force=True)
    # dcm_info = dcm.dcmread("/app/home/jhk22/MAR/data/raw_data/new_data_220109/SNUH_RO_HN_Metal/Non-OMAR(Anony)/ID001/ID001_NonOMAR_005.dcm", force=True)
    new_check = total.cpu().numpy().astype(np.uint16)
    new_check = new_check.reshape(total.shape[1], total.shape[2])
    dcm_info.Rows = new_check.shape[0]
    dcm_info.Columns = new_check.shape[1]
    dcm_info.PixelData = new_check.tobytes()
    ### output insertion to DCM pixel data
    dcm_info.PixelData = new_check.tobytes()
    dcm_info.save_as(save_path_dicom + '.dcm')
    # except:
    #     print("check data_path", "/app/home/jhk22/MAR/data/raw_data/new_data_220109/SNUH_RO_HN_Metal/Non-OMAR(Anony)/%s/%s.dcm" % (imgName[:5], imgName[:-4]))
    