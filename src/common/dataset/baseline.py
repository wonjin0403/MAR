import os
import json
import torch
import cv2
import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    RandCoarseDropout,
    OneOf,
    AdjustContrast,
    GaussianSmooth,
    Rand2DElastic,
    RandRotate,
    RandZoom,
    RandFlip
)
from PIL import Image
import sys
sys.path.append("../../")
from common.utils.normalization import min_max_normalization

class CT_Dataset(Dataset):
    def __init__(self, fold_path, dataset_type, transform=None, torch_type="float", data_type="dicom", infer=False):
        with open(fold_path, 'r') as j:
            contents = json.loads(j.read())
            img_paths = contents[dataset_type]
        #img_paths = json.loads(fold_path)[dataset_type]
        if len(img_paths) == 0:
            raise ValueError("Check data path : %s"%(img_paths))
        self.img_paths = img_paths
        self.transform=transform
        if dataset_type=="Train":
            self.transform = Compose([
                RandCoarseDropout(holes=50, spatial_size=10, dropout_holes=True, fill_value=0, max_holes=100, max_spatial_size=20, prob=0.3),
                OneOf([
                    RandRotate(range_x=[0.1, 6.2], prob=0.3),
                    RandZoom(min_zoom=0.5, max_zoom=5.0, prob=0.3),
                    RandFlip(spatial_axis=1, prob=0.3),
                ]),
                Rand2DElastic(
                    prob=0.5,
                    spacing=(10, 20), # distance between control point
                    magnitude_range=(1,7),  # 1~7
                    rotate_range=(0),# ~ np.pi
                    scale_range=(0), # 0.5 넘지 않게
                    translate_range=(0,100), # 위 아래, 양 옆 이동
                    padding_mode="zeros"
                    )
            ]) 
        self.torch_type = torch.float32# if torch_type == float else torch.half # float32 or float16
        self.face_mask = "/app/home/jhk22/MAR/data/only_mask_230128"
        self.data_type = data_type
        self.infer = infer
        
    def __getitem__(self, idx):
        return self._loader(idx)

    def __len__(self):
        return len(self.img_paths)

    def read_face_mask(self, face_mask_path, img):
        face_mask = Image.open(face_mask_path)
        face_mask = np.array(face_mask)
        face_mask = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY).astype(float)
        face_mask = np.where(face_mask<175, 0, face_mask)
        face_mask = np.where(face_mask>0, 1, face_mask)
        return face_mask
    
    def _np2tensor(self, np):
        # (x,x) 이미지 -> (1,x,x) 텐서
        tmp = torch.from_numpy(np).view(1, *np.shape)
        # Tensor()는 텐서를 위한 새로운 메모리를 할당하지만 from_numpy()는 넘파이배열의 기존 메모리를 그대로 상속 (텐서의 값을 바꾸면 넘파이배열의 값도 바뀜)
        return tmp.to(dtype=self.torch_type) # dtype, device 등 지정 가능

    # _loader 수정함_dataset새로 생성했기때문_reconcat한 dataset을 사용할 경우 이전 _loader를 사용해야함
    def _loader(self, idx):
        img_path = self.img_paths[idx]
        if self.data_type == "dicom":
            img = dcm.dcmread(img_path).pixel_array
        else:
            img = np.load(img_path)

        if self.infer:
            return self._np2tensor(min_max_normalization(img, min_new=-1.0, max_new=1.0)), None, os.path.basename(img_path)
        ## 요기에 augmentation code 추가하자.
        a1 = img[:, 512*2: 512*3] #input-white
        # a3 = img[:, 512*4: 512*5] #input-black
        a2 = img[:, 512*3: 512*4] #target
        # mask = img[:, 512*4:] #mask
        
        ################################얼굴 마스크
        face_path = os.path.join(self.face_mask, os.path.basename(img_path)[-17:-4])
        face = self.read_face_mask(face_path.rsplit(".", 1)[0]+".png", a2)
        face_m = np.where(face > 0, 1, np.zeros(face.shape))
        ###########################################################
        
        
        a1 = np.where(a1<4095, 0 , a1) # metal input threshold 4095
        inserted = a1 + a2 # threshold metal + full_image
        inserted_img = np.where(inserted > 4095, 4095, inserted) 
        inserted_img = inserted_img * face_m
        
        input_np = min_max_normalization(img[:, 512*2: 512*3]*face_m, min_new=-1.0, max_new=1.0)
        # input_np_black = min_max_normalization(img[:, 512*4:512*5], min_new=-1.0, max_new=1.0)
        target_np = min_max_normalization(inserted_img, min_new=-1.0, max_new=1.0)#img[:, 512*4: 512*5])

        input_ = self._np2tensor(input_np)
        # input_black = self._np2tensor(input_np_black)
        target_ = self._np2tensor(target_np)
        mask_ = self._np2tensor(face_m)

        a = np.random.randint(3000, size=1)
        
        seed_list = []
        if self.transform:
            seed_list.append(a[0])
            self.transform.set_random_state(seed=a[0])
            input_ = self.transform(input_)
            self.transform.set_random_state(seed=a[0])
            target_ = self.transform(target_)
            self.transform.set_random_state(seed=a[0])
            mask_ = self.transform(mask_)
        return input_, target_, mask_, os.path.basename(img_path)
        # return input_, target_, os.path.basename(img_path)