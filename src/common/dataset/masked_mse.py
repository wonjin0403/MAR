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

class Masked_CT_Dataset(Dataset):
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
        self.bone_mask = "/app/home/jhk22/MAR/data/mask_all_data"
        self.bone_mask_new = "/app/home/jhk22/MAR/data/mask_all_data_230913"
        self.face_mask = "/app/home/jhk22/MAR/data/only_mask_230128"
        self.torch_type = torch.float32 # if torch_type == float else torch.half # float32 or float16
        self.data_type = data_type
        self.infer = infer
        
    def __getitem__(self, idx):
        return self._loader(idx)

    def __len__(self):
        return len(self.img_paths)

    def _np2tensor(self, np):
        # (x,x) 이미지 -> (1,x,x) 텐서
        tmp = torch.from_numpy(np).view(1, *np.shape)
        # Tensor()는 텐서를 위한 새로운 메모리를 할당하지만 from_numpy()는 넘파이배열의 기존 메모리를 그대로 상속 (텐서의 값을 바꾸면 넘파이배열의 값도 바뀜)
        return tmp.to(dtype=self.torch_type) # dtype, device 등 지정 가능

    def read_bone_mask(self, bone_mask_path, img):
        bone_mask = Image.open(bone_mask_path)
        bone_mask = np.array(bone_mask)
        bone_mask = cv2.cvtColor(bone_mask, cv2.COLOR_BGR2GRAY).astype(float)
        bone_mask = np.where(bone_mask<175, 0, bone_mask)
        bone_mask = np.where(bone_mask>0, 1, bone_mask)

        metal_mask = np.where(img>=4095, 1, np.zeros(img.shape))
        return bone_mask+metal_mask
    
    def read_face_mask(self, face_mask_path, img):
        face_mask = Image.open(face_mask_path)
        face_mask = np.array(face_mask)
        face_mask = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY).astype(float)
        face_mask = np.where(face_mask<175, 0, face_mask)
        face_mask = np.where(face_mask>0, 1, face_mask)
        return face_mask
    
    # _loader 수정함_dataset새로 생성했기때문_reconcat한 dataset을 사용할 경우 이전 _loader를 사용해야함
    def _loader(self, idx):
        img_path = self.img_paths[idx]
        if self.data_type == "dicom":
            img = dcm.dcmread(img_path).pixel_array
        else:
            img = np.load(img_path)

        #infer 데이터는 dicom으로 metal이 있는 경우만 존재
        if self.infer:
            return self._np2tensor(min_max_normalization(img, min_new=-1.0, max_new=1.0)), None, os.path.basename(img_path)
        
        ## 요기에 augmentation code 추가하자.
        a1 = img[:, 512*2: 512*3] #input-white
        # a3 = img[:, 512*4: 512*5] #input-black
        a2 = img[:, 512*3: 512*4] #target
        # mask = img[:, 512*4:] #mask
        
        ################################기존 코드
        bone_path = os.path.join(self.bone_mask, os.path.basename(img_path))
        bone = self.read_bone_mask(bone_path.rsplit(".", 1)[0]+".png", a2)
        bone1 = np.where(bone > 0, 1, np.zeros(bone.shape))
        bone2 = np.where(bone==0, 1, np.zeros(bone.shape))
        ###########################################################
        # ##################################변경 코드
        # bone_path = os.path.join(self.bone_mask_new, os.path.basename(img_path)[-17:-4])
        # bone = self.read_bone_mask(bone_path.rsplit(".", 1)[0]+".png", a2)
        # bone1 = np.where(bone > 0, 1, np.zeros(bone.shape))
        # bone2 = np.where(bone==0, 1, np.zeros(bone.shape))
        # ################################
        ################################얼굴 마스크
        face_path = os.path.join(self.face_mask, os.path.basename(img_path)[-17:-4])
        face = self.read_face_mask(face_path.rsplit(".", 1)[0]+".png", a2)
        face_m = np.where(face > 0, 1, np.zeros(face.shape))
        ###########################################################
        
        a1 = np.where(a1<4095, 0 , a1) # metal input threshold 4095
        metal_size = a1.sum()
        inserted = a1 + a2 # threshold metal + full_image
        inserted_img = np.where(inserted > 4095, 4095, inserted)
        inserted_img = inserted_img * face_m

        input_np = min_max_normalization((img[:, 512*2: 512*3])*face_m, min_new=-1.0, max_new=1.0)
        # input_np_black = min_max_normalization(img[:, 512*4:512*5], min_new=-1.0, max_new=1.0)
        target_np = min_max_normalization(inserted_img, min_new=-1.0, max_new=1.0)#img[:, 512*4: 512*5])

        input_ = self._np2tensor(input_np)
        # input_black = self._np2tensor(input_np_black)
        target_ = self._np2tensor(target_np)
        bone1 = self._np2tensor(bone1)
        bone2 = self._np2tensor(bone2)
        mask_ = self._np2tensor(face_m)
        
        # if random.choice([True, False]):
        #     input_ = torch.cat([input_,input_black], dim=0)
        #     target_ = torch.cat([target_,target_], dim=0)
        # else:
        #     input_ = torch.cat([input_black, input_], dim=0)
        #     target_ = torch.cat([target_,target_], dim=0)
        

        a = np.random.randint(3000, size=1)
        
        seed_list = []
        if self.transform:
            seed_list.append(a[0])
            self.transform.set_random_state(seed=a[0])
            input_ = self.transform(input_)
            self.transform.set_random_state(seed=a[0])
            target_ = self.transform(target_)
            self.transform.set_random_state(seed=a[0])
            bone1 = self.transform(bone1)
            self.transform.set_random_state(seed=a[0])
            bone2 = self.transform(bone2)
            self.transform.set_random_state(seed=a[0])
            mask_ = self.transform(mask_)



        # concat_img = np.concatenate((input_.reshape(512,512), target_.reshape(512,512), bone1.reshape(512,512), bone2.reshape(512,512), mask_.reshape(512,512)), axis=1)
        # plt.imsave('/app/home/jhk22/MAR/img_save_path/231003/input'+'_'+str(a[0])+'.png', concat_img)

        
        # return input_, target_, os.path.basename(img_path)
        return input_, target_, bone1, bone2, metal_size, os.path.basename(img_path)