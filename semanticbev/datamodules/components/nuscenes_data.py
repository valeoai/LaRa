"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from glob import glob

from semanticbev.utils.tools import (img_transform, normalize_img, to_tensor, gen_dx_bx, update_intrinsics)



class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, update_intrinsics=True, normalize=True):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.update_intrinsics = update_intrinsics
        self.normalize=normalize

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (
                        rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH)
            # crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop_w = newW - fW
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def parse_pose(self, record, inv=False, flat=False, transformation_matrix=True):

        rotation, translation = record['rotation'], record['translation']

        if flat:
            yaw = Quaternion(rotation).yaw_pitch_roll[0]
            R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        else:
            R = Quaternion(rotation)

        R = R if not inv else R.inverse

        t = np.array(translation, dtype=np.float32)
        t = t if not inv else -t

        if transformation_matrix:
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R
            pose[:3, -1] = t

            return pose
        else:
            return R, t

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []


        for cam in cams:
            cam_sample = self.nusc.get('sample_data', rec['data'][cam])

            imgname = os.path.join(self.nusc.dataroot, cam_sample['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            cam = self.nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
            intrin = torch.Tensor(cam['camera_intrinsic'])

            rot = torch.Tensor(Quaternion(cam['rotation']).rotation_matrix)
            tran = torch.Tensor(cam['translation'])


            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )

            if self.update_intrinsics:
                intrin = update_intrinsics(intrin,
                                           top_crop=crop[1], left_crop=crop[0],
                                           scale_width=resize, scale_height=resize)


            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            if self.normalize:
                img = normalize_img(img)
            else:
                img = to_tensor(img)

            imgs.append(img)
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))


    def get_binimg(self, rec):

        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])

        # Move box to ego vehicle coord system parallel to world z plane.
        # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#289
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        trans = -np.array(lidar_pose['translation'])

        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse

        img = np.zeros((self.nx[0], self.nx[1]))

        visibility = np.full((self.nx[0], self.nx[1]), 255, dtype=np.uint8)

        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue

            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]


            cv2.fillPoly(visibility, [pts], int(inst['visibility_token'])) # plot on visibility mask
            #cv2.polylines(visibility, [pts], True, int(inst['visibility_token']), thickness=1)

            if int(inst['visibility_token']) < self.data_aug_conf['min_visibility']:
                continue

            cv2.fillPoly(img, [pts], 1.0) # only plot on GT if visibility level >= min_visibility

        return torch.Tensor(img).unsqueeze(0), torch.Tensor(visibility).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg, visibility = self.get_binimg(rec)

        output = {
            'imgs': imgs,
            'rots': rots,
            'trans': trans,
            'intrins': intrins,
            'post_rots': post_rots,
            'post_trans': post_trans,
            'binimg': binimg,
            'visibility': visibility
        }

        return output

class WeatherNuscData(NuscData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split and does not have 'rain' in the description
        rain_samples = []
        for samp in samples:
            scene = self.nusc.get('scene', samp['scene_token'])
            if scene['name'] in self.scenes and 'rain' in scene['description']:
                rain_samples.append(samp)

        # sort by scene, timestamp (only to make chronological viz easier)
        rain_samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return rain_samples


class WeatherSegmentationData(WeatherNuscData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg, visibility = self.get_binimg(rec)

        output = {
            'imgs': imgs,
            'rots': rots,
            'trans': trans,
            'intrins': intrins,
            'post_rots': post_rots,
            'post_trans': post_trans,
            'binimg': binimg,
            'visibility': visibility
        }

        return output



def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz, nworkers, pin_memory, parser_name,
                 train_shuffle=True, prefetch_factor=4, update_intrinsics=True, normalize=True):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=False)
    parser = {
        'segmentationdata': SegmentationData,
        'rainy_segmentationdata': WeatherSegmentationData,
    }[parser_name]
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf, grid_conf=grid_conf,
                       update_intrinsics=update_intrinsics, normalize=normalize)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf, grid_conf=grid_conf,
                     update_intrinsics=update_intrinsics, normalize=normalize)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=train_shuffle,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init,
                                              pin_memory=pin_memory,
                                              prefetch_factor=prefetch_factor,
                                              )
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers,
                                            pin_memory=pin_memory,
                                            prefetch_factor=prefetch_factor,
                                            )

    return trainloader, valloader
