from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from skimage import transform as trans


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', type=bool, default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def crop_face(net, device, cfg, data_dir, target_dir, left_scale=0.0, right_scale=0.0, up_scale=0.0, low_scale=0.0):
    resize = 1

    landmark_target_dir = target_dir + '_landmark'

    # testing begin
    for dir, dirs, files in os.walk(data_dir):
        new_dir = dir.replace(data_dir, target_dir)
        new_landmark_dir = dir.replace(data_dir, landmark_target_dir)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)

        if not os.path.isdir(new_landmark_dir):
            os.mkdir(new_landmark_dir)

        for file in files:
            filepath = os.path.join(dir, file)
            print(filepath)
            image_path = filepath
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if img_raw is None:
                continue

            im_height, im_width, _ = img_raw.shape
            print(img_raw.shape)

            scale_with = 640
            scale_height = 480
            if im_height > scale_height:
                scale_rate = scale_height / im_height
                img_raw = cv2.resize(img_raw, (int(im_width * scale_rate), scale_height))
            elif im_width > scale_with:
                scale_rate = scale_with / im_width
                img_raw = cv2.resize(img_raw, (scale_with, int(im_height * scale_rate)))

            img = np.float32(img_raw)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            # tic = time.time()
            loc, conf, landms = net(img)  # forward pass
            # print('net forward time: {:.4f}'.format(time.time() - tic))

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # save image
            if args.save_image:
                max_bb = 0
                max_index = 0
                if len(dets) == 0:
                    continue
                elif len(dets) > 1:
                    print('warning detect more than one:', filepath)

                    # find maximum bounding box
                    for di, b in enumerate(dets):
                        if b[4] < args.vis_thres:
                            continue
                        b = list(map(int, b))
                        b = [p if p > 0 else 0 for p in b]
                        box_w = abs(b[1] - b[3])
                        box_h = abs(b[0] - b[2])

                        if max_bb < max(box_w, box_h):
                            max_bb = max(box_w, box_h)
                            max_index = di
                di = max_index
                b = list(map(int, dets[max_index]))

                b = [p if p > 0 else 0 for p in b]
                b[1] -= int((b[3] - b[1]) * up_scale)
                b[3] += int((b[3] - b[1]) * low_scale)
                b[0] -= int((b[2] - b[0]) * left_scale)
                b[2] += int((b[2] - b[0]) * right_scale)
                b[1] = b[1] if b[1] >= 0 else 0
                b[3] = b[3] if b[3] < im_height else im_height - 1
                b[0] = b[0] if b[0] >= 0 else 0
                b[2] = b[2] if b[2] < im_width else im_width - 1

                # retain background
                b_width = b[2] - b[0]
                b_height = b[3] - b[1]

                if b_width > b_height:
                    b[1] -= abs(b_width - b_height) // 2
                    b[3] += abs(b_width - b_height) // 2
                elif b_width < b_height:
                    b[0] -= abs(b_width - b_height) // 2
                    b[2] += abs(b_width - b_height) // 2

                b[1] = b[1] if b[1] >= 0 else 0
                b[3] = b[3] if b[3] < im_height else im_height - 1
                b[0] = b[0] if b[0] >= 0 else 0
                b[2] = b[2] if b[2] < im_width else im_width - 1

                roi_image = np.copy(img_raw[b[1]:b[3], b[0]:b[2]])
                box_w = abs(b[1] - b[3])
                box_h = abs(b[0] - b[2])

                landms[di][0:landms.shape[1]:2] -= b[0]
                landms[di][1:landms.shape[1]:2] -= b[1]

                show_image = roi_image.copy()
                leftEyeCenter = (int(landms[di][0]), int(landms[di][1]))
                rightEyeCenter = (int(landms[di][2]), int(landms[di][3]))
                noseCenter = (int(landms[di][4]), int(landms[di][5]))
                mouth1 = (int(landms[di][6]), int(landms[di][7]))
                mouth2 = (int(landms[di][8]), int(landms[di][9]))

                cv2.circle(show_image, (leftEyeCenter[0], leftEyeCenter[1]), 3, (0, 255, 0), -1)
                cv2.circle(show_image, (rightEyeCenter[0], rightEyeCenter[1]), 3, (0, 255, 0), -1)
                cv2.circle(show_image, (noseCenter[0], noseCenter[1]), 3, (0, 255, 0), -1)
                cv2.circle(show_image, (mouth1[0], mouth1[1]), 3, (0, 255, 0), -1)
                cv2.circle(show_image, (mouth2[0], mouth2[1]), 3, (0, 255, 0), -1)

                aligned_image = norm_crop(roi_image, landms[di].reshape(-1, 2))

                new_path = filepath.replace(data_dir, target_dir)
                new_landmark_path = filepath.replace(data_dir, landmark_target_dir)
                new_path = new_path.replace(new_path.split('.')[-1], 'jpg')
                new_landmark_path = new_landmark_path.replace(new_landmark_path.split('.')[-1], 'jpg')
                print(new_path)
                cv2.imwrite(new_path, aligned_image)
                cv2.imwrite(new_landmark_path, show_image)

                cv2.imshow('crop', roi_image)
                cv2.imshow('aligned', aligned_image)
                cv2.imshow('landmark', show_image)
                cv2.waitKey(1)


def main():
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # data_dir = '../face_dataset/masked_whn'
    # target_dir = '../face_dataset/masked_whn_crop'

    # data_dir = '../face_dataset/CASIA-maxpy-clean'
    # target_dir = '../face_dataset/CASIA-maxpy-clean_crop'

    # data_dir = '../frvtTestbed/pnas/images'
    # target_dir = '../frvtTestbed/pnas_crop'
    #
    # crop_face(net, device, cfg, data_dir, target_dir)
    #
    # data_dir = '../frvtTestbed/common/images'
    # target_dir = '../frvtTestbed/mugshot_crop'
    #
    # crop_face(net, device, cfg, data_dir, target_dir)

    # data_dir = '../face_dataset/calfw/aligned_images'
    # target_dir = '../face_dataset/calfw/aligned_images_crop'
    #
    # crop_face(net, device, cfg, data_dir, target_dir)
    #
    # data_dir = '../face_dataset/cplfw/aligned_images'
    # target_dir = '../face_dataset/cplfw/aligned_images_crop'
    #
    # crop_face(net, device, cfg, data_dir, target_dir)

    # data_dir = '../face_dataset/Celeba/img_align_celeba'
    # target_dir = '../face_dataset/Celeba/img_align_celeba_crop'
    #
    # crop_face(net, device, cfg, data_dir, target_dir)

    # data_dir = '../face_dataset/GEO_enroll'
    # target_dir = '../face_dataset/GEO_enroll_crop'
    # crop_face(net, device, cfg, data_dir, target_dir)
    #
    # data_dir = '../face_dataset/GEO_enroll'
    # target_dir = '../face_dataset/GEO_enroll_large_crop'
    # crop_face(net, device, cfg, data_dir, target_dir, left_scale=0.1, right_scale=0.1, up_scale=0.1, low_scale=0.1)
    #
    # data_dir = '../face_dataset/GEO_Mask_Testing_Dataset'
    # target_dir = '../face_dataset/GEO_Mask_Testing_Dataset_large_crop'
    # crop_face(net, device, cfg, data_dir, target_dir, left_scale=0.05, right_scale=0.05, up_scale=0.05, low_scale=0.05)

    # data_dir = '../face_dataset/GEO_Mask_Testing_Dataset'
    # target_dir = '../face_dataset/GEO_Mask_Testing_Dataset_crop'
    # crop_face(net, device, cfg, data_dir, target_dir)
    #
    # data_dir = '../face_dataset/GEO_env_dataset'
    # target_dir = '../face_dataset/GEO_env_dataset_crop'
    # crop_face(net, device, cfg, data_dir, target_dir)
    #
    # data_dir = '../face_dataset/GEO_identity'
    # target_dir = '../face_dataset/GEO_identity_crop'
    # crop_face(net, device, cfg, data_dir, target_dir)

    # data_dir = '../face_dataset/MEDS_II'
    # target_dir = '../face_dataset/MEDS_II_crop'
    # crop_face(net, device, cfg, data_dir, target_dir)
    #
    # data_dir = '../face_dataset/MEDS_II_mask'
    # target_dir = '../face_dataset/MEDS_II_mask_crop'
    # crop_face(net, device, cfg, data_dir, target_dir)

    # data_dir = '/media/bossun/Bossun_TX2/face_dataset/CACD_VS'
    # target_dir = '/media/bossun/Bossun_TX2/face_dataset/CACD_VS_crop'
    # crop_face(net, device, cfg, data_dir, target_dir)

    data_dir = '../face_dataset/CASIA-maxpy-clean'
    target_dir = '../face_dataset/CASIA-maxpy-clean_large_crop'
    crop_face(net, device, cfg, data_dir, target_dir, left_scale=0.1, right_scale=0.1, up_scale=0.1, low_scale=0.1)

    data_dir = '../face_dataset/1N_test_dataset_origin/GEO_Mask_Testing_Dataset_1N/identity'
    target_dir = '../face_dataset/1N_test_dataset/GEO_Mask_Testing_Dataset_large_crop_1N/identity'
    crop_face(net, device, cfg, data_dir, target_dir, left_scale=0.1, right_scale=0.1, up_scale=0.1, low_scale=0.1)


if __name__ == '__main__':
    main()


