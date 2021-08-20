# -*- coding: utf-8 -*-

import os
import cv2
import dlib
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

models_folder_path = os.path.join(here, 'models')  # 模型保存文件夹
faces_folder_path = os.path.join(here, 'faces')  # 人脸图片保存文件夹
predictor_path = os.path.join(models_folder_path, 'shape_predictor_68_face_landmarks.dat')  # 模型路径
image_face_path = os.path.join(faces_folder_path, 'JayChou.png')  # 人脸图片路径

detector = dlib.get_frontal_face_detector()  # dlib的正向人脸检测器
predictor = dlib.shape_predictor(predictor_path)  # dlib的人脸形状检测器


def get_image_size(image):
    """
    获取图片大小（高度,宽度）
    :param image: image
    :return: （高度,宽度）
    """
    image_size = (image.shape[0], image.shape[1])
    return image_size


def get_face_landmarks(image, face_detector, shape_predictor):
    """
    获取人脸标志，68个特征点
    :param image: image
    :param face_detector: dlib.get_frontal_face_detector
    :param shape_predictor: dlib.shape_predictor
    :return: np.array([[],[]]), 68个特征点
    """
    dets = face_detector(image, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found.")
        return None
    shape = shape_predictor(image, dets[0])
    face_landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return face_landmarks


def get_face_mask(image, face_landmarks):
    """
    获取人脸掩模
    :param image: 图片
    :param face_landmarks: 68个特征点
    :return: image_mask, 掩模图片
    """
    image_size = get_image_size(image)
    mask = np.zeros(image_size, dtype=np.uint8)
    points = np.concatenate([face_landmarks[0:16], face_landmarks[26:17:-1]])
    cv2.fillPoly(img=mask, pts=[points], color=255)

    # mask = np.zeros(image_size, dtype=np.uint8)
    # points = cv2.convexHull(face_landmarks)  # 凸包
    # cv2.fillConvexPoly(mask, points, color=255)
    return mask


def get_affine_image(image1, image2, face_landmarks1, face_landmarks2):
    """
    获取图片1仿射变换后的图片
    :param image1: 图片1, 要进行仿射变换的图片
    :param image2: 图片2, 只要用来获取图片大小，生成与之大小相同的仿射变换图片
    :param face_landmarks1: 图片1的人脸特征点
    :param face_landmarks2: 图片2的人脸特征点
    :return: 仿射变换后的图片
    """
    three_points_index = [18, 8, 25]
    M = cv2.getAffineTransform(face_landmarks1[three_points_index].astype(np.float32),
                               face_landmarks2[three_points_index].astype(np.float32))
    dsize = (image2.shape[1], image2.shape[0])
    affine_image = cv2.warpAffine(image1, M, dsize)
    return affine_image.astype(np.uint8)


def get_mask_center_point(image_mask):
    """
    获取掩模的中心点坐标
    :param image_mask: 掩模图片
    :return: 掩模中心
    """
    image_mask_index = np.argwhere(image_mask > 0)
    miny, minx = np.min(image_mask_index, axis=0)
    maxy, maxx = np.max(image_mask_index, axis=0)
    center_point = ((maxx + minx) // 2, (maxy + miny) // 2)
    return center_point


def get_mask_union(mask1, mask2):
    """
    获取两个掩模掩盖部分的并集
    :param mask1: mask_image, 掩模1
    :param mask2: mask_image, 掩模2
    :return: 两个掩模掩盖部分的并集
    """
    mask = np.min([mask1, mask2], axis=0)  # 掩盖部分并集
    mask = ((cv2.blur(mask, (5, 5)) == 255) * 255).astype(np.uint8)  # 缩小掩模大小
    mask = cv2.blur(mask, (3, 3)).astype(np.uint8)  # 模糊掩模
    return mask


def skin_color_adjustment(im1, im2, mask=None):
    """
    肤色调整
    :param im1: 图片1
    :param im2: 图片2
    :param mask: 人脸 mask. 如果存在，使用人脸部分均值来求肤色变换系数；否则，使用高斯模糊来求肤色变换系数
    :return: 根据图片2的颜色调整的图片1
    """
    if mask is None:
        im1_ksize = 55
        im2_ksize = 55
        im1_factor = cv2.GaussianBlur(im1, (im1_ksize, im1_ksize), 0).astype(np.float)
        im2_factor = cv2.GaussianBlur(im2, (im2_ksize, im2_ksize), 0).astype(np.float)
    else:
        im1_face_image = cv2.bitwise_and(im1, im1, mask=mask)
        im2_face_image = cv2.bitwise_and(im2, im2, mask=mask)
        im1_factor = np.mean(im1_face_image, axis=(0, 1))
        im2_factor = np.mean(im2_face_image, axis=(0, 1))

    im1 = np.clip((im1.astype(np.float) * im2_factor / np.clip(im1_factor, 1e-6, None)), 0, 255).astype(np.uint8)
    return im1
    
def transfer_img(face, frame, face_mask, landmarks1, landmarks2):
    """
    :param face: 脸图
    :param frame: 图片帧图
    :param face_mask: 脸图face_mask
    :param landmarks1: 脸图landmarks1
    :param landmarks2: 图片帧图landmarks2
    :return: 根据图片2的颜色调整的图片1
    """
    frame_mask = get_face_mask(frame, landmarks2)  # 摄像头图片人脸掩模

    affine_face = get_affine_image(face, frame, landmarks1, landmarks2)  # face（脸图）仿射变换后的图片
    affine_face_mask = get_affine_image(face_mask, frame, landmarks1, landmarks2)  # face（脸图）仿射变换后的图片的人脸掩模

    union_mask = get_mask_union(frame_mask, affine_face_mask)  # 掩模合并

    affine_face = skin_color_adjustment(affine_face, frame, mask=union_mask)  # 肤色调整
    point = get_mask_center_point(affine_face_mask)  # face（脸图）仿射变换后的图片的人脸掩模的中心点
    frame_swapped = cv2.seamlessClone(affine_face, frame, mask=union_mask, p=point, flags=cv2.NORMAL_CLONE)  # 进行泊松融合
    return frame_swapped


