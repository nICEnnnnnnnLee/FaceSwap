# -*- coding: utf-8 -*-

import os
import cv2
import dlib
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

models_folder_path = os.path.join(here, 'models')  # 模型保存文件夹
faces_folder_path = os.path.join(here, 'faces')  # 人脸图片保存文件夹
predictor_path = os.path.join(models_folder_path, 'shape_predictor_68_face_landmarks.dat')  # 模型路径


FACE_NAME = 'Aobama.jpg' #'JayChou.png'/'Aobama.jpg'
VIDEO_SRC = 'model1.mp4'
VIDEO_DST = 'model1_trans_JayChou2.avi' #'trump_trans.mp4'
SWAP_FACE_ALG = 2
if SWAP_FACE_ALG == 2:
    import transfer2 as transfer # 五官分别进行替换
else:
    import transfer1 as transfer # 脸型整体进行替换

image_face_path = os.path.join(faces_folder_path, FACE_NAME)  # 人脸图片路径
video_source_path = os.path.join(faces_folder_path, VIDEO_SRC)  # 源视频路径
video_dest_path = os.path.join(faces_folder_path, VIDEO_DST)  # 源视频路径
video_dest_fourcc = 'mp4v' if video_dest_path.endswith('mp4') else 'XVID'




def trans_video():
    # 读取源视频信息
    video = cv2.VideoCapture(video_source_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print("当前视频帧数：", frameCount)
    if False:
        return None
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 读取要换的脸的图片
    im1 = cv2.imread(image_face_path)  # face_image
    im1 = cv2.resize(im1, (600, im1.shape[0] * 600 // im1.shape[1]))
    detector = dlib.get_frontal_face_detector()  # dlib的正向人脸检测器
    predictor = dlib.shape_predictor(predictor_path)  # dlib的人脸形状检测器
    landmarks1 = transfer.get_face_landmarks(im1, detector, predictor)  # 68_face_landmarks
    if landmarks1 is None:
        print('{}:检测不到人脸'.format(image_face_path))
        exit(1)
    im1_mask = transfer.get_face_mask(im1, landmarks1)  # 脸图人脸掩模
    
    # 逐帧处理图像
    #可以用(*'DVIX')或(*'X264'),(*'AVC1'),(*'MJPG'), (*'XVID')如果都不行先装ffmepg
    videoWriter = cv2.VideoWriter(video_dest_path, cv2.VideoWriter_fourcc(*video_dest_fourcc), fps, size)  
    success, frame = video.read()  
    index = 1
    while success :  
        landmarks2 = transfer.get_face_landmarks(frame, detector, predictor)  # 68_face_landmarks
        if landmarks2 is not None:
            frame_swapped = transfer.transfer_img(im1, frame, im1_mask, landmarks1, landmarks2)
            videoWriter.write(frame_swapped)
        else:
            videoWriter.write(frame)
        
        success, frame = video.read()
        index += 1
        if index%100 == 0:
            #break
            print("处理帧数：" , index)

    video.release()

def trans_capture():
    # 读取源视频信息
    camera = cv2.VideoCapture(0)

    # 读取要换的脸的图片
    im1 = cv2.imread(image_face_path)  # face_image
    im1 = cv2.resize(im1, (600, im1.shape[0] * 600 // im1.shape[1]))
    detector = dlib.get_frontal_face_detector()  # dlib的正向人脸检测器
    predictor = dlib.shape_predictor(predictor_path)  # dlib的人脸形状检测器
    landmarks1 = transfer.get_face_landmarks(im1, detector, predictor)  # 68_face_landmarks
    if landmarks1 is None:
        print('{}:检测不到人脸'.format(image_face_path))
        exit(1)
    im1_mask = transfer.get_face_mask(im1, landmarks1)  # 脸图人脸掩模
    
    # 逐帧处理图像
    success, frame = camera.read()  
    while success :  
        landmarks2 = transfer.get_face_landmarks(frame, detector, predictor)  # 68_face_landmarks
        if landmarks2 is not None:
            frame_swapped = transfer.transfer_img(im1, frame, im1_mask, landmarks1, landmarks2)
            cv2.imshow('seamless_im', frame_swapped)
        else:
            cv2.imshow('seamless_im', frame)
        
        success, frame = camera.read()
        if cv2.waitKey(1) == 27:  # 按Esc退出
            break

    cv2.destroyAllWindows()
if __name__ == '__main__':
    trans_video()
    #trans_capture()
