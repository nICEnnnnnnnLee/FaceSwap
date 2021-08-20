<p align="center">
      <strong>
        <a href="https://github.com/nICEnnnnnnnLee/FaceSwap" target="_blank">VideoFaceSwap</a>&nbsp;
      </strong>
  <br>
        简单的视频换脸程序
  <br>
      源自<strong>
        <a href="https://github.com/ButterAndButterfly" target="_blank">ButterAndButterfly</a><br>
      </strong>  
        Butter, 寓意宅男; Butterfly, 寓意美好的事物。 
        <br/> 美好的世界由我们创造!  
</p>

## 技术原理  
参考了以下两个项目：
+ [simple_faceswap](https://github.com/Jacen789/simple_faceswap)  
    + 实现了摄像头捕捉的视频流的简单换脸，将其实时显示。   
    + 换脸思路是直接整个面部到面部的转换。   

+ [AI-Change-face-in-the-video](https://github.com/Liangwe/AI-Change-face-in-the-video)  
    + 实现了视频中的人脸到指定照片中的人脸的更换，并且输出视频。  
        但中间有很多冗余步骤，将视频转换为图片保存，再处理图片，再合并视频。  
        这其中多出了大量磁盘读写IO，且视频帧频率信息丢失。       
    + 换脸思路是五官到五官的转换。   

考虑到项目2的缺陷，我们可以参考一个简单的opencv读取复制视频的例子，对其进行改善。  
```python
# opencv读取复制视频
import cv2

video = cv2.VideoCapture("faces/trump.avi")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('trans.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)  
success, frame = video.read()  
index = 1
while success :  
	videoWriter.write(frame)
	success, frame = video.read()
	index += 1
video.release()
```


## 效果预览
<https://www.bilibili.com/video/BV1ff4y137ep>


## 可能的坑  
+ 安装dlib报错  
根据报错信息提示，使用pip安装Cmake即可。  
但安装Cmake可能也会报错，你需要根据提示再安装Visual Studio，主要是要保证C++的编译环境。  

+ 转换后的视频花屏    
原始的image矩阵是0~255的整型的，经过处理后可能变成了浮点型，需要再进行`img.astype(np.uint8)`操作。


## 如何使用  
+ 根据`requirements.txt`安装依赖  
+ 修改[`L15~L18`](https://github.com/nICEnnnnnnnLee/FaceSwap/blob/2ebb74a32b3ed9be50e946cbdde60993e62b1359/main.py#L15-L18)行的参数，指定输入的图片、视频即可。  

