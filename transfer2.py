import cv2
import dlib
import numpy
import sys
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" 
SCALE_FACTOR = 1
FEATURE_AMOUNT = 11

FACE_POINTS = list(range(17, 68))  #脸部轮廓
MOUTH_POINTS = list(range(48, 61))  #嘴部轮廓
RIGHT_BROW_POINTS = list(range(17, 22))  #右眉毛
LEFT_BROW_POINTS = list(range(22, 27))  #左眉毛
RIGHT_EYE_POINTS = list(range(36, 42))  #右眼
LEFT_EYE_POINTS = list(range(42, 48))   #左眼
NOSE_POINTS = list(range(27, 35))   #鼻子
JAW_POINTS = list(range(0, 17))    #颚部

# Points used to line up the images
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of
# each element will be overlaid
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS
                  + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
    ]

# Amount of blur to use during color correction, as a fraction of the
# pupillary distance
COLOUR_CORRECT_BLUR_FRAC = 0.05

#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

## input: an image in the form of a numpy array
## return: a 68 * 2 element matrix, each row corresponding with
## the x, y coordintes of a pariticular feature point in the input image
def get_face_landmarks(im, detector, predictor):
    rects = detector(im, 1)

    """
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    """
    if len(rects) == 0:
        print("Sorry, there were no faces found.")
        return None
    # the feature extractor (predictor) requires a rough bounding box as input
    # to the algorithm. This is provided by a traditional face detector (
    # detector) which returns a list of rectangles, each of which corresponding
    # a face in the image
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annote_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_face_landmarks(im)

    return im, s

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
    
        sum || s*R*p1,i + T - p2,i||^2
        
    is minimized.
    """

    # Solve the procrustes problem by substracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATURE_AMOUNT, FEATURE_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATURE_AMOUNT, FEATURE_AMOUNT), 0)

    return im

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colors(im1, im2, landmarks1,landmarks2): #修改
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        numpy.mean(landmarks2[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1

   

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors:
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
            im2_blur.astype(numpy.float64))


def transfer_img(face, frame, face_mask, landmarks_face, landmarks_frame):
    """
    :param face: 脸图
    :param frame: 图片帧图
    :param face_mask: 脸图face_mask
    :param landmarks_face: 脸图landmarks_face
    :param landmarks_frame: 图片帧图landmarks_frame
    :return: 根据图片2的颜色调整的图片1
    """
    
    M = transformation_from_points(landmarks_frame[ALIGN_POINTS],
                                   landmarks_face[ALIGN_POINTS])

    
    warped_mask = warp_im(face_mask, M, frame.shape)
    combined_mask = numpy.max([get_face_mask(frame, landmarks_frame), warped_mask],axis=0)

    warped_face = warp_im(face, M, frame.shape)
    warped_corrected_face = correct_colors(frame, warped_face, landmarks_frame,landmarks_face)

    frame_swapped = frame * (1.0 - combined_mask) + warped_corrected_face * combined_mask
    
    return frame_swapped.astype(np.uint8)