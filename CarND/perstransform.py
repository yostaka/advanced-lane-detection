import numpy as np
import cv2

# Apply a perspective transform to rectify binary image ("birds-eye view")
def warp(img):
    img_size = (img.shape[1], img.shape[0])

    # Four source coordinates
    src = np.float32(
        [[761, 499],
         [1034, 673],
         [274, 673],
         [526, 499]])

    dst = np.float32(
        [[1034, 499],
         [1034, 673],
         [274, 673],
         [274, 499]])

    # Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)

    # Create warped image - uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped


def unwarp(img):
    img_size = (img.shape[1], img.shape[0])

    # Four source coordinates
    src = np.float32(
        [[761, 499],
         [1034, 673],
         [274, 673],
         [526, 499]])

    # Four desired coordinates
    # Need to fix values
    # dst = np.float32(
    #     [[761, 499],
    #      [761, 673],
    #      [277, 673],
    #      [277, 499]])
    dst = np.float32(
        [[1034, 499],
         [1034, 673],
         [274, 673],
         [274, 499]])

    # Compute the perspective transform, M
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create warped image - uses linear interpolation
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)

    return unwarped
