import cv2
import numpy as np
from sklearn.preprocessing import normalize

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

class DMap:
    def __init__(self,
                calibrationData = "/home/vlad/calibration.npz",
                frameWidth = FRAME_WIDTH,
                frameHeight = FRAME_HEIGHT,
                device = 0,
                out = None,
                outL = None,
                flip = True,
                numDisparities = 64):

        calibration = np.load(calibrationData, allow_pickle=False)
        self.imageSize = tuple(calibration["imageSize"])
        self.leftMapX = calibration["leftMapX"]
        self.leftMapY = calibration["leftMapY"]
        self.leftROI = tuple(calibration["leftROI"])
        self.rightMapX = calibration["rightMapX"]
        self.rightMapY = calibration["rightMapY"]
        self.rightROI = tuple(calibration["rightROI"])

        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.flip = flip
        self.cam = cv2.VideoCapture(device)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth * 2)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
         	
        window_size = 3

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=numDisparities,
            blockSize=5,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
        self.wls_filter.setLambda(80000)
        self.wls_filter.setSigmaColor(1.2)

        if out != None:
            self.videoWriter = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'XVID'), 25, (frameWidth * 2, frameHeight), True)
        else:
             self.videoWriter = None

        if outL != None:
            self.videoWriterL = cv2.VideoWriter(outL, cv2.VideoWriter_fourcc(*'XVID'), 25, (frameWidth, frameHeight), True)
        else:
            self.videoWriterL = None

    def __del__(self):
        if self.videoWriter != None:
            self.videoWriter.release()
        if self.videoWriterL != None:
            self.videoWriterL.release()
        cv2.destroyAllWindows()

    def captureDepthMap(self):
        # Capture frame
        if not self.cam.grab():
            raise Exception('No more frames')
        _, frame = self.cam.retrieve()

        # Flip
        if self.flip:
            frame = cv2.flip(frame, flipCode = -1)

        # Split left and right frames
        leftFrame = frame[0:self.frameHeight, 0:self.frameWidth]
        rightFrame = frame[0:self.frameHeight, self.frameWidth:(self.frameWidth * 2)]

        # Out frame
        if self.videoWriter != None:
            self.videoWriter.write(frame)
        if self.videoWriterL != None:
            self.videoWriterL.write(leftFrame)

        # Calculate disparity map
        disparity_left  = self.left_matcher.compute(leftFrame, rightFrame)
        disparity_right = self.right_matcher.compute(rightFrame, leftFrame)
        disparity_left  = np.int16(disparity_left)
        disparity_right = np.int16(disparity_right)
        filteredImg     = self.wls_filter.filter(disparity_left, leftFrame, None, disparity_right)
        depth_map = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)
        depth_map = cv2.bitwise_not(depth_map)
     
        # Debug draw
        # cv2.imshow('cam', frame)
        # cv2.imshow('left', leftFrame)
        # cv2.imshow('right', rightFrame)
        # cv2.imshow('depth_map', depth_map)

        return depth_map
