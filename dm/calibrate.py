import cv2
import numpy as np

CHESSBOARD_SIZE = (7, 9)
CHESSBOARD_OPTIONS = (cv2.CALIB_CB_ADAPTIVE_THRESH
                        | cv2.CALIB_CB_NORMALIZE_IMAGE
                        | cv2.CALIB_CB_FAST_CHECK)
OBJECT_POINT_ZERO = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
OBJECT_POINT_ZERO[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
OPTIMIZE_ALPHA = 0.25
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def findObjAndImgPoints(frame):
    hasCorners, corners = cv2.findChessboardCorners(
        frame,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_FAST_CHECK)
    return hasCorners, corners

def calibrate(outputFile = "/home/vlad/calibration.npz",
              frameWidth = 640,
              frameHeight = 480,
              chessboard = "/home/vlad/chessboard.avi"):
    cam = cv2.VideoCapture(chessboard)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth * 2)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    
    leftObjPoints = []
    leftImgPoints = []
    rightObjPoints = []
    rightImgPoints = []

    i = 0

    while (True):
        i = i + 1

        # Capture frame
        if not cam.grab():
            break
        _, frame = cam.retrieve()

        if i < 200:
            continue
        if i > 200 + 128:
            break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Split left and right frames
        leftFrame = grayFrame[0:frameHeight, 0:frameWidth]
        rightFrame = grayFrame[0:frameHeight, frameWidth:(frameWidth * 2)]

        # Find corners
        hasCorners, corners = findObjAndImgPoints(leftFrame)
        if hasCorners:
            leftObjPoints.append(OBJECT_POINT_ZERO)
            cv2.cornerSubPix(leftFrame, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
            leftImgPoints.append(corners)

        cv2.drawChessboardCorners(leftFrame, CHESSBOARD_SIZE, corners, hasCorners)
        cv2.imshow('leftFrame', leftFrame)

        hasCorners, corners = findObjAndImgPoints(rightFrame)
        if hasCorners:
            rightObjPoints.append(OBJECT_POINT_ZERO)
            cv2.cornerSubPix(rightFrame, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
            rightImgPoints.append(corners)

        cv2.drawChessboardCorners(rightFrame, CHESSBOARD_SIZE, corners, hasCorners)
        cv2.imshow('rightFrame', rightFrame)

        # Wait key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    objectPoints = leftObjPoints

    _, leftCameraMatrix, leftDistortionCoefficients, _, _ = cv2.calibrateCamera(
        objectPoints, leftImgPoints, (frameWidth, frameHeight), None, None)

    _, rightCameraMatrix, rightDistortionCoefficients, _, _ = cv2.calibrateCamera(
        objectPoints, rightImgPoints, (frameWidth, frameHeight), None, None)

    (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        objectPoints, leftImgPoints, rightImgPoints,
        leftCameraMatrix, leftDistortionCoefficients,
        rightCameraMatrix, rightDistortionCoefficients,
        (frameWidth, frameHeight), None, None, None, None,
        cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)

    (leftRectification, rightRectification, leftProjection, rightProjection,
            dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                    leftCameraMatrix, leftDistortionCoefficients,
                    rightCameraMatrix, rightDistortionCoefficients,
                    (frameWidth, frameHeight), rotationMatrix, translationVector,
                    None, None, None, None, None,
                    cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            leftCameraMatrix, leftDistortionCoefficients, leftRectification,
            leftProjection, (frameWidth, frameHeight), cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            rightCameraMatrix, rightDistortionCoefficients, rightRectification,
            rightProjection, (frameWidth, frameHeight), cv2.CV_32FC1)

    np.savez_compressed(outputFile, imageSize=(frameWidth, frameHeight),
            leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
            rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate()
