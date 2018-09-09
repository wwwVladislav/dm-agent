from dm import dmap
from dm import dnet
import cv2
import argparse
from udp_client import UDPClient

def parseArgs():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    return vars(ap.parse_args())

def main():
    dm = dmap.DMap(device = 1,
                   flip = False,
                   calibrationData = "/home/vlad/calibration.npz")
    args = parseArgs()
    dn = dnet.DNet(args["prototxt"], args["model"])

    while(True):
        frame, depth_map = dm.captureDepthMap()
        detections = dn.detect(frame)

        for detection in detections:
            centerX = detection[2]
            centerY = detection[3]
            objClass = detection[0]
            label = "{}: x{:d} y{:d}, ".format(objClass, centerX, centerY)

            # cv2.circle(frame, (centerX, centerY), 5, (0, 255, 0), -1)
            # cv2.putText(frame, label, (centerX - 10, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), lineType=cv2.LINE_AA)

            if detection[0] == "person":
                UDPClient.rotate("10.10.10.20", 10001, "{:.2f}".format(70-(centerX * 40 / 380)), "95")
                UDPClient.fire("10.10.10.20", 10001)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
