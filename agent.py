from dm import dmap
import cv2

def main():
    dm = dmap.DMap(device = 1, flip = False)
    # "/home/vlad/video.avi")

    while(True):
        depth_map = dm.captureDepthMap()
        cv2.imshow('depth_map', depth_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
