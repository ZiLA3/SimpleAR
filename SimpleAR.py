import numpy as np
import cv2 as cv

WINDOW_NAME = "Calibration"
VIDEO_PATH = "data/chessboard.mp4"
IMG_PATH = "data/brick.jpg"
WAIT_MSEC = 10

BOARD_PATTERN = (8, 6)
BOARD_CELL_SIZE = 0.025
BOARD_CRITERIA = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

K = np.array([[465.308435, 0, 486.21601542],
              [0, 464.23588018, 271.28307259],
              [0, 0, 1]])

DIST_COEFF = np.array([0.00143715, 0.06756223, -0.00190889, 0.0012528, -0.09642049])

OBJ_POINTS = BOARD_CELL_SIZE * np.array([[c, r, 0] for r in range(BOARD_PATTERN[1]) for c in range(BOARD_PATTERN[0])])

PROJ_PLANE = BOARD_CELL_SIZE * np.array([[[4, 2, -1], [5, 2, -1], [5, 2, 0], [4, 2, 0]],
                                        [[4, 3, -1], [5, 3, -1], [5, 3, 0], [4, 3, 0]],
                                        [[4, 2, -1], [4, 3, -1], [4, 3, 0], [4, 2, 0]],
                                        [[5, 2, -1], [5, 3, -1], [5, 3, 0], [5, 2, 0]],
                                        [[4, 2, -1], [5, 2, -1], [5, 3, -1], [4, 3, -1]]]
                                        ,dtype=np.float32)

def key_event(key):
    if key == 27:
        return False
    if key == ord(' '):
        cv.waitKey()
    return True

video = cv.VideoCapture(VIDEO_PATH)
assert video.isOpened(), "Video file cannot be opened"

stone_img = cv.imread(IMG_PATH)
assert stone_img is not None, "Image cannot be loaded"
stone_img_h, stone_img_w = stone_img.shape[:2]
img_corners = np.array([[0, 0], [stone_img_w, 0], [stone_img_w, stone_img_h], [0, stone_img_h]], dtype=np.float32)


while key_event(cv.waitKey(WAIT_MSEC)):
    valid, img = video.read()
    if not valid:
        break
    success, img_points = cv.findChessboardCorners(img, BOARD_PATTERN, BOARD_CRITERIA)
    if success:
        ret, rvec, tvec = cv.solvePnP(OBJ_POINTS, img_points, K, DIST_COEFF)

        for proj_plane in PROJ_PLANE:
            plane_corners, _ = cv.projectPoints(proj_plane, rvec, tvec, K, DIST_COEFF)
            plane_corners = np.int32(plane_corners).reshape(-1, 2)

            H = cv.getPerspectiveTransform(img_corners, plane_corners.astype(np.float32))
            warped = cv.warpPerspective(stone_img, H, (img.shape[1], img.shape[0]))

            mask = np.zeros_like(img, dtype=np.uint8)
            cv.fillConvexPoly(mask, plane_corners, (255, 255, 255))
            masked_img = cv.bitwise_and(warped, mask)
            mask = cv.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
            masked_bg = cv.bitwise_and(img, cv.bitwise_not(mask))

            img = cv.add(masked_img, masked_bg)

        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]'
        cv.putText(img, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))

    cv.imshow(WINDOW_NAME, img)

video.release()
cv.destroyAllWindows()