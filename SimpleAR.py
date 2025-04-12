import numpy as np
import cv2 as cv

WINDOW_NAME = "Calibration"
VIDEO_PATH = "data/chessboard.mp4"
IMG_PATH = "data/brick.jpg"

WAIT_MSEC = 10
CAPTURE_MSEC = 100

BOARD_PATTERN = (8, 6)
BOARD_CELL_SIZE = 0.025
BOARD_CRITERIA = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

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

def capture_timer(count, img, img_select):
    timeout_count = CAPTURE_MSEC // WAIT_MSEC

    if count == timeout_count:
        img_select.append(img)
        return 0

    return count + 1

def select_video_and_show():
    video = cv.VideoCapture(VIDEO_PATH)
    assert video.isOpened()

    img_select = []
    capture_time, key = 0, 0
    img = np.array([])
    while key_event(key):
        valid, img = video.read()
        if not valid:
            break
        capture_time = capture_timer(capture_time, img, img_select)
        display = img.copy()
        cv.imshow(WINDOW_NAME, display)
        key = cv.waitKey(WAIT_MSEC)
    return img_select

def calibration_camera(images, k= None, d_cf= None):
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, BOARD_PATTERN)
        if complete:
            img_points.append(pts)

    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(BOARD_PATTERN[1]) for c in range(BOARD_PATTERN[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * BOARD_CELL_SIZE] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], cameraMatrix=k, distCoeffs=d_cf)

def draw_bricks(k, d_cf):
    print("Press space to capture image")
    video = cv.VideoCapture(VIDEO_PATH)
    assert video.isOpened(), "Video file cannot be opened"

    stone_img = cv.imread(IMG_PATH)
    assert stone_img is not None, "Image cannot be loaded"

    stone_img_h, stone_img_w = stone_img.shape[:2]
    img_corners = np.array([[0, 0], [stone_img_w, 0], [stone_img_w, stone_img_h], [0, stone_img_h]], dtype=np.float32)

    key = 0
    while key_event(key):
        valid, img = video.read()
        if not valid:
            break
        success, img_points = cv.findChessboardCorners(img, BOARD_PATTERN, BOARD_CRITERIA)
        if success:
            ret, rvec, tvec = cv.solvePnP(OBJ_POINTS, img_points, k, d_cf)

            for proj_plane in PROJ_PLANE:
                plane_corners, _ = cv.projectPoints(proj_plane, rvec, tvec, k, d_cf)
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
        key = cv.waitKey(WAIT_MSEC)

if __name__ == "__main__":
    img_select = select_video_and_show()
    rms, K, dist_coeff, rvecs, tvecs = calibration_camera(img_select)
    print(f"RMS: {rms}")
    print(f"Camera Matrix:\n{K}")
    print(f"Distortion Coefficients:\n{dist_coeff}")
    draw_bricks(K, dist_coeff)
    cv.destroyAllWindows()