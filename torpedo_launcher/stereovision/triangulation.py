import numpy as np

def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):

    height_right, width_right , depth_right = frame_right.shape
    height_left , width_left , depth_left = frame_left.shape

    # ENSURING THEY HAVE THE SAME WIDTH ELSE COOKED

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print("left and right camera frames unfortunately don't have the same pixel width")

    x_right = right_point[0]
    x_left = left_point[0]

    # PARALLAX a.k.a DISPARITY

    parallax = x_left - x_right

    # DEPTH (THE VALUE WILL COME OUT IN CM)

    Depth = (baseline*f_pixel)/parallax

    return abs(Depth)
