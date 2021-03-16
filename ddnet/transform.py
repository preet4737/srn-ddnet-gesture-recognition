def transform_x(x):
    """
    Input:
        x - x coordinate of joint in depth image space
    Output:
        x - x coordinate of joint in world space
    """
    # return format((((x * 640) - 320) / 320). ".2f") -- Only for visualization
    return ((x * 640) - 320) / 320


def transform_y(y):
    """
    Input:
        x - x coordinate of joint in depth image space
    Output:
        x - x coordinate of joint in world space
    """
    # return format((((y * 480) - 240) / 240), ".2f") -- Only for visualization
    return ((y * 480) - 240) / 240
