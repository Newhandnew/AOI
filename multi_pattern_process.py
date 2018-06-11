import cv2


def get_pattern_image_path(series_image_path, pattern_extension, image_extension):
    pattern_path_list = []
    for pattern in pattern_extension:
        pattern_path = series_image_path + '_' + pattern + '.' + image_extension
        pattern_path_list.append(pattern_path)
    return pattern_path_list


def read_image_array(pattern_path_list):
    image_array = []
    for path in pattern_path_list:
        image = cv2.imread(path, 0)
        image_array.append(image)
    return image_array