from math import ceil
import random
import os
import cv2
import numpy as np


class CropImage(object):

    def __init__(self, save_image_dir='', num_class=0):
        self.grid_array = []
        self.save_image_dir = ''
        self.image = 0
        if save_image_dir != '':
            self.set_save_dir(save_image_dir, num_class)

    def get_grid_axis(self, img_height, img_width, crop_size):
        crop_width = crop_size[0]
        crop_height = crop_size[1]
        available_width = img_width - crop_width
        available_height = img_height - crop_height
        num_x = ceil(available_width / crop_width)
        num_y = ceil(available_height / crop_height)
        grid_x = available_width / num_x
        grid_y = available_height / num_y

        grid_array = []
        # add 1 for final crop point
        for y in range(num_y + 1):
            for x in range(num_x + 1):
                axis_x = round(x * grid_x)
                axis_y = round(y * grid_y)
                grid_array.append((axis_x, axis_y))

        return grid_array

    def crop_ok_image(self, img_path, crop_size):
        self.image = cv2.imread(img_path, 0)
        height, width = self.image.shape
        height = height - 1
        width = width - 1
        self.grid_array = self.get_grid_axis(height, width, crop_size)
        ok_images = []
        for grid in self.grid_array:
            x = grid[0]
            y = grid[1]
            ok_images.append(self.image[y:y + crop_size[1], x:x + crop_size[0]])

        return ok_images

    def random_crop(self, input_array, defect_point, crop_size):
        """Random crop image including defect point.

            input_array: origin AOI image, gray scale
            defect_point: defect point (x, y)
            crop_size: crop size in [width, height]

          Returns:
            sub-array in input_array including defect point
          """
        margin = 2
        y_min = margin
        x_min = margin
        y_max, x_max = input_array.shape
        x_max = x_max - crop_size[0] - margin
        y_max = y_max - crop_size[1] - margin
        random_x_min = max(defect_point[0] - crop_size[0] + margin, x_min)
        random_x_max = min(defect_point[0] - margin, x_max)
        random_y_min = max(defect_point[1] - crop_size[1] + margin, y_min)
        random_y_max = min(defect_point[1] - margin, y_max)
        crop_x = random.randint(random_x_min, random_x_max)
        crop_y = random.randint(random_y_min, random_y_max)
        # print('({}, {}, {}, {})'.format(random_x_min, random_x_max, random_y_min, random_y_max))
        return input_array[crop_y:crop_y + crop_size[1], crop_x:crop_x + crop_size[0]]

    def crop_ng_image(self, img_path, defect_point, crop_size, crop_number):
        image = cv2.imread(img_path, 0)
        ng_images = []
        for i in range(crop_number):
            crop_image = self.random_crop(image, defect_point, crop_size)
            ng_images.append(crop_image)
            ng_images.append(cv2.flip(crop_image, 1))
            ng_images.append(cv2.flip(crop_image, 0))
            ng_images.append(cv2.flip(crop_image, -1))

        return ng_images

    def set_save_dir(self, save_image_dir, num_class):
        self.save_image_dir = save_image_dir
        for i in range(num_class):
            path = os.path.join(save_image_dir, str(i))
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)

    def save_image(self, image_array, image_name, label):
        image_dir = os.path.join(self.save_image_dir, str(label))
        for i, image in enumerate(image_array):
            file_name = '{}_{}.png'.format(image_name, i)
            image_path = os.path.join(image_dir, file_name)
            cv2.imwrite(image_path, image)

    def get_defect_axis(self, prediction):
        defect = np.nonzero(prediction)[0]
        result = []
        for i in defect:
            result.append(self.grid_array[i])
        return result

    def save_defect_image(self, prediction, image_name, crop_size):
        defects = self.get_defect_axis(prediction)
        for defect in defects:
            cv2.rectangle(self.image, defect, (defect[0] + crop_size[0], defect[1] + crop_size[1]), 255)
        image_path = os.path.join(self.save_image_dir, image_name)
        cv2.imwrite(image_path, self.image)


def main():
    img_path = '/home/new/Downloads/dataset/AOI/1.25/6P7BCY581TZZ/6P7BCY581TZZ1.tif'
    crop_size = [224, 224]

    # test ok crop
    # show_number = 50
    # ok_images = crop_ok_image(img_path, crop_size)
    # for index, image in enumerate(ok_images):
    #     if index < show_number:
    #         cv2.imshow(str(index), image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # test ng crop
    defect_point = (6486,3970)
    crop_number = 5
    save_image_dir = 'picture'
    num_class = 2

    crop_image = CropImage(save_image_dir, num_class)
    ng_images = crop_image.crop_ng_image(img_path, defect_point, crop_size, crop_number)
    crop_image.save_image(ng_images, 'test', '1')
    for index, image in enumerate(ng_images):
        cv2.imshow(str(index), image)
    cv2.waitKey()


if __name__ == '__main__':
    main()

