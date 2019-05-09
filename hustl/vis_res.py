import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.transform import resize, rescale

def main():
    downscale_factor = 1

    ## original images
    path = '../../cv_img/DSC_3188_'
    names = ["%d" % number for number in np.arange(102, 117, 3)]
    suffix = '.jpg'
    files = [path + name + suffix for name in names]


    ## res images
    res_path = '../../cv_img_color/'
    res_names = ["%d" % number for number in np.arange(102, 117, 3)]
    res_suffix = '_res.jpg'
    res_files = [res_path + name + res_suffix for name in res_names]


    ori_arr = []
    res_arr = []
    print("begin stacking")
    for i in range(len(files)):
        img = io.imread(files[i]) #it reads the image 3 times for some reason
        img = rescale(img, downscale_factor, anti_aliasing=True, multichannel=True, mode='reflect')
        ori_arr.append(img)

        res = io.imread(res_files[i]) #it reads the image 3 times for some reason
        # res = rescale(res, downscale_factor, anti_aliasing=True, multichannel=True, mode='reflect')
        res_arr.append(res)

    ori_output = np.column_stack(ori_arr)
    res_output = np.column_stack(res_arr)
    print("saving")
    io.imsave('../res/' + names[0].split(".")[0] + "_ori_ALL.jpg", img_as_ubyte(ori_output))
    io.imsave('../res/' + res_names[0].split(".")[0] + "_res_ALL.jpg", img_as_ubyte(res_output))
    print("saved")

if __name__ == '__main__':
    main()