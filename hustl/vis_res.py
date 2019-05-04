import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.transform import resize, rescale

def main():
    downscale_factor = 3/23

    ## original images
    path = '../../CVFinalProj_Data/'
    names = ['DSC_3113.JPG', 'DSC_3114.JPG', 'DSC_3115.JPG', 'DSC_3116.JPG', 'DSC_3117.JPG', 'DSC_3118.JPG']
    files = [path + name for name in names]

    ## res images
    res_path = '../res/'
    res_names = [name.split(".")[0] + "_res.jpg" for name in names]
    res_files = [res_path + name for name in res_names]

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
    io.imsave('../res/' + res_names[0].split(".")[0] + "_ALL.jpg", img_as_ubyte(res_output))
    print("saved")

if __name__ == '__main__':
    main()
