import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.transform import resize, rescale

def main():
    downscale_factor = 3/23

    ## original images
    path = '../../CVFinalProj_Data/naive/'
    names = ['DSC_3188_40', 'DSC_3188_41','DSC_3188_42','DSC_3188_43']
    # names = ['DSC_3113.JPG', 'DSC_3114.JPG', 'DSC_3115.JPG', 'DSC_3116.JPG', 'DSC_3117.JPG', 'DSC_3118.JPG']
    files = [path + name + '.jpg' for name in names]

    ## res images
    res_path = '../../CVFinalProj_Data/optimal/'
    res_names = ['DSC_3188_38', 'DSC_3188_39','DSC_3188_40','DSC_3188_41']
    res_files = [res_path + name + '.jpg' for name in res_names]

    ori_arr = []
    res_arr = []
    print("begin stacking")
    for i in range(len(files)):
        img = io.imread(files[i]) #it reads the image 3 times for some reason
        # img = rescale(img, downscale_factor, anti_aliasing=True, multichannel=True, mode='reflect')
        ori_arr.append(img)

        res = io.imread(res_files[i]) #it reads the image 3 times for some reason
        # res = rescale(res, downscale_factor, anti_aliasing=True, multichannel=True, mode='reflect')
        res_arr.append(res)

    ori_output = np.column_stack(ori_arr)
    res_output = np.column_stack(res_arr)
    print("saving")
    io.imsave('../../CVFinalProj_Data/' + "_naive_stack.jpg", img_as_ubyte(ori_output))
    io.imsave('../../CVFinalProj_Data/' + "_optimal_stack.jpg", img_as_ubyte(res_output))
    print("saved")

if __name__ == '__main__':
    main()
