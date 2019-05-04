from hustl import utils

# img = utils.read_img("data/CV Final Project/Data_0/DSC_2973.JPG")
imgs = utils.read_imgs("data/CV Final Project/Data_0/DSC_2973.JPG",
                       "data/CV Final Project/Data_0/DSC_2974.JPG",
                       "data/CV Final Project/Data_0/DSC_2975.JPG")

res = map(utils.extract_sift_features, imgs)
fd = []
for r in res:
    fd.append(r[1])

matches = utils.match_features(*fd)

print(matches)
