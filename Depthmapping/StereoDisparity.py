import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image

# 기본적인 stereo matching 알고리즘에서 출발
# denoising 알고리즘 역시 적용해야 할 것으로 보임



def stereo_match(left_image, right_image, kernel, max_offset):
    h, w = left_image.shape  # assume that both images are same size

    # Depth (or disparity) map
    depth = np.zeros((h, w), np.uint8)   # 뎁스맵 생성
    # depth.shape = h, w

    disparity = np.zeros((h, w), np.uint8)  # Disparity에 대한 맵 생성
    # disparity.shape = h, w

    kernel_half = int(kernel / 2)  # 커널은 window의 크기.
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
    for y in range(kernel_half, h - kernel_half):  # 가로길이만큼
        # print(".", end="", flush=True)  # let the user know that something is happening (slowly!)

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                # v and u are the x,y of our local window search, used to ensure a good
                # match- going by the squared differences of two pixels alone is insufficient,
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster

                        ssd_temp = int(left_image[y + v, x + u]) - int(right_image[y + v, (x + u) - offset])
                        # if y + u - offset < 0:
                        #     ssd_temp = 0
                        # else:
                        #     ssd_temp = int(left_image[y + v, x + u]) - int(right_image[y + v, (x + u) - offset])
                        # ssd_temp = int(left_image[y + v, x + u]) - int(right_image[(y+v)- offset, x+u])
                        ssd += ssd_temp * ssd_temp

                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust
            disparity[y, x] = best_offset
    # Convert to PIL and save it
    # Image.fromarray(depth).save('depth.png')
    # Image.fromarray(disparity).save('disparity.png')
    return depth


if __name__ == '__main__':
    mode = "V"  # V : vertical direction, H: horizontal directrion

    col = 3  # 0부터 시작
    row = 5

    if mode == "V":
        index = col * lensy + row
        print("This mode is {}".format(mode))
        print("Index is {}".format(index))
    elif mode == "H":
        index = row * lensx + col
        print("This mode is {}".format(mode))
        print("Index is {}".format(index))
    else:
        print("This is nothing")

    image1 = Image.open("view0.png").convert('L')
    img1 = np.asarray(image1)
    image2 = Image.open("view1.png").convert('L')
    img2 = np.asarray(image2)

    # img1 = Total_sub_image[:,:,index]
    # img2 = Total_sub_image[:,:,index+1]

    kernel = 6
    max_offset = 60

    disparity = stereo_match(img1, img2, kernel, max_offset)

    f1, axarr1 = plt.subplots(1, 3, figsize=(10, 10))

    axarr1[0].imshow(img1, cmap=cm.Greys_r)
    axarr1[0].axis('off')
    axarr1[1].imshow(img2, cmap=cm.Greys_r)
    axarr1[1].axis('off')
    axarr1[2].imshow(disparity, cmap=cm.Greys_r)
    axarr1[2].axis('off')

    f1.tight_layout()
    plt.show()