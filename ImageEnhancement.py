import cv2
import numpy as np

def image_enhancement(normalized, background_block=16, histogram_block=32):
    """Correct lighting and enhance local contrast, returning a uint8 iris image.
    Paper section 3.2.3
    """
    #the function takes the 64x512 normalized iris image, corrects uneven lighting, and improves local contrast like the paper describes 
    
    # convert the image to floating point since the next operation needs to subtract estimated background brightness
    image = normalized.astype(np.float32)
    height, width = image.shape

    # estimate background brightness using 16x16 blocks
    background = np.zeros((height // background_block, width // background_block), dtype=np.float32)
    #it creates a gird of 4 blocks and 32 block columns

    # the loops fill each grid cell with the corresponding image block's average brightness
    for row in range(background.shape[0]):
        for col in range(background.shape[1]):
            #here col and row identify a block and they locate that block in the original image
            y = row * background_block
            x = col * background_block
            block = image[y:y + background_block, x:x + background_block]

            #averaging reduces fine texture, giving a rough estimate of broad lighting differences.
            background[row, col] = block.mean()

    # each entry in background summarizes a whole 16 × 16 region.
    # to subtract an estimated background value at every pixel, we first expand this small grid to the image’s dimensions:
    illumination = cv2.resize(background, (width, height), interpolation=cv2.INTER_CUBIC)
    corrected = image - illumination


    # after subtraction, the values might occupy a range such as -20 to 30
    # so we rescale that range and convert the datatype
    # for the input of the function: corrected is the input image
    # None lee openCV allocate the output array
    # 0,255 is the desired min and max
    # Norm_minmax is what uses the min-max scaling
    # CV_8U is the desired output type, which is 8-bit unsigned integer
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # then we qualize each 32x32 block independently to bring out local texture.
    enhanced = np.zeros_like(corrected)
    # the loops visit nonoverlapping blocks and write them to the same positions.
    for y in range(0, height, histogram_block):
        for x in range(0, width, histogram_block):
            block = corrected[y:y + histogram_block, x:x + histogram_block]
            enhanced[y:y + histogram_block, x:x + histogram_block] = cv2.equalizeHist(block)
    return enhanced
