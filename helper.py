from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

VGG_MEAN = [123.68, 116.779, 103.939]
# RGB format ^

def resize_and_rescale_img_content(content_image_path, output_path_, output_filename):
    # This will scale down in the range of pixel values from [0-255] to [0-1]   
    if os.path.isfile(content_image_path):
        img = Image.open(content_image_path)
        if not os.path.exists(output_path_):
            os.makedirs(output_path_)
        img.save(output_path_ + output_filename)
        img_array = np.array(img, dtype=np.float32)
        if min(img_array.shape[0], img_array.shape[1]) < 224:
            print("Please use image having higher resolution than 224 x 224.")
        else:
            img_array = np.expand_dims(img_array, 0)
            img_array = img_array / 255
            return img_array
    else:
        print("No image found in given location.")

def resize_and_rescale_img_style(content_image_path, style_image_path, output_path_, output_filename):
    # This will resize the style image as per the dimension of content image and then scale down in the range of pixel values from [0-255] to [0-1]   
    if os.path.isfile(content_image_path):
        c_img = Image.open(content_image_path)
        c_img = np.array(c_img)
        c_h, c_w = c_img.shape[0], c_img.shape[1]
        if os.path.isfile(style_image_path):
            s_img = Image.open(style_image_path)
            s_img_resized = s_img.resize(size=(c_w, c_h))
            if not os.path.exists(output_path_):
                os.makedirs(output_path_)
            s_img_resized.save(output_path_ + output_filename)
            s_img_array = np.array(s_img_resized, dtype=np.float32)
            s_img_array = np.expand_dims(s_img_array, 0)
            s_img_array = s_img_array / 255
            return s_img_array
        else:
            print("Style image not found at a given location.")
    else:
        print("Content image not found at a given location.")

def post_process_and_display(cnn_output, output_path, output_filename, save_file=True):
    # This will take input_noise of (1, w, h, channels) shapped array taken from tensorflow operation
    # and ultimately displays the image
    
    x = np.clip(cnn_output, 0.0, 1.0)
    x = np.squeeze(x)
    x -= np.amin(x)
    x /= (np.amax(x) - np.amin(x))
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    plt.imshow(x)
    plt.show()
    img = Image.fromarray(x, mode='RGB')
    img.show()
    if save_file:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img.save(output_path + output_filename)
    return x