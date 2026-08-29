from io import BytesIO
from os.path import dirname, join
from time import time
from uuid import uuid4

from cv2 import cvtColor, imwrite, resize, COLOR_GRAY2RGB
from numpy import load as np_load, save as np_save, where as np_where, uint8, min as np_min, max as np_max

from .dirs import datasets_dir


def generate_task_id():
    return str(uuid4()) + "_" + str(time())


def np_arr_to_octet_stream(np_arr):
    response_buffer = BytesIO()
    np_save(response_buffer, np_arr)
    response_stream = response_buffer.getvalue()
    return response_stream


def octet_stream_to_np_arr(octet_stream):
    buffer = BytesIO(octet_stream)
    np_arr = np_load(buffer)
    return np_arr


def get_volume_absolute_path(volume_path):
    return join(datasets_dir, volume_path, "volume.nii.gz")


def get_segmentation_path_from_volume_path(volume_path):
    volume_dir = dirname(volume_path)
    segmentation_path = join(volume_dir, "segmentation.nii.gz")
    return segmentation_path


def print_value_meta(value):
    print(f" Type: {type(value)}")

    try:
        print(f" dtype: {value.dtype}")
    except:
        pass

    try:
        print(f" shape: {value.shape}")
    except:
        pass


def bool_2D_arr_to_image(bool_2D_arr):
    image = np_where(bool_2D_arr, 255, 0).astype(uint8)
    # rgb_image = cvtColor(uint8_2D_arr, COLOR_GRAY2RGB)
    return image


def float32_4D_tensor_to_image(float32_4D_tensor):
    float32_4D_tensor = float32_4D_tensor.clone()
    
    if float32_4D_tensor.requires_grad:
        float32_4D_tensor = float32_4D_tensor.detach()
    
    if float32_4D_tensor.is_cuda:
        float32_4D_tensor = float32_4D_tensor.cpu()
    

    bool_4D_tensor = float32_4D_tensor * 255
    bool_2D_arr = bool_4D_tensor.squeeze().squeeze().numpy().astype(uint8)
    return bool_2D_arr

    image = bool_2D_arr_to_image(bool_2D_arr)
    return image


def float32_4D_tensor_to_multiple_images(float32_4D_tensor):
    float32_4D_tensor = float32_4D_tensor.clone()
    
    if float32_4D_tensor.requires_grad:
        float32_4D_tensor = float32_4D_tensor.detach()
    
    if float32_4D_tensor.is_cuda:
        float32_4D_tensor = float32_4D_tensor.cpu()
    
    uint8_4D_tensor = float32_4D_tensor * 255

    num_images = uint8_4D_tensor.shape[1]
    images = []

    for i in range(num_images):
        image_tensor = uint8_4D_tensor[0][i]
        image = image_tensor.numpy().astype(uint8)
        images.append(image)
    
    return images


def int16_2D_arr_to_image(int16_2D_arr):
    min_val = np_min(int16_2D_arr)
    max_val = np_max(int16_2D_arr)

    normalized_int16_2D_arr = ((int16_2D_arr - min_val) / (max_val - min_val)) * 255
    
    image = normalized_int16_2D_arr.astype(uint8)
    return image


def float32_2D_tensor_to_image(float32_2D_tensor):
    float32_2D_tensor = float32_2D_tensor.clone()
    
    if float32_2D_tensor.requires_grad:
        float32_2D_tensor = float32_2D_tensor.detach()
    
    if float32_2D_tensor.is_cuda:
        float32_2D_tensor = float32_2D_tensor.cpu()
    
    uint8_2D_tensor = float32_2D_tensor * 255
    
    image = uint8_2D_tensor.numpy().astype(uint8)
    return image


def save_debug_images(save_folder, objects):
    print(objects.keys())

    for key, value in objects.items():
        print(f"Saving debug image(s) for type: {key} ...")
        
        if value is None:
            print("    ... no images to save, value is None.")
            continue

        if key == "multi_label_target_prediction":
            num_labels = value.shape[1]

            for i in range(0, num_labels):
                value_image = float32_2D_tensor_to_image(value[0][i])
                
                image_file_path = join(save_folder, f"{key}-{i}.png")
                imwrite(image_file_path, value_image)
        elif key == "multi_label_preprocessed_scribbles_masks":
            num_labels = len(value)
            print(num_labels)

            for i in range(0, num_labels):
                value_image = bool_2D_arr_to_image(value[i])
                
                image_file_path = join(save_folder, f"{key}-{i}.png")
                imwrite(image_file_path, value_image)
        elif key.startswith("multi_label_model_output"):
            num_labels = value.shape[1]

            for i in range(0, num_labels):
                value_image = float32_2D_tensor_to_image(value[0][i])
                
                image_file_path = join(save_folder, f"{key}-{i}.png")
                imwrite(image_file_path, value_image)
        # elif key == "slice":
        #     value_image = int16_2D_arr_to_image(value_image)


    return


    # if key.startswith("multi_label"):
    #     # value_images = float32_4D_tensor_to_multiple_images(value_image)
        
    #     # for i, value_image in enumerate(value_images):
    #     #     image_file_path = join(save_folder, f"{key}-{i}.png")
    #     #     imwrite(image_file_path, value_image)
    #     value_image = bool_2D_arr_to_image(value_image)
    #     imwrite(image_file_path, value_image)
    # else:
    #     return
    #     if key == "preprocessed_slice": # batch of 4 dims, float 32, last 2 dims contain the image
    #         value_image = float32_4D_tensor_to_image(value_image)
    #     elif key == "slice": # int 16 arr
    #         value_image = int16_2D_arr_to_image(value_image)
    #     elif key == "windowed_slice": # uint8 arr
    #         pass
    #     elif key == "model_output": # batch of 4 dims, float 32, last 2 dims contain the image
    #         value_image = float32_4D_tensor_to_image(value_image)
    #     elif key == "postprocessed_model_output": # numpy bool arr of 2 dims
    #         value_image = bool_2D_arr_to_image(value_image)
    #     elif key == "received_model_prev_output": # numpy bool arr of 2 dims
    #         value_image = bool_2D_arr_to_image(value_image)
    #     elif key == "model_prev_output": # ???
    #         continue
    #     elif key == "scribbles_fg_arr": # numpy bool arr of 2 dims
    #         value_image = bool_2D_arr_to_image(value_image)
    #     elif key == "scribbles_bg_arr": # numpy bool arr of 2 dims
    #         value_image = bool_2D_arr_to_image(value_image)
    #     elif key == "postprocessed_model_prev_output": # numpy bool arr of 2 dims
    #         value_image = bool_2D_arr_to_image(value_image)
    #     elif key == "preprocessed_scribbles_fg_arr": # numpy bool arr of 2 dims
    #         value_image = bool_2D_arr_to_image(value_image)
    #     elif key == "preprocessed_scribbles_bg_arr": # numpy bool arr of 2 dims
    #         value_image = bool_2D_arr_to_image(value_image)
    #     elif key.startswith("model_output_iteration"):
    #         value_image = float32_4D_tensor_to_image(value_image)
    #     elif key == "target_prediction":
    #         value_image = float32_4D_tensor_to_image(value_image)
    #     else:
    #         print(f"Encountered object of unconfigured type {key} for saving.")
        
    #     imwrite(image_file_path, value_image)


def resize_np_bool_arr(np_bool_arr, new_size):
    np_uint8_arr = np_bool_arr.astype(uint8) * 255

    resized_uint8_arr = resize(np_uint8_arr, (new_size[1], new_size[0]))
    
    resized_bool_arr = (resized_uint8_arr > 127).astype(bool)
    return resized_bool_arr
