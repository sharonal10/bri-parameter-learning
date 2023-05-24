from PIL import Image
import numpy as np
from copy import deepcopy
import numpy as np
from model import *
import sys
    
def get_bounding_box(coord, x_size, y_size, width=6, height=6):
    assert (width <= x_size and height <= y_size)
    offset_x = 0
    offset_y = 0
    x, y, w, h = (coord[0] - width // 2, coord[1] - height // 2, width, height)
    if x < 0:
        offset_x = x
        x = 0
    if x+w > x_size:
        offset_x = x+w-x_size
        x = x_size-w
    if y < 0:
        offset_y = y
        y = 0
    if y+h > y_size:
        offset_y = y+h-y_size 
        y = y_size-h 
    return (x, y, w, h), offset_x, offset_y

def get_feature_from_box(image, box):
    x, y, w, h = box
    return image[y:y+h, x:x+w, :]

def find_identical_feature(feature, image):
    height, width, _ = feature.shape
    min_mse = float('inf')
    best_match = None
    for y in range(image.shape[0] - height):
        for x in range(image.shape[1] - width):
            candidate = image[y:y+height, x:x+width]
            mse = ((feature - candidate) ** 2).mean()
            if mse < min_mse:
                min_mse = mse
                best_match = (x, y)
                
    return best_match

def predict_coordinate(train_img, train_coord, test_img, patch_size=14):
    concat_img = np.concatenate((train_img, test_img), axis=0)

    concat_img = Image.fromarray(concat_img)
    concat_img = preprocess_rgb(concat_img)
    model = SAMDinoV2Model('cuda')
    
    pca_features = model.forward(concat_img)[0].detach().cpu().numpy()[0, :, :]

    train_coord = (int((train_coord[0]//14)), int((train_coord[1]//14))) 

    y_size, x_size, _ = pca_features.shape
    y_size //= 2 

    bbox, offset_x, offset_y = get_bounding_box(train_coord, x_size, y_size, width=patch_size, height=patch_size)

    feature = get_feature_from_box(pca_features, bbox)

    pca_test_full = deepcopy(pca_features)
    pca_test_full[:y_size, :, :] = 0
    match_coord = find_identical_feature(feature, pca_test_full)
    
    rel_position = (train_coord[0] - bbox[0], train_coord[1] - bbox[1])

    pred_coord = (match_coord[0] + rel_position[0] + offset_x, match_coord[1] + rel_position[1] + offset_y)

    return pca_features, pred_coord 

def run(test_img_fn, task_name):
    task_coords = {
        'pick_bread': (69, 57),
        'place_bread': (137, 148),
        'pick_pot': (291, 130),
        'pour': (175, 47),
        'pick_spoon': (121, 166),
        'place_spoon': (179, 173)
    }
    filepath = rf"train_imgs"

    train_img=Image.open(rf"{filepath}\{task_name}.png")
    train_img = train_img.convert('RGB')
    train_img = np.array(train_img)
    train_coord_orig = task_coords[task_name]
    train_coord = (int((train_coord_orig[0]/0.45)), int((train_coord_orig[1]/0.45))) 

    test_img = Image.open(test_img_fn)
    test_img = test_img.convert('RGB')
    test_img = np.array(test_img)

    _, pred_coord = predict_coordinate(train_img, train_coord, test_img)
    pred_coord = (int((pred_coord[0] * 14 + 7)*0.45), int((pred_coord[1] * 14 + 7)*0.45 - test_img.shape[0]))

    return pred_coord

def generate_samples(coord, std_x=1, std_y=1, num_samples=1000):
    """Generate samples from a 2D normal distribution centered at a given (x, y), returns list of lists"""
    x, y = coord
    sample_1 = np.array([[x,y]])
    samples_n = (np.random.normal([x, y], [std_x, std_y], (num_samples, 2)))
    samples = np.concatenate((sample_1, samples_n))
    return samples

