import cv2
import pandas as pd
import os
from ColorCorrectionPipeline.core import ColorCorrection, Config
from ColorCorrectionPipeline.key_functions import to_float64

IMG_PATH = 'Data/Sample_1.JPG' # image to compute color correction
WHITE_IMAGE_PATH = 'Data/white.JPG' # for doing FFC
TEST_IMAGE_PATH = 'Data/Sample_1.JPG' # image to test color correction, does not need to contain color card

MODEL_PATH = 'Data/plane_det_model_YOLO_512_n.pt' # path to yolo model for plane detection to do FFC
SAVE_PATH = os.path.join(os.getcwd(), 'results') # path to save results if config.save is True


img_rgb = to_float64(cv2.imread(IMG_PATH))[:, :, ::-1]
white_bgr = cv2.imread(WHITE_IMAGE_PATH)
test_img = to_float64(cv2.imread(TEST_IMAGE_PATH))[:, :, ::-1]

img_name = os.path.splitext(os.path.basename(IMG_PATH))[0]

# Edit the parameters here fopr each step

ffc_kwargs = {
    'model_path': MODEL_PATH,
    'manual_crop': False,
    'show': False, # whether to show plots
    'bins': 50, # number of bins to use
    'smooth_window': 5, # window size for smoothing
    'get_deltaE': True, # whether to compute deltaE
    'fit_method': 'pls', # linear, nn, pls, svm  
    'interactions': True, # whether to use interactions
    'max_iter': 1000, # max iterations for fitting
    'tol': 1e-8, # tolerance for fitting
    'verbose': False, # whether to print verbose output
    'random_seed': 0, # random seed
}

gc_kwargs = {
    'max_degree': 5, # maximum fitting degree for gamma correction
    'show': False, # whether to show plots
    'get_deltaE': True, # whether to compute deltaE
}

wb_kwargs = {
    'show': False, # whether to show plots
    'get_deltaE': True, # whether to compute deltaE
}

cc_kwargs = {
    'cc_method': 'ours', # method to use for color correction
    'method': 'Finlayson 2015', # if cc_method is 'conv', this is the method
    'mtd': 'nn', # if cc_method is 'ours', this is the method, linear, nn, pls, svm

    'degree': 2, # degree of polynomial to fit
    'max_iterations': 5000, # max iterations for fitting
    'random_state': 0, # random seed
    'tol': 1e-8, # tolerance for fitting
    'verbose': False, # whether to print verbose output
    'param_search': False, # whether to use parameter search
    'show': False, # whether to show plots
    'get_deltaE': True, # whether to compute deltaE
    'n_samples': 25, # number of samples to use for parameter search

    # only if mtd == 'pls'
    'ncomp': 1, # number of components to use

    # only if mtd == 'nn'
    'nlayers': 100, # number of layers to use
    'hidden_layers': [64, 32, 16], # hidden layers for neural network
    'learning_rate': 0.001, # learning rate for neural network
    'batch_size': 16, # batch size for neural network
    'patience': 10, # patience for early stopping
    'dropout_rate': 0.1, # dropout rate for neural network
    'optim_type': 'adam', # optimizer type for neural network
    'use_batch_norm': True, # whether to use batch normalization

}

config = Config(
    do_ffc=True,
    do_gc=True,
    do_wb=True,
    do_cc=True,
    save = False,
    save_path = "", # os.path.join(os.getcwd(), 'results'),
    check_saturation = True,
    FFC_kwargs=ffc_kwargs,
    GC_kwargs=gc_kwargs,
    WB_kwargs=wb_kwargs,
    CC_kwargs=cc_kwargs
)

cc = ColorCorrection()
metrics, corrected_imgs, errors = cc.run(
    Image=img_rgb,
    White_Image=white_bgr,
    name_=img_name,
    config=config
)
# convert metrics to pandas def
print(metrics)

test_imgs_corrected = cc.predict_image(test_img, show=True)

