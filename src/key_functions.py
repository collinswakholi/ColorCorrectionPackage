import cv2
import colour
import colour.plotting
from colour_checker_detection import detect_colour_checkers_segmentation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Tuple 
import itertools    
import glob
import json

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import check_X_y

import threading
import psutil

import re
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.base import BaseEstimator, RegressorMixin


from utils.logger_ import log_, match_keywords
import gc

is_cuda = torch.cuda.is_available()
# is_cuda = False
device_ = torch.device("cuda" if is_cuda else "cpu")
# print device Name
if is_cuda:
    device_name = torch.cuda.get_device_name(0)
    # print(f"Using GPU: {device_name}")
else:
    device_name = "CPU"
    # print("Using CPU")

# torch.set_num_threads(1)

torch.backends.cuda.matmul.allow_tf32 = True

gc.enable()



__all__ = [
    'FF', 'WP', 'CMFS', 'REF_ILLUMINANT', 'n_proc',
    'metrics', 'Regressor_Model', 'to_float64', 'to_uint8',
    'get_illuminant_from', 'srgb_to_cielab_D50', 'extract_color_chart', 'extract_color_charts', 'extract_color_chart_ex',
    'do_color_adaptation', 'get_color_patch_size', 'extract_neutral_patches',
    'poly_func','extrapolate_if_saturated_mat', 'estimate_gamma_profile', 'wb_correction', #'poly_func_cupy',
    'compute_diag', 'color_correction_1', 'color_correction', 'scatter_RGB', 'compute_mae',
    'convert_to_LCHab', 'get_poly_features', 'predict_', 'predict_image', 'nan_if_saturated',
    'fit_model', 'compute_temperature', 'estimate_fit', 'get_metrics', 'convert_to_lab', 'which_is_saturated',
    'load_spectra', 'adapt_chart', 'arrange_metrics', 'get_json_file', #'generate_powers_with_combinations',
    'extrapolate_if_sat_image', 'CustomNN', 'plot_raincloud', 'get_stats', 'save_stats', 'other_plots', 'poly_func_torch',
    'generate_powers_with_combinations_torch', 'check_memory', 'free_memory', 'get_color_chart_hex', 'get_attr',
]


# start gc
gc.enable()

FF = 1e-15
WP = "D65" # default white point
CMFS = "CIE 1931 2 Degree Standard Observer" 
# CMFS = "CIE 1964 10 Degree Standard Observer"
REF_ILLUMINANT = colour.CCS_ILLUMINANTS[CMFS][WP] 
color_chart = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]

n_proc = os.cpu_count()


# checker detection parameters
CDP = cv2.mcc.DetectorParameters().create()
CDP.adaptiveThreshWinSizeMin = 5
CDP.adaptiveThreshWinSizeStep = 8
CDP.confidenceThreshold = 0.50
CDP.maxError = 0.15
CDP.minGroupSize = 7 

sns.set_theme(style="whitegrid")

def get_attr(attr, key, default=None):
    if attr is None:
        return default
    if isinstance(attr, dict):
        return attr.get(key, default)
    return getattr(attr, key, default)

def adapt_chart(CHART, ILLUMINANT):
    if np.abs(np.array(CHART.illuminant) - np.array(ILLUMINANT)).sum() < 1e-4:
        return CHART
    
    else:
        data_xyz = colour.xyY_to_XYZ(list(CHART.data.values()))
        data_xyz_adapted = colour.adaptation.chromatic_adaptation(
            data_xyz,
            colour.xy_to_XYZ(CHART.illuminant),
            colour.xy_to_XYZ(ILLUMINANT),
        )

        CHART = colour.characterisation.ColourChecker(
            CHART.name+"_Adapted",
            dict(zip(CHART.data.keys(), colour.XYZ_to_xyY(data_xyz_adapted))),
            ILLUMINANT,
            CHART.rows,
            CHART.columns
        )
        return CHART
   


color_chart = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
def get_color_chart_hex(color_chart, REF_ILLUMINANT):
    
    color_chart = adapt_chart(color_chart, REF_ILLUMINANT)

    rgb_values = (255 * np.clip(
        colour.XYZ_to_sRGB(colour.xyY_to_XYZ(list(color_chart.data.values()))), 0, 1)
    ).astype(np.uint8)

    names = list(color_chart.data.keys())
    hex_values = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in rgb_values]

    hex_dict = dict(zip(names, hex_values))

    return {k: v for k, v in sorted(hex_dict.items(), key=lambda item: item[0])}

# hex_ = get_color_chart_hex(color_chart, REF_ILLUMINANT)
# print(hex_)



#------------------------------------------------------------------------------------------------------------------------------
# 2. Functions and Classes                                                                                                            
#------------------------------------------------------------------------------------------------------------------------------

# CLASSES
class metrics:
    def __init__(self, deltaE, mse, mae):
        self.deltaE = deltaE
        self.mse = mse
        self.mae = mae

        columns = ['DE_mean', 'DE_max', 'DE_min', 'DE_Q1', 'DE_Q2', 'DE_Q3', 'DE_Q90',
                   'MSE_mean', 'MSE_max', 'MSE_min', 'MSE_Q1', 'MSE_Q2', 'MSE_Q3', 'MSE_Q90',
                   'MAE_mean', 'MAE_max', 'MAE_min', 'MAE_Q1', 'MAE_Q2', 'MAE_Q3', 'MAE_Q90']

        self.Data_summary = pd.DataFrame(columns=columns)
        data = np.array([deltaE, mse, mae]).T

        self.Data = pd.DataFrame(columns=['DeltaE', 'MSE', 'MAE'],
                                 data=data)

        
        self.run()

    def run(self):
        deltaE_IQR = self.compute_iqr(self.deltaE)
        mse_IQR = self.compute_iqr(self.mse)
        mae_IQR = self.compute_iqr(self.mae)
        deltaE_mean = np.mean(self.deltaE)
        deltaE_max = np.max(self.deltaE)
        deltaE_min = np.min(self.deltaE)
        mse_mean = np.mean(self.mse)
        mse_max = np.max(self.mse)
        mse_min = np.min(self.mse)
        mae_mean = np.mean(self.mae)
        mae_max = np.max(self.mae)
        mae_min = np.min(self.mae)

        self.Data_summary.loc[0] = [deltaE_mean, deltaE_max, deltaE_min, *deltaE_IQR, 
                                    mse_mean, mse_max, mse_min, *mse_IQR, 
                                    mae_mean, mae_max, mae_min, *mae_IQR]

    @staticmethod
    def compute_iqr(mat):
        Q1 = np.percentile(mat, 25)
        Q2 = np.percentile(mat, 50)
        Q3 = np.percentile(mat, 75)

        Q90 = np.percentile(mat, 90)
        return Q1, Q2, Q3, Q90
    
    def __str__(self):
        return self
    
    def print(self):
        print(self.Data)


class Regressor_Model:
    def __init__(self):
        self.mtd = 'linear'
        self.model = None
        self.degree = 3
        self.max_iterations = 100
        self.random_state = 42
        self.tol = 1e-6
        self.verbose = False
        self.ncomp = 1
        self.nlayers = 100
        self.param_search = False

        self.hidden_layers = [64, 32, 16]
        self.learning_rate = 0.001
        self.batch_size = 16
        self.use_batch_norm = False
        self.patience = 10
        self.dropout_rate = 0.2
        self.optim_type = 'Adam'


class CustomNN(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layers=[64, 32, 16], optim_type = 'Adam', random_state=42,
                 learning_rate=0.001, max_epochs=1000, tol=1e-6, verbose=False,
                 batch_size=32, patience=10, dropout_rate=0.2, use_batch_norm=False):
        self.input_dim = None
        self.output_dim = None
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.verbose = verbose
        self.batch_size = batch_size
        self.early_stopping = True
        self.reduce_on_plateau = True
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.validation_split = 0.2
        self.patience = patience
        self.optim_type = optim_type
        self.random_state = random_state
        self.temp_path = 'best_model_temp.pth'

        self.model = None
        self.loss_fn = nn.MSELoss()
        self.device = None
        

    def _build_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_(f"Using device: '{self.device}'", 'purple', 'italic') if self.verbose else None
        layers = []
        current_dim = self.input_dim  # Starting dimension: 3 (nxn input)
        
        # Add hidden layers
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))  # Fully connected layer
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim, eps=self.tol))  # Batch normalization
            layers.append(nn.ReLU())  # Activation function
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))  # Dropout
            current_dim = hidden_dim
        
        # Add output layer
        # layers.append(nn.Tanh())
        layers.append(nn.Linear(current_dim, self.output_dim))  # Final output layer: nx1
        
        # print(f"Output dim: {self.output_dim}")

        # Assign the sequential model
        self.model = nn.Sequential(*layers).to(self.device)

        # Print the model architecture
        if self.verbose:
            from torchsummary import summary
            log_('Model architecture:'.upper(), 'green', 'italic', 'bold')
            summary(self.model, input_size=(self.input_dim,))

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)  # Validate the input dimensions
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1] if len(y.shape) > 1 else 1

        self._build_model()  # Build the model architecture

        # Convert data to PyTorch tensors
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                 torch.tensor(y, dtype=torch.float32))
        
        # Split into training and validation sets
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(self.random_state)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=True)

        optimizer_types = {
            'Adam': optim.Adam,
            'SGD': optim.SGD,
            'RMSprop': optim.RMSprop,
            'Adagrad': optim.Adagrad,
            'Adadelta': optim.Adadelta,
            'AdamW': optim.AdamW,
            'Adamax': optim.Adamax,
            'ASGD': optim.ASGD,
            'LBFGS': optim.LBFGS,
            'Rprop': optim.Rprop,
            'SparseAdam': optim.SparseAdam,
        }

        optimizer = optimizer_types[self.optim_type](self.model.parameters(), lr=self.learning_rate)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.90, patience=int(0.15*self.patience)) if self.reduce_on_plateau else None

        best_val_loss = float('inf')
        epochs_no_improve = 0
        n_verbose=50
        interval = int(self.max_epochs/n_verbose)
        interval = 1 if interval == 0 else interval

        for epoch in range(self.max_epochs):
            # Training loop
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation loop
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    predictions = self.model(batch_X)
                    loss = self.loss_fn(predictions, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if self.verbose and epoch % interval == 0:
                log_(f"Epoch {epoch + 1}/{self.max_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}", 'light_blue', 'italic')

            if scheduler:
                scheduler.step(val_loss)

            # model_over_fit if  val_loss > 1.2*train_loss else None
            model_over_fit = val_loss > 0.8*train_loss
            if val_loss < best_val_loss and not model_over_fit:
                log_(f"Saving model with 'validation loss' = {val_loss:.8f}, @ epoch {epoch + 1}", 'green', 'italic')
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.temp_path)
            else:
                epochs_no_improve += 1

            if self.early_stopping and epochs_no_improve >= self.patience:
                if self.verbose:
                    log_(f"Early stopping triggered @ epoch {epoch + 1}", 'yellow', 'italic')
                break

        if os.path.exists(self.temp_path):
            self.model.load_state_dict(torch.load(self.temp_path))
            os.remove(self.temp_path)


        if self.verbose:
            log_('Training complete.', 'green', 'italic', 'bold')

        # release memory
        del train_loader, val_loader, train_dataset, val_dataset, dataset, optimizer, scheduler

        # release cuda resources
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        gc.collect()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            try:
                predictions = self.model(X_tensor)
            except Exception as e:
                log_(f"Error: {e}", 'red', 'italic', 'bold')

                torch.cuda.empty_cache() if self.device == 'cuda' else None
                batch_size = 2
                success = False
                while not success:
                    try:
                        log_(f'Trying to break prediction into batches of {batch_size}', 'yellow', 'italic')
                        predictions = torch.cat([self.model(X_tensor[i:i+batch_size]) for i in range(0, X_tensor.shape[0], batch_size)], dim=0)
                        success = True
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            log_(f"Out of memory error with batch size {batch_size}: {e}", 'red', 'italic')
                            batch_size *= 2
                            torch.cuda.empty_cache() if self.device == 'cuda' else None
                        else:
                            raise e
            finally:
                log_('Prediction complete.', 'green', 'italic', 'bold')

            results = np.array(predictions.cpu().numpy())

        torch.cuda.empty_cache() if self.device == 'cuda' else None

        # do gabbage collection
        gc.collect()

        return results


# FUNCTIONS

memory_lock = threading.Lock()

try:
    # cv2.setNumThreads(0)
    # torch.set_num_threads(1)
    # cv2.ocl.setUseOpenCL(False)
    torch.cuda.set_per_process_memory_fraction(0.99, device=0)
except:
    pass

def free_memory():
    """Attempt to free memory by emptying CUDA cache and running GC.
       Also resets max memory counters.
    """
    with memory_lock:
        if is_cuda:
            torch.cuda.empty_cache()
        gc.collect()

def check_memory():
    with memory_lock:
        ram_usage = psutil.virtual_memory().percent
        if ram_usage > 95:
            log_(f'RAM usage is {ram_usage:.2f}%, waiting for free memory...', 'yellow', 'bold')

        if 'torch' in globals() and torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved() / 1024**2
            max_reserved = torch.cuda.max_memory_reserved() / 1024**2
            vram_usage = (reserved / max_reserved * 100) if max_reserved > 0 else 0.0
            # if vram_usage > 95:
            #     log_(f'VRAM usage is high {vram_usage:.2f}%...', 'yellow', 'bold')
            log_(f'Used GPU memory: {torch.cuda.memory_usage(0):.2f}%', 'light_blue', 'italic')


def get_available_gpu_memory():
    """Returns the available GPU memory in bytes."""
    if torch.cuda.is_available():
        # torch.cuda.synchronize()
        rem_memory = torch.cuda.mem_get_info()[0]
        is_less_than_1gb = rem_memory < 1024**3
        return rem_memory, is_less_than_1gb
    return 0


def get_color_patch_size(color_checker_img: np.ndarray) -> np.ndarray:

    v = cv2.cvtColor(color_checker_img, cv2.COLOR_BGR2HSV)[:, :, 2]
    v = cv2.GaussianBlur(v, (5, 5), 0)
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.01 * color_checker_img.shape[0] * color_checker_img.shape[1] # 1% of image area

    dims = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        
        # Calculate dimensions of the rotated bounding box
        w = np.linalg.norm(box[0] - box[1])
        h = np.linalg.norm(box[1] - box[2])
        
        # Filter based on aspect ratio and minimum area
        if h > 0 and w > 0:
            area = w * h
            if area > min_area and min(w / h, h / w) > 0.8:  # Ratio close to 1 for squares
                dims.append((w, h))
    
    # Return the average dimensions if found, else [0, 0]
    if dims:
        return np.round(np.mean(dims, axis=0))
    return np.array([0, 0])


def detect_refine_charts(detector, img, n_charts, params=CDP):
    # Use a higher count for chart detection
    n = n_charts+1 # increase n for extra chart detections

    def run_detection(current_params):
        detector.process(img, cv2.mcc.MCC24, nc=n, params=current_params)
        list_ = detector.getListColorChecker()
        if len(list_) < 0.5:
            log_('No color chart found in Image', 'red', 'italic', 'warn')
            assert False
        return detector

    try:
        det_ = run_detection(params)
    except Exception:
        log_('Retrying with Adjusted parameters...', 'light_yellow', 'italic', 'warn')
        DP = cv2.mcc.DetectorParameters().create()
        DP.adaptiveThreshWinSizeMin = 3
        DP.adaptiveThreshWinSizeStep = 8
        DP.confidenceThreshold = 0.50
        DP.maxError = 0.35
        DP.minGroupSize = 4
        det_ = run_detection(DP)

    if n_charts == 1:
        return [det_.getBestColorChecker()]

    elif n_charts > 1:
        charts = det_.getListColorChecker()
        charts = sorted(charts, key=lambda c: c.getCost())
        len_ = len(charts)
        # log_(f"Detected {len_} charts.", 'light_yellow', 'italic', 'warn')
        if (len_-1) < n_charts:
            log_(f"Detected {len(charts)-1} charts instead of {n_charts}.", 'yellow', 'italic', 'warn')
            log_('Consider adjusting `cv2.mcc.DetectorParameters()` for improved results.', 'yellow', 'italic', 'warn')
        elif (len_-1) >= n_charts:
            # log_(f"Selecting top {n_charts} charts out of {len_} detected charts based on lowest cost.", 'yellow', 'italic', 'warn')
            charts = sorted(charts, key=lambda c: c.getCost())[:n_charts]

        return charts
    else:
        return None


def extract_color_chart(img, get_patch_size=False):

    img_blur = cv2.medianBlur(img, 5)
    detector = cv2.mcc.CCheckerDetector_create()
    best_checker = detect_refine_charts(detector, img_blur, n_charts=1, params=CDP)[0]

    if best_checker is None:
        log_('No color chart found in Image', 'red', 'italic', 'warn')
        return None, None, None
    
    cdraw = cv2.mcc.CCheckerDraw_create(best_checker)
    img_draw = img.copy()   
    cdraw.draw(img_draw)

    chartSRGB = best_checker.getChartsRGB()
    w,_ = chartSRGB.shape[:2]
    roi = chartSRGB[0:w, 1]

    box_pts = best_checker.getBox()
    x1, x2 = int(min(box_pts[:, 0])), int(max(box_pts[:, 0]))
    y1, y2 = int(min(box_pts[:, 1])), int(max(box_pts[:, 1]))

    rows = roi.shape[0]
    src = chartSRGB[:, 1].reshape(int(rows/3), 1, 3).reshape(24, 3)

    dims = []
    if get_patch_size:
        img_roi = img[y1:y2, x1:x2]
        dims = get_color_patch_size(img_roi)
    
    else:
        return np.array(src), img_draw, dims


# det_params = 
def extract_color_charts(img: np.ndarray, n_charts: int = 1)-> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extracts all color charts from an image and returns them as a list of numpy arrays of shape (24, 3)
    and a numpy array (image with marked charts) of shape (height, width, 3)

    Args:
        img (np.ndarray): an np.uint8 Image to extract color charts from (should be BGR, not RGB)

    Returns:
        Tuple[List[np.ndarray], np.ndarray]: List of color charts and image with marked charts
    """
    img_blur = cv2.medianBlur(img, 5)
    detector = cv2.mcc.CCheckerDetector_create()

    checkers = detect_refine_charts(detector, img_blur, n_charts=n_charts, params=CDP)

    if checkers is None:
        return [], img
        
    charts = []
    img_draw = img.copy()
    

    for i, checker in enumerate(checkers):
        cdraw = cv2.mcc.CCheckerDraw_create(checker)
        cdraw.draw(img_draw)
        # add text to chart
        # left_corner of chart
        box_pts = checker.getBox()
        x1, x2 = int(min(box_pts[:, 0])), int(max(box_pts[:, 0]))
        y1, y2 = int(min(box_pts[:, 1])), int(max(box_pts[:, 1]))

        cv2.putText(img_draw, f"Chart {i+1}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, int(img.shape[0]/600), (0, 0, 255), int(img.shape[0]/400))
        
        # conf_score = checker.getCost()
        # print(conf_score)
        chartSRGB = checker.getChartsRGB()
        w,_ = chartSRGB.shape[:2]
        roi = chartSRGB[0:w, 1]

        rows = roi.shape[0]
        src = chartSRGB[:, 1].reshape(int(rows/3), 1, 3).reshape(24, 3)

        charts.append(np.array(src))

    return charts, img_draw
    

def extract_color_chart_ex(
        img:np.ndarray, # BGR image uint8
        ref:np.ndarray, # RGB 64 bit 24x3 matrix (range 0:1)
        npts:int = 50, # number of points to extract from each color patch
        show:bool = False, # show image with marked color patches
        randomize:bool = True, # randomize the order of color patches
)->tuple[np.ndarray, np.ndarray]:
    
    img_blur = cv2.medianBlur(img, 5)
    # detect chart
    detector = cv2.mcc.CCheckerDetector_create()

    best_checker = detect_refine_charts(detector, img_blur, n_charts=1, params=CDP)[0]

    if best_checker is None:
        log_('No color chart found in Image', 'red', 'italic', 'warn')
        return None, None

    if show:
        img_draw = img.copy()
        cdraw = cv2.mcc.CCheckerDraw_create(best_checker)
        cdraw.draw(img_draw)
        colour.plotting.plot_image(to_float64(img_draw[:,:,::-1]))
    

    box_pts = best_checker.getBox() # for whole box
    x1, x2 = int(min(box_pts[:, 0])), int(max(box_pts[:, 0]))
    y1, y2 = int(min(box_pts[:, 1])), int(max(box_pts[:, 1]))

    # crop box
    img_crop = img[y1:y2, x1:x2,:]

    # colour.plotting.plot_image(img_crop)
    img_crop_rgb = to_float64(img_crop[:,:,::-1])
        
    data = detect_colour_checkers_segmentation(img_crop_rgb, samples=int(1.2*npts), additional_data=True)
    if len(data)<1:
        try:
            log_('Attempting to read color card again... with "n_samples =  1pt"', 'light_yellow', 'italic', 'warn')

            chart, _, _ = extract_color_chart(img_crop)
            # print(chart.shape)

            return np.array(to_float64(chart)), np.array(ref)
        
        except Exception as e:
            log_(e, 'red', 'bold')
            assert False

        
    _, masks, image_, _ = (data[0].values)
    mask_list = list(masks)

    chart_ex = []
    ref_ex = []

    # Process each mask and corresponding reference color
    for mask, ref_color in zip(mask_list, ref):
        x_min, x_max, y_min, y_max = mask
        masked_data = image_[x_min:x_max, y_min:y_max, :].reshape(-1, 3)

        # Compute statistics
        mean_val = np.mean(masked_data, axis=0)
        max_val = np.nanmax(masked_data, axis=0)
        min_val = np.min(masked_data, axis=0)

        # Sample random points from the masked data
        sampled_data = masked_data[np.random.choice(masked_data.shape[0], npts-3, replace=False)]

        # Concatenate data
        all_data = np.vstack([sampled_data, mean_val, max_val, min_val])

        # Extend reference to match data size
        extended_ref = np.broadcast_to(ref_color, all_data.shape)

        # Append results
        chart_ex.append(all_data)
        ref_ex.append(extended_ref)

    if randomize:
        idx = np.random.permutation(len(chart_ex))
        chart_ex = [chart_ex[i] for i in idx]
        ref_ex = [ref_ex[i] for i in idx]

    return np.vstack(chart_ex), np.vstack(ref_ex)


def compute_temperature(rgb):

    rgb = np.array(rgb) 

    if rgb.ndim == 1:
        rgb = rgb[np.newaxis, :]

    X = -0.14282 * rgb[:, 0] + 1.54924 * rgb[:, 1] - 0.95641 * rgb[:, 2]
    Y = -0.32466 * rgb[:, 0] + 1.57837 * rgb[:, 1] - 0.73191 * rgb[:, 2]
    Z = -0.68202 * rgb[:, 0] + 0.77073 * rgb[:, 1] + 0.56332 * rgb[:, 2]

    # sRGB to XYZ
    # XYZ = colour.sRGB_to_XYZ(rgb)*255.0
    # X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

    denominator = X + Y + Z
    x = X / denominator
    y = Y / denominator
                                     
    n = (x - 0.3320) / (0.1858 - y ) 

    CCT = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33

    return np.mean(CCT),  np.mean(x), np.mean(y)


def to_uint8(img: np.ndarray)-> np.ndarray:
    return (img*255).astype(np.uint8)


def to_float64(img: np.ndarray)-> np.ndarray:
    return (img.astype(np.float64)/255.0)


def extract_neutral_patches(img: np.ndarray, return_one: bool=True, show: bool=False):

    patch_names = list(colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014'].data.keys())
    names_rows = patch_names[-6:]
    names_columns = ['R', 'G', 'B']

    if return_one:
        src_charts, img_draw, _ = extract_color_chart(img)
    else:
        src_charts, img_draw = extract_color_charts(img, n_charts=3, get_patch_size=True)


    if show:
        colour.plotting.plot_image(to_float64(img_draw[:, :, ::-1]), title=f'Charts found: {len(src_charts)}')

    if src_charts is None:
        log_('No charts detected in Image', 'red', 'italic')
    
    if return_one:
        # will return only the first chart
        srgb_chart = to_float64(src_charts)
        n_patch = srgb_chart[-6:, :]
        n_patch_df = pd.DataFrame(n_patch, columns = names_columns, index = names_rows)
        n_patch_all = pd.DataFrame(srgb_chart, columns = names_columns, index = patch_names)

        return n_patch_df, n_patch_all
    
    else:
        # will return all charts in the image as a list
        Neutral_Patches = []
        All_Patches = []

        for chart in src_charts:
            srgb_chart = to_float64(chart)

            n_patch = srgb_chart[-6:, :]
            n_patch_df = pd.DataFrame(n_patch, columns = names_columns, index = names_rows)
            n_patch_all = pd.DataFrame(srgb_chart, columns = names_columns, index = patch_names)

            Neutral_Patches.append(n_patch_df)
            All_Patches.append(n_patch_all)

        return Neutral_Patches, All_Patches
    

def poly_func(x, coeffs): # horner's method
    result = 0
    for c in coeffs:
        result = result * x + c
    return result


def poly_func_torch(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x)  # Initialize result as a tensor of zeros with the same shape as x
    for c in coeffs:
        result = result * x + c  # Horner's method
    return result


def estimate_fit(measured: np.ndarray, reference: np.ndarray, degree: int = 3):
    measured = measured.flatten()
    reference = reference.flatten()

    # Check for matching dimensions
    if measured.shape != reference.shape:
        raise ValueError("Measured and reference arrays must have the same shape.")
    
    # Create the design matrix for the specified polynomial degree
    X = np.vstack([measured**i for i in range(degree, -1, -1)]).T
    
    # Solve for coefficients using lstsq
    # coeffs, _, _, _ = np.linalg.lstsq(X, reference, rcond=None)
    coeffs = np.linalg.lstsq(X, reference, rcond=None)[0]

    return coeffs
    # return coeffs # Coefficients [a, b, c, d]


def predict_image(img: np.ndarray, coeffs: np.ndarray, ref_illuminant: np.ndarray=colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])-> np.ndarray:
    """
    GPU-accelerated prediction with fallback to CPU, using consistent logic.
    """
    
    H, W, C = img.shape
    img_flat = img.reshape(-1, C)  # Flatten to (N, C)
    img_flat_lab = convert_to_lab(img_flat, illuminant=ref_illuminant, c_space='srgb')
    img_flat_L = img_flat_lab[:, 0]

    if is_cuda:
        try:
            coeffs_copy = coeffs.copy()
            img_flat_gpu = torch.from_numpy(img_flat_L.copy()).to(device_, dtype=torch.float32)
            coeffs_gpu = torch.from_numpy(coeffs_copy).to(device_, dtype=torch.float32)  # Reverse for consistency
            result_gpu = poly_func_torch(img_flat_gpu, coeffs_gpu)
            result = result_gpu.cpu().numpy()
            

            del img_flat_gpu, coeffs_gpu, result_gpu
            gc.collect()

        except Exception as e:
            log_(f'Error: {e}', 'red', 'italic')
            result = poly_func(img_flat_L, coeffs)
    else:
        result = poly_func(img_flat_L, coeffs)

    result_lab = img_flat_lab.copy()
    result_lab[:, 0] = result

    # Convert back to RGB space
    result_srgb = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(result_lab, ref_illuminant), ref_illuminant)
    gc.collect()

    return result_srgb.reshape(H, W, C)  # Reshape to original dimensions


def get_metrics(mat1: np.ndarray, mat2: np.ndarray, illuminant: np.ndarray, c_space: str = 'srgb')-> np.ndarray:

    c_space_list = ['xyy', 'lab', 'xyz', 'srgb']
    c_space = match_keywords(c_space, c_space_list)

    # compute delta E
    mat1_lab = convert_to_lab(mat1, illuminant, c_space)
    mat2_lab = convert_to_lab(mat2, illuminant, c_space)
    deltaE = colour.delta_E(mat1_lab, mat2_lab)

    # mean squared error
    mse = np.mean((mat1 - mat2)**2, axis=1) 

    # mean angle error
    mae = compute_mae(mat1, mat2)

    # print(deltaE.shape, mse.shape, mae.shape)
    
    return metrics(deltaE, mse, mae)


def arrange_metrics(metrics_before, metrics_after, name=''):

    Metrics_summary = pd.DataFrame()
    All_data_pd = pd.DataFrame()
    Metrics_all = {}

    Metrics_summary = pd.concat({
        f'Before_{name}': metrics_before.Data_summary,
        f'After_{name}': metrics_after.Data_summary
    })

    all_data_before = metrics_before.Data
    all_data_after = metrics_after.Data

    # add Before_wb and After_wb to the column names
    all_data_before.columns = [f'Before_{name}_' + col for col in all_data_before.columns]
    all_data_after.columns = [f'After_{name}_' + col for col in all_data_after.columns]

    All_data_pd = pd.concat([all_data_before, all_data_after], axis=1)

    Metrics_all = {
        'Summary': Metrics_summary,
        'All': All_data_pd
    }

    return Metrics_all


def convert_to_lab(mat: np.ndarray, illuminant: np.ndarray = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'], c_space: str = 'xyz')-> np.ndarray:
    c_space_list = ['xyy', 'lab', 'xyz', 'srgb']
    c_space = match_keywords(c_space.lower(), c_space_list)

    if c_space == 'xyy':
        return colour.XYZ_to_Lab(colour.xyY_to_XYZ(mat), illuminant)
    elif c_space == 'lab':
        return mat
    elif c_space == 'xyz':
        return colour.XYZ_to_Lab(mat, illuminant)
    elif c_space == 'srgb':
        return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(mat), illuminant)
    else:
        raise ValueError(f"Invalid colour space: {c_space}")
    

def srgb_to_cielab_D50(srgb_mat: np.ndarray, srgb_mat_illuminant: np.ndarray)-> np.ndarray:
    
    # converts srgb to cie lab (ICC D50 profile, 2 degree standard observer)

    # d50 illuminant
    out_illuminant = colour.CCS_ILLUMINANTS[CMFS]['D50']

    xyz_mat = colour.sRGB_to_XYZ(srgb_mat, srgb_mat_illuminant)
    # chromatic adaptation
    xyz_mat = colour.adaptation.chromatic_adaptation(
        xyz_mat, 
        colour.xy_to_XYZ(srgb_mat_illuminant),
        colour.xy_to_XYZ(out_illuminant),
        )
    lab_mat = colour.XYZ_to_Lab(xyz_mat, out_illuminant)
    
    return lab_mat


def convert_to_LCHab(mat: np.ndarray, illuminant: np.ndarray, c_space: str = 'xyz')-> np.ndarray:
    c_space_list = ['xyy', 'lab', 'xyz', 'srgb']
    c_space = match_keywords(c_space, c_space_list)

    if c_space == 'xyy':
        return colour.Lab_to_LCHab(colour.xyY_to_XYZ(mat))
    elif c_space == 'lab':
        return colour.Lab_to_LCHab(mat)
    elif c_space == 'xyz':
        return colour.Lab_to_LCHab(colour.XYZ_to_Lab(mat,illuminant))
    elif c_space == 'srgb':
        return colour.Lab_to_LCHab(colour.XYZ_to_Lab(colour.sRGB_to_XYZ(mat), illuminant))
    else:
        raise ValueError(f"Invalid colour space: {c_space}")


def which_is_saturated(mat: np.ndarray, threshold: float=0.99) -> Tuple[bool, np.ndarray]:
    sat_idx = (mat >= threshold).any(axis=1).astype(int)  # Example logic
    return sat_idx.any(), sat_idx


def extrapolate_if_saturated_mat(mat: np.ndarray, mat_ref: np.ndarray, n_proc: int = 4) -> np.ndarray:
    """
    Extrapolates values for rows in `mat` that are saturated (>=0.99) using `mat_ref` as reference.
    
    Parameters:
    - mat: Input matrix to extrapolate values for.
    - mat_ref: Reference matrix for fitting regression.
    - n_proc: Number of processors available (default: 4).
    
    Returns:
    - Modified `mat` with extrapolated values for saturated rows.
    """
    # Determine saturation
    sat_bool, sat_idx = which_is_saturated(mat, 0.99)
    
    if sat_bool:
        # Find rows with saturation

        idx_rows = np.where(sat_idx == 1)[0]

        log_(f'Extrapolating for {mat[idx_rows, :]}', 'cyan', 'italic')
        log_(f'Saturated rows: [{idx_rows}], have intensity > 0.99', 'cyan', 'italic') 
        
        # Exclude saturated rows for regression
        mat_reduced = np.delete(mat, idx_rows, axis=0)
        mat_ref_reduced = np.delete(mat_ref, idx_rows, axis=0)
        
        # Fit linear regression
        model_ = LinearRegression(
            fit_intercept=True, 
            n_jobs=max(1, n_proc - int(n_proc * 0.2)),  # Ensure at least 1 job
            positive=True
        )
        lstsq_ = model_.fit(mat_ref_reduced, mat_reduced)
        
        # Predict new values for saturated rows
        new_mat = lstsq_.predict(mat_ref)
        
        # Replace saturated rows in `mat` with predicted values
        mat[idx_rows] = new_mat[idx_rows]
    
    return mat


def extrapolate_if_sat_image(img: np.ndarray, mat_ref: np.ndarray) -> np.ndarray:
    # img is a 64bit float (mxnx3) matrix
    img_bgr = to_uint8(img[:,:,::-1])
    _, cps_before = extract_neutral_patches(img_bgr, return_one=True)
    cps = cps_before.values

    sat_bool, sat_idx = which_is_saturated(cps, threshold=0.99)
    
    if not sat_bool:
        return img, None, None

    
    shape_ = img.shape
    img_flat = img.reshape(-1, 3)  # Flatten to (N, 3)

    # ID rows with saturation
    idx_rows = np.where(sat_idx == 1)[0]
    log_(f'Extrapolating for {cps[idx_rows, :]}', 'cyan', 'italic')
    log_(f'Saturated rows: [{idx_rows}], have intensity > 0.99', 'cyan', 'italic') 
    
    # Exclude saturated rows for regression
    mat_reduced = np.delete(cps, idx_rows, axis=0)
    mat_ref_reduced = np.delete(mat_ref, idx_rows, axis=0)
    
    # Fit linear regression
    model_ = LinearRegression()
    lstsq_ = model_.fit(mat_ref_reduced, mat_reduced)
    
    # Predict new values for saturated rows
    new_img_flat = lstsq_.predict(img_flat)
    # should not extrapolate beyond -0.2 and 1.3
    new_img_flat[new_img_flat > 1.3] = 1.3
    new_img_flat[new_img_flat < -0.2] = -0.2

    # normalize image to fit between 0 and 1
    new_img_flat = (new_img_flat - new_img_flat.min()) / (new_img_flat.max() - new_img_flat.min())
    new_img = new_img_flat.reshape(shape_)

    return new_img, cps[idx_rows, :], idx_rows


def nan_if_saturated(mat: np.ndarray)-> np.ndarray:

    sat_bool, sat_idx = which_is_saturated(mat, 0.99)
    if sat_bool:
        mat[sat_idx==1] = np.nan

    return mat


def generate_powers_with_combinations_torch(features, names, degree=1):
    n_samples, n_features = features.shape
    check_memory()
    # Create the constant term with the same dtype and device as features.
    terms = [torch.ones(n_samples, dtype=features.dtype, device=features.device)]
    f_names = ['1']

    # Precompute feature powers for all combinations.
    feature_powers = {
        (i, d): features[:, i] ** d 
        for i in range(n_features) 
        for d in range(1, degree + 1)
    }

    # For each degree, build the combination terms.
    for d in range(1, degree + 1):
        for comb in itertools.combinations_with_replacement(range(n_features), d):
            unique_comb = sorted(set(comb))  # Only need to loop over unique indices.
            # Start with a tensor of ones (with the same dtype and device as features)
            term = torch.ones(n_samples, dtype=features.dtype, device=features.device)
            for i in unique_comb:
                count = comb.count(i)
                term *= feature_powers[(i, count)]
            terms.append(term)
            # Build the feature name; remove the exponent for 1 (i.e. replace '^1' with '')
            f_names.append(' '.join(f'{names[i]}^{comb.count(i)}' for i in unique_comb).replace('^1', ''))

    # Stack all the terms into a new feature matrix.
    combined_features = torch.stack(terms, dim=1)
    free_memory()
    return combined_features, f_names

def process_in_chunks(RGB, feature_names, degree, n_chunk):
    """Processes the RGB matrix in smaller chunks to avoid OOM errors."""
    X_poly_chunks = []

    # use numpy to split the matrix into chunk equal parts in the zero dimension
    chunks = np.vsplit(RGB, n_chunk)

    for i in range(n_chunk):
        # Extract chunk
        RGB_chunk = chunks[i]
        
        # Move to GPU
        # RGB_torch = torch.from_numpy(RGB).to(device_, dtype=torch.float32)
        RGB_torch = torch.from_numpy(RGB_chunk).pin_memory().to(device_, dtype=torch.float32)
        
        # Compute polynomial features
        X_poly_, names = generate_powers_with_combinations_torch(RGB_torch, feature_names, degree=degree)
        
        # Move back to CPU and store result
        X_poly_chunks.append(X_poly_.cpu().numpy())
        
        # Free memory
        del RGB_torch, X_poly_
        free_memory()
    
    # Combine processed chunks
    return np.vstack(X_poly_chunks), names

def estimate_chunk_size(RGB):
    available_memory, do_chunks = get_available_gpu_memory()
    chunksize = 0
    if do_chunks:
        log_(f'Available GPU memory: {available_memory / 1024**3:.2f} GB is low', 'yellow', 'italic')
        row_size = int(available_memory / (RGB.shape[1] * RGB.dtype.itemsize))
        chunksize = int(available_memory / (row_size * 3)) # safety factor of 3

        log_(f'Spliting matrix into {chunksize} chunks', 'yellow', 'italic')

    return max(chunksize, 1)

def get_poly_features(RGB, degree=1):

    feature_names = ['r', 'g', 'b']
    gc.collect()

    try:
        if is_cuda:
            # GPU case with torch
            chunk_size = estimate_chunk_size(RGB)
            X_poly, names = process_in_chunks(RGB, feature_names, degree, chunk_size)
                
            # chunk_size = int(available_memory / (RGB.shape[1] * RGB.dtype.itemsize
            # RGB_torch = torch.from_numpy(RGB).to(device_, dtype=torch.float32)
            # RGB_torch = torch.from_numpy(RGB).pin_memory().to(device_, dtype=torch.float32)

            # X_poly_, names = generate_powers_with_combinations_torch(RGB_torch, feature_names, degree=degree)
            # X_poly = X_poly_.cpu().numpy()
            free_memory()
        else:
            # CPU case
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(RGB)
            names = poly.get_feature_names_out(feature_names)
    except Exception as e:
        log_(f"Error: {e}", 'light_yellow', 'italic', 'warning')
        log_('Attempting to get polynomial features using CPU', 'yellow', 'italic')

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(RGB)
        names = poly.get_feature_names_out(feature_names)


    gc.collect()

    return X_poly, names


def fit_model(det_p, ref_p, kwargs=None):

    M = Regressor_Model()
    M.mtd = kwargs.get('mtd', 'linear')
    M.degree = kwargs.get('degree', 3)
    M.max_iterations = kwargs.get('max_iterations', 100)
    M.random_state = kwargs.get('random_state', 42)
    M.tol = kwargs.get('tol', 1e-6)
    M.verbose = kwargs.get('verbose', False)
    M.ncomp = kwargs.get('ncomp', 1)
    M.nlayers = kwargs.get('nlayers', 100)
    M.param_search = kwargs.get('param_search', False)

    M.hidden_layers = kwargs.get('hidden_layers', M.hidden_layers)
    M.learning_rate = kwargs.get('learning_rate', M.learning_rate)
    M.batch_size = kwargs.get('batch_size', M.batch_size)
    M.patience = kwargs.get('patience', M.patience)
    M.dropout_rate = kwargs.get('dropout_rate', M.dropout_rate)
    M.optim_type = kwargs.get('optim_type', M.optim_type)
    M.use_batch_norm = kwargs.get('use_batch_norm', M.use_batch_norm)


    X = det_p
    Y = ref_p

    X, _ = get_poly_features(X, degree=M.degree)

    if M.ncomp == -1 or M.ncomp > X.shape[1]:
        M.ncomp = X.shape[1]-1

    if M.mtd == 'linear':
        M.param_search = False


    options = ['linear', 'nn', 'pls', 'custom']

    fit_method = match_keywords(M.mtd, options)

    model_dict = {
        'linear': LinearRegression(
            fit_intercept=True,
            # n_jobs = 4,
            # n_jobs=int(0.9*n_proc), # use 80% of cores
        ),

        'nn': MLPRegressor(
            activation='relu', # relu, tanh, identity, logistic
            solver='adam', # lbfgs, sgd, adam
            learning_rate='adaptive', # constant, invscaling, adaptive
            learning_rate_init=0.001,
            hidden_layer_sizes=(M.nlayers,),
            max_iter=1000 if M.max_iterations == -1 else M.max_iterations,
            shuffle=False,
            random_state=M.random_state,
            tol=M.tol,
            verbose=M.verbose,
            nesterovs_momentum=True,
            early_stopping=True,
            n_iter_no_change=int(M.max_iterations * 0.15),
            validation_fraction=0.15,
        ),
        
        'pls': PLSRegression(
            n_components=M.ncomp,
            max_iter=500 if M.max_iterations == -1 else M.max_iterations,
            tol=M.tol,
        ),

        'custom': CustomNN(
            hidden_layers=M.hidden_layers,
            optim_type=M.optim_type,
            learning_rate=M.learning_rate,
            max_epochs=M.max_iterations,
            batch_size=M.batch_size,
            patience=M.patience,
            use_batch_norm=M.use_batch_norm,
            tol=M.tol,
            verbose=M.verbose,
            dropout_rate=M.dropout_rate,
            random_state=M.random_state,
        )
    }

    if fit_method not in model_dict:
        raise ValueError(f'"{fit_method}" is not a valid model type. Available model types: {model_dict.keys()}')

    model = model_dict[fit_method]

    # determine optimum parameters
    if M.param_search:
        if fit_method == 'pls':
            param_grid = {
                'n_components': range(1, M.ncomp),
                'max_iter': range(500, M.max_iterations, 200),
                'tol': [M.tol],
            }
            search = GridSearchCV(
                model, 
                param_grid, 
                cv=5, 
                scoring='neg_mean_squared_error', 
                n_jobs=int(n_proc*0.8), # use 80% of cores
                verbose=0
            )

        elif fit_method == 'nn':
            param_random = {
                'hidden_layer_sizes': [(i,) for i in range(70, M.nlayers, 10)],
                'activation': ['relu', 'tanh', 'identity', 'logistic'],
                'solver': ['sgd', 'adam'],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': np.linspace(0.0001, 0.1, 10),
                'max_iter': range(500, M.max_iterations, 200),
                'momentum': np.linspace(0.92, 0.999, 10),
                'tol': [M.tol],
                'n_iter_no_change': range(10, int(M.max_iterations*0.2), 5),
            }
            search = RandomizedSearchCV( # random search is faster
                estimator=model,
                param_distributions=param_random,
                n_iter=300,
                cv=5,
                random_state=M.random_state,
                scoring='neg_mean_squared_error',
                refit=True, # use any of the above as the refit metric or set to False or True
                n_jobs=int(0.8*n_proc), # use 80% of cores
            )


        search.fit(X, Y)
        best_params = search.best_params_
        print(f'Best parameters: {best_params}')

        model.set_params(**best_params)

    model.fit(X, Y)
    M.model = model

    return M


def predict_(RGB, M):
    # if RGB has shape of mxnx3, reshape to mx3
    if len(RGB.shape) == 3:
        X = RGB.reshape(-1, 3) 
    else:
        X = RGB

    X, _= get_poly_features(X, degree=M.degree)
    pred = M.model.predict(X)

    return pred.reshape(RGB.shape)


def compute_diag(mat_ref, mat_det, rf = 0.95):

    factors = np.nanmedian(mat_ref/ mat_det, axis=0)
    diag = np.diag(rf*factors)
    
    return diag


def get_illuminant_from(type_='', var=None, CMFS="CIE 1931 2 Degree Standard Observer"):
    type_ = type_.lower()

    illuminants_list = list(colour.CCS_ILLUMINANTS.get(CMFS).data.keys())[:20]

    d65_spectra_dist = colour.SDS_ILLUMINANTS['D65']
    CRI_d65 = colour.colour_rendering_index(d65_spectra_dist)
    CRI = None
    spectra = None

    if var is None:
        if  type_.upper() in illuminants_list:
            CRI = colour.colour_rendering_index(colour.SDS_ILLUMINANTS[type_])
            REF_ILLUMINANT = colour.CCS_ILLUMINANTS[CMFS][type_]
            log_(f'Using illuminant: {type_.upper()}', 'cyan', 'italic')
        else:
            CRI = colour.colour_rendering_index(colour.SDS_ILLUMINANTS['D65'])
            REF_ILLUMINANT = colour.CCS_ILLUMINANTS[CMFS]['D65']
            log_(f'Using Default illuminant: D65', 'cyan', 'italic')
    elif var is not None:
        if type_ == 'spectra':
            var = str(var) # must be a string path
            REF_ILLUMINANT, spectra = load_spectra(var)
            log_(f'Using Spectral Distribution, detected illuminant: x: {REF_ILLUMINANT[0]}, y: {REF_ILLUMINANT[1]}', 'cyan', 'italic')
            
            CRI = colour.colour_rendering_index(spectra)
            
        elif type_ == 'xy':
            # convert xy to illuminant
            REF_ILLUMINANT = np.array(var)
            log_(f'Using xy: x: {REF_ILLUMINANT[0]}, y: {REF_ILLUMINANT[1]}', 'cyan', 'italic')

        elif type_ == 'uv':
            # convert uv to illuminant
            REF_ILLUMINANT = colour.UCS_uv_to_xy(var)
            log_(f'Computed from uv* chromaticity values: x: {REF_ILLUMINANT[0]}, y: {REF_ILLUMINANT[1]}', 'cyan', 'italic')
            
        elif type_ == 'detect':
            # dectect illuminant from neutral patches
            np_,_ = extract_neutral_patches(to_uint8(var))
            n_6_5 = np_.iloc[2, :]
            # T, x, y = compute_temperature(np_.values)
            T, x, y = compute_temperature(n_6_5.values)
            print(T, x, y)
            # T, x, y = compute_temperature_2(n_6_5.values)
            log_(f'Detected Temperature from neutral patches, Temperature: {T.astype(int)}K, \tx: {x}, \ty: {y}', 'cyan', 'italic')

            REF_ILLUMINANT = np.array([x, y])
        else:
            REF_ILLUMINANT = colour.CCS_ILLUMINANTS[CMFS]['D65']
            log_(f'Using Default illuminant: D65', 'cyan', 'italic')
    
    if CRI is not None:
        der = ''
        if spectra is None:
            spectra = colour.XYZ_to_sd(colour.xy_to_XYZ(REF_ILLUMINANT))
            der = '(Derived from xy)'
        SSI = colour.spectral_similarity_index(
                spectra,
                d65_spectra_dist,
                False
            )
        log_(f'Current Illuminant CRI: {CRI}', 'light_green', 'italic')
        log_(f'D65 CRI: {CRI_d65}', 'light_green', 'italic')
        log_(f'Spectral{der} Similarity Index between Current Illuminant and D65: {SSI}', 'green', 'italic')


    return REF_ILLUMINANT


def load_spectra(path):
    """
    Load spectral data from a file (a CSV from Sekonic Spectrometer), compute its spectral distribution, 
    and return the corresponding xy chromaticity coordinates.
    
    Parameters:
        path (str): Path to the spectral data file.
    
    Returns:
        tuple or None: xy chromaticity coordinates, or None if file is not found.
    """
    if not os.path.exists(path):
        log_(f'File not found: {path}', 'red', 'bold')
        return None

    # Load the data
    spectra = pd.read_csv(path, header=None, delimiter=";")
    
    # Extract rows containing "Spectral" data
    spectral_rows = spectra[0].loc[spectra[0].str.contains("Spectral")]

    # Split the data into x (wavelength) and y (intensity)
    spectra_x, spectra_y = zip(*[
        (float(row.split('Data ')[-1].split('[')[0]), float(row.split(",")[1]))
        for row in spectral_rows
    ])

    # Remove duplicates while keeping the first occurrence
    spectra_x = np.array(spectra_x)
    spectra_y = np.array(spectra_y)
    _, unique_indices = np.unique(spectra_x, return_index=True)

    spectra_x = spectra_x[unique_indices]
    spectra_y = spectra_y[unique_indices]

    # Create spectral distribution
    Spectra_dict = dict(zip(spectra_x, spectra_y))
    Spectral_distribution = colour.SpectralDistribution(Spectra_dict)

    # Compute xy chromaticity
    xy = colour.XYZ_to_xy(colour.sd_to_XYZ(Spectral_distribution))

    return xy, Spectral_distribution


def estimate_gamma_profile(
        img_rgb: np.ndarray, # RGB image to be corrected (64bit floats, 0-1, RGB)
        ref_cp: np.ndarray, # reference color patch RGB values (64bit floats, 0-1, RGB)
        ref_illuminant: np.ndarray,
        max_degree: int = 7, 
        show: bool = False,
        get_deltaE: bool = False
        ) -> Tuple[np.ndarray, np.ndarray, dict]:  
    
    # redo "find optimal gamma profile" # find optimal degree
     
    Metrics_all = {}

    ref_lab = convert_to_lab(ref_cp, ref_illuminant, c_space='srgb')
    ref_L = ref_lab[:,0]
    # print(ref_L)

    img_bgr = to_uint8(img_rgb[:,:,::-1])
    _, cps_before = extract_neutral_patches(img_bgr, return_one=True)
    values_cps = cps_before.values  # values 0:1

    values_lab = convert_to_lab(values_cps, ref_illuminant, c_space='srgb')
    values_L = values_lab[:,0]


    values_L_ = np.insert(values_L, 0, [100], axis=0)
    ref_L_ = np.insert(ref_L, 0, [100], axis=0)
    
    values_L_ = np.append(values_L_, [0], axis=0)
    ref_L_ = np.append(ref_L_, [0], axis=0)

    Coeffs = []
    MSE = []

    range_ = range(1, max_degree+1)
    # print(f'range_: {range_}')
    
    for degree in range_:
        coeffs = estimate_fit(values_L_, ref_L_, degree)
        Coeffs.append(coeffs)

        pred_ = poly_func(values_L, coeffs)

        mse = np.mean((ref_L - pred_)**2)
        # print(f'degree: {degree}), mse: {mse}')
        MSE.append(mse)

    grad_MSE = np.gradient(MSE)
    # print(f'grad_MSE: {grad_MSE}')
    
    min_idx = np.argmin(grad_MSE)

    def filter_fn(grads, min_idx, threshold=2):
        grads = np.array(grads)
        while True and min_idx>0:
            before_min_grad = grads[:min_idx]
            range_min_grad = np.max(before_min_grad) - grads[min_idx]

            try:
                if range_min_grad<threshold:
                    min_idx = np.argmin(before_min_grad)
                    print(f'adjusted min_idx: {min_idx}')
                else:
                    break
            except:
                break

        return min_idx

    min_idx = filter_fn(grad_MSE, min_idx, threshold=1.7)

    degree = min_idx + 1
    coeffs = Coeffs[min_idx]

    log_(f'Optimal polynomial degree: {degree}', 'cyan', 'italic')

    # find optimal degree
    # coeffs = estimate_fit(values_L_, ref_L_, degree)
    
    eq = "F(x) = "+"+".join(f"{round(c, 2)}x^{i}" for i, c in enumerate(reversed(coeffs)))
    # print(eq)

    # apply gamma profile to image
    img_rgb_gc = predict_image(img_rgb, coeffs, ref_illuminant=ref_illuminant)

    if get_deltaE:

        img_bgr2 = to_uint8(img_rgb_gc[:,:,::-1])
        _, cps_after = extract_neutral_patches(img_bgr2)
        
        metrics_b = get_metrics(ref_cp, values_cps, ref_illuminant, 'srgb')
        metrics_a = get_metrics(ref_cp, cps_after.values, ref_illuminant, 'srgb')

        Metrics_all = arrange_metrics(metrics_b, metrics_a, name='GC')
    
    if show:
        if is_cuda:
            # prediction = poly_func_cupy(
            #     cp.asarray(values_L.flatten()), 
            #     cp.asarray(coeffs)
            # )
            check_memory()
            free_memory()
            prediction = poly_func_torch(
                torch.from_numpy(values_L.flatten()).to(device_, dtype=torch.float32),
                torch.from_numpy(coeffs).to(device_, dtype=torch.float32)
            )
            prediction = prediction.cpu().numpy()
            # prediction = cp.asnumpy(prediction)

            free_memory()
        else:
            prediction = poly_func(values_L.flatten(), coeffs)
        prediction = prediction.reshape(values_L.shape)
        # print(prediction)

        # max__ = np.max([ref_L, values_L, prediction])
        # min__ = np.min([ref_L, values_L, prediction])
        # Plot results
        plt.scatter(ref_L, values_L, c='red', label="Measured vs. Reference")
        plt.xlim(-1, 101)
        plt.ylim(-1, 101)
        plt.xlabel('Reference')
        plt.ylabel('Measured')
        
        # Line of best fit
        x_line = np.linspace(0, 100, 25)
        y_line = poly_func(x_line, coeffs)
        plt.plot(y_line, x_line, 'm-.', label=f"Order {degree} Curve")
        plt.plot(x_line, x_line, 'c--', label=f"1:1 Curve")
        
        # Plot scatter of predictions
        plt.scatter(ref_L, prediction, c='blue', label="Predictions")
        plt.legend()
        plt.show()

        figure = plt.figure( figsize=(10, 5) )
        ax = figure.add_subplot(1, 2, 1)
        ax.imshow(img_rgb)
        ax.set_title("Original Image")

        ax = figure.add_subplot(1, 2, 2)
        ax.imshow(img_rgb_gc)
        ax.set_title("Gamma Corrected Image")

        plt.show()

    gc.collect()

    return coeffs, img_rgb_gc, Metrics_all


def wb_correction(
        img_rgb: np.ndarray, # RGB image to be corrected (64bit floats, 0-1, RGB)
        ref_cp: np.ndarray, # reference RGB values (64bit floats, 0-1, RGB)
        ref_illuminant: np.ndarray,
        show: bool = False,
        get_deltaE: bool = False
        )-> Tuple[np.ndarray, np.ndarray, dict]:
    
    Metrics_all = {}
    ref_np = ref_cp[-6:]
    
    img_bgr = to_uint8(img_rgb[:,:,::-1])
    values_, cps_before = extract_neutral_patches(img_bgr, return_one=True)
    values_np = values_.values

    diag_ = compute_diag(ref_np, values_np)

    img_pred = img_rgb @ diag_

    # print(f'max: {np.max(img_pred)}')
    # print(f'min: {np.min(img_pred)}')

    if get_deltaE:
        m_b = get_metrics(ref_cp, cps_before.values, ref_illuminant, 'srgb')
        _, cps_after = extract_neutral_patches(to_uint8(img_pred[:,:,::-1]))
        m_a = get_metrics(ref_cp, cps_after.values, ref_illuminant, 'srgb')

        Metrics_all = arrange_metrics(m_b, m_a, name='WB')
        

    if show:
        figure = plt.figure( figsize=(10, 5) )
        ax = figure.add_subplot(1, 2, 1)
        ax.imshow(img_rgb)
        ax.set_title("Original Image")

        ax = figure.add_subplot(1, 2, 2)
        ax.imshow(img_pred)
        ax.set_title("WB Corrected Image")

        plt.show()

    return diag_, img_pred, Metrics_all


# method 1
def color_correction_1(
        img_rgb: np.ndarray,
        ref_rgb: np.ndarray,
        ref_illuminant: np.ndarray,
        cc_kwargs: dict = None,
        show: bool = False,
        get_deltaE: bool = False
        )-> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    
    Metrics_ = {}
    corrected_color_card = None

    img_bgr = to_uint8(img_rgb[:,:,::-1])
    _, color_patches = extract_neutral_patches(img_bgr, show=show)

    # cp_names = color_patches.index
    cp_values = color_patches.values

    ccm = colour.characterisation.matrix_colour_correction(
        M_T = cp_values,
        M_R = ref_rgb,
        **cc_kwargs
    )

    log_(f'CCM: \n{ccm}', 'light_green', 'italic')

    img_rgb_ccm = colour.characterisation.apply_matrix_colour_correction(
        RGB=img_rgb,
        CCM=ccm,
        **cc_kwargs
    )
    img_rgb_ccm = np.clip(img_rgb_ccm, 0, 1)

    img_bgr_ccm = to_uint8(img_rgb_ccm[:,:,::-1])
    
    _, color_patches_corrected = extract_neutral_patches(img_bgr_ccm)

    if get_deltaE:
        m_b = get_metrics(ref_rgb, cp_values, ref_illuminant, 'srgb')
        m_a = get_metrics(ref_rgb, color_patches_corrected.values, ref_illuminant, 'srgb')

        Metrics_ = arrange_metrics(m_b, m_a, name='CC_M1')
        
    if show:
        
        # create corrected color card
        corrected_color_card = colour.characterisation.ColourChecker(
            name = "Corrected Colour Checker",
            data = dict(zip(color_patches_corrected.index, colour.XYZ_to_xyY(
                colour.sRGB_to_XYZ(
                    color_patches_corrected.values, ref_illuminant
                )))),
            illuminant=ref_illuminant,
            rows=4,
            columns=6
        )

        

        figure = plt.figure( figsize=(10, 5) )
        ax = figure.add_subplot(1, 2, 1)
        ax.imshow(img_rgb)
        ax.set_title("Original Image")

        ax = figure.add_subplot(1, 2, 2)
        ax.imshow(img_rgb_ccm)
        ax.set_title("CCM Corrected Image")

        plt.show()

        scatter_RGB(
            reference=ref_rgb,
            mats={'Original': cp_values, 'Corrected': color_patches_corrected.values},
            point_lw=1,
            maker_size=200,
            best_fit=True,
        )

    return ccm, img_rgb_ccm, corrected_color_card, Metrics_


# method 2
def color_correction(
        img_rgb: np.ndarray,
        ref_rgb: np.ndarray,
        ref_illuminant: np.ndarray,
        cc_kwargs: dict = None,
        show: bool = False,
        get_deltaE: bool = False,
        n_samples: int = 50,
        )-> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    
    Metrics_ = {}
    corrected_color_card = None

    img_bgr = to_uint8(img_rgb[:,:,::-1])

    _, color_patches = extract_neutral_patches(img_bgr, show=show if n_samples == 1 else False)
    cp_values = color_patches.values

    if n_samples == 1:
        ref_ex = ref_rgb
        cp_values_ex = cp_values
    else:
        chart_ex, ref_ex = extract_color_chart_ex(img_bgr, ref=ref_rgb, npts=n_samples, show=show, randomize=True)
        cp_values_ex = chart_ex


    M_RGB = fit_model(
        det_p=cp_values_ex,
        ref_p=ref_ex,
        kwargs=cc_kwargs
    )

    img_rgb_ccm = predict_(
        RGB=img_rgb,
        M=M_RGB
    )

    img_bgr_ccm = to_uint8(np.clip(img_rgb_ccm[:,:,::-1], 0, 1))
    _, color_patches_corrected = extract_neutral_patches(img_bgr_ccm)

    
    if get_deltaE:
        m_b = get_metrics(ref_rgb, cp_values, ref_illuminant, 'srgb')
        m_a = get_metrics(ref_rgb, color_patches_corrected.values, ref_illuminant, 'srgb')
        Metrics_ = arrange_metrics(m_b, m_a, name='CC_M2')
        
    if show:
        
        # create corrected color card
        corrected_color_card = colour.characterisation.ColourChecker(
            name = "Corrected Colour Checker",
            data = dict(zip(color_patches_corrected.index, colour.XYZ_to_xyY(
                colour.sRGB_to_XYZ(
                    color_patches_corrected.values, ref_illuminant
                )))),
            illuminant=ref_illuminant,
            rows=4,
            columns=6
        )

        figure = plt.figure( figsize=(10, 5) )
        ax = figure.add_subplot(1, 2, 1)
        ax.imshow(img_rgb)
        ax.set_title("Original Image")

        ax = figure.add_subplot(1, 2, 2)
        ax.imshow(np.clip(img_rgb_ccm, 0, 1))
        ax.set_title("CCM Corrected Image")

        if n_samples > 1: # extract the color chart again, if n_samples > 1
            _, color_patches = extract_neutral_patches(img_bgr, show=False)
            cp_values = color_patches.values
            ref_ex = ref_rgb

        scatter_RGB(
            reference=ref_ex,
            mats={'original': cp_values, 'corrected': color_patches_corrected.values},
            point_lw=1.5,
            maker_size=200,
            best_fit=True,
            font_size=14
        )

        plt.show()

    return M_RGB, img_rgb_ccm, corrected_color_card, Metrics_


def scatter_RGB(
    reference: np.ndarray, # reference values of RGB (24x3)
    mats: dict[str, np.ndarray], # dictionary of named matrices, each matrix is 24x3
    point_lw: float = 1.5,
    maker_size: float = 100,
    best_fit: bool = True,
    font_size: float = 14,
    save_ = None,
):
    
    assert reference.shape[1] == 3
    for name, mat in mats.items():
        assert mat.shape == reference.shape, f"Matrix {name} does not have same shape as reference"
    
    # edge_colors = ['r','g','b','c','m','y']
    shapes = ['^','o','v','*','D','p']
    alphas = [1.0, 0.75, 0.5, 0.25, 0.125]
    sizes = maker_size*np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    plt.figure(figsize=(10,10))
    font_headings = int(1.2*font_size)
    plt.title("RGB Scatter Plot", fontsize=font_headings)
    plt.xlabel("Reference", fontsize=font_headings)
    plt.ylabel("Predicted", fontsize=font_headings)


    for i, (k, v) in enumerate(mats.items()):
        # edge_color_ = edge_colors[i]
        shape_ = shapes[i]
        alpha_ = alphas[i]
        size_ = sizes[i]


        plt.scatter(
            np.mean(reference, axis=1),
            np.mean(v, axis=1),
            marker=shape_,
            color=v,
            s=size_,
            edgecolor=0.25*v,
            linewidth=point_lw,
            alpha=alpha_,
            label=k
        )

        if best_fit:
            degree = 1
            coeffs = estimate_fit(reference, v, degree=degree)

            x = np.linspace(0, 1, 10)
            prediction = poly_func(x, coeffs)
            plt.plot(
                x,
                prediction,
                color=[0.25, 0.25, 0.25],  
                linestyle='--',
                label=f"{k} Order {degree} Best Fit Curve",
                linewidth=2,
                alpha=alpha_
            )
            
    plt.plot(
        reference.flatten(),
        reference.flatten(),
        color='k',
        linestyle='-',
        label=f'1:1 Line',
        linewidth=1.5
    )
    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.box(True)
    plt.tight_layout()
    plt.grid()

    if save_ is not None:
        plt.savefig(save_, dpi=1200)
    plt.show()


def do_color_adaptation(img, orig_illuminant, dest_illuminant):

    # check if there is a need for adaptation
    delta = np.abs(np.hypot(*np.subtract(orig_illuminant, dest_illuminant)))
    print(f'delta: {delta}')
    if delta < 1e-5:
        img_adapted = img
    else:
        # convert to XYZ
        img_XYZ = colour.sRGB_to_XYZ(img, orig_illuminant)

        # chromatic adaptation
        img_XYZ_adapted = colour.adaptation.chromatic_adaptation(
            img_XYZ, 
            colour.xy_to_XYZ(dest_illuminant),
            colour.xy_to_XYZ(orig_illuminant),
            # colour.xy_to_XYZ(dest_illuminant),
            )
        
        # convert back to sRGB
        img_adapted = colour.XYZ_to_sRGB(img_XYZ_adapted, dest_illuminant)

    return img_adapted


def compute_mae(mat1, mat2):

    mat1 = np.asarray(mat1)
    mat2 = np.asarray(mat2)

    assert mat1.shape == mat2.shape, "Matrices must have same shape"

    mat1_flat = mat1.reshape(-1, mat1.shape[-1])
    mat2_flat = mat2.reshape(-1, mat2.shape[-1])

    norms = np.linalg.norm(mat1_flat, axis=1) * np.linalg.norm(mat2_flat, axis=1)
    dots_prods = np.einsum('ij,ij->i', mat1_flat, mat2_flat)

    dot_product = np.clip(dots_prods / (norms+FF), -1, 1)
    angular_error = np.degrees(np.arccos(dot_product))
    return angular_error
 

def get_json_file(folder):
    details = glob.glob(os.path.join(folder, '*.json'), recursive=True)
    if len(details) == 1:
        # read json file
        with open(details[0], 'r') as f:
            data = json.load(f)

        # print(data)
        return data
    else:
        return False 
    

def plot_raincloud(data_frame, violin_=True, condition_="col", patch_names=None, palette_name=None, fig_size = (16, 9), rotate_=True,
                   x_label="Condition", y_label="DeltaE", title_name="DeltaE vs Condition", save_=None, show=True):

    # Determine the hex color values based on color_chart, REF_ILLUMINANT, and patch_names
    if color_chart is not None and REF_ILLUMINANT is not None:
        hex_values = get_color_chart_hex(color_chart, REF_ILLUMINANT)
        if patch_names is not None:
            hex_values_ = []
            for name in patch_names:
                key = match_keywords(name, hex_values.keys())
                if key is not None:
                    hex_values_.append(hex_values[key])
                else:
                    hex_values_.append("#808080")
                    log_(f'Patch "{name}" not found in color chart', 'yellow', 'italic', 'warn')
        else:
            hex_values_ = list(hex_values.values())
    else:
        hex_values_ = '#808080'

    # Reshape the data for plotting
    if condition_ == "col":
        conditions = data_frame.columns            # Conditions are in columns.
        deltaE_values = data_frame.values.flatten()  # Flatten row-wise.
        condition_labels = np.tile(conditions, data_frame.shape[0])
    else:
        conditions = data_frame.index               # Conditions are in rows.
        deltaE_values = data_frame.values.flatten()
        condition_labels = np.tile(conditions, data_frame.shape[1])

    multiple = len(condition_labels)//len(hex_values_)
    if multiple > 1:

        hex_values_  = np.array(hex_values_)
        if condition_ == "col":
            hex_values_ = np.repeat(hex_values_, multiple, axis=0)
        else:
            hex_values_ = np.tile(hex_values_, multiple)

        hex_values_ = hex_values_.tolist()
        if len(hex_values_) > len(condition_labels):
            hex_values_ = hex_values_[:len(condition_labels)]
        elif len(hex_values_) < len(condition_labels):
            # fill with #808080
            hex_values_ = np.append(hex_values_, ['#808080'] * (len(condition_labels) - len(hex_values_)))

    
    plot_data = pd.DataFrame({
        'Condition': condition_labels,
        'Metric': deltaE_values
    })


    if palette_name is not None:
        palette_ = sns.color_palette(palette_name, n_colors=len(np.unique(conditions)))
    else:
        palette_ = sns.color_palette('Set3', n_colors=len(np.unique(conditions)))

    # Set up the plot
    plt.figure(figsize=fig_size)
    # Overlay a boxplot for context
    sns.boxplot(x='Condition', y='Metric', data=plot_data, width=0.5,
                palette=palette_, hue='Condition',
                fliersize=0, linewidth=1.2, boxprops={'zorder': 2})
    # Optionally overlay a violin plot
    if violin_:
        sns.violinplot(x='Condition', y='Metric', data=plot_data, inner=None, hue='Condition',
                       linewidth=1, width=0.8, palette=palette_,
                       alpha=0.2, zorder=1)
    # Create the scatter (strip) plot where each point gets its unique face color.
    plt.scatter(plot_data['Condition'], plot_data['Metric'], color=hex_values_, edgecolor='black',
                s=100, alpha=0.6, linewidth=1, zorder=3)

    # Customize plot appearance
    plt.title(title_name, fontsize=15)
    plt.xlabel(x_label, fontsize=14)
    plt.xticks(rotation=60) if rotate_ else None
    plt.ylabel(y_label, fontsize=14)
    plt.tight_layout()

    if save_:
        plt.savefig(fname=save_, dpi=1200)
    if show:
        plt.show()
    plt.close()


def get_stats(data: pd.DataFrame, condition_: str = "col")-> tuple:
    # Reshape data into long format
    if condition_ == "row":
        data = data.T

    # Initialize dataframes
    t_test_results = pd.DataFrame()
    anova_results = pd.DataFrame()
    long_data = pd.DataFrame()
    tukey_df = pd.DataFrame()

    long_data = data.reset_index()
    long_data = long_data.melt(id_vars=['index'], value_vars=data.columns, 
                               var_name='Condition', value_name='DeltaE')
    long_data.rename(columns={'index': 'Patch'}, inplace=True)

    # Ensure that 'DeltaE' is numeric
    long_data['DeltaE'] = pd.to_numeric(long_data['DeltaE'], errors='coerce')

    # Drop rows with NaN values in 'DeltaE'
    long_data.dropna(subset=['DeltaE'], inplace=True)

    # Perform t-test
    try:
        t_stat, p_value, dof = sm.stats.ttest_ind(long_data[long_data['Condition'] == long_data['Condition'].unique()[0]]['DeltaE'], 
                                            long_data[long_data['Condition'] == long_data['Condition'].unique()[1]]['DeltaE'])
        # print("T-Test Results:\n", t_test_results)
        t_test_results = pd.DataFrame({'T-Statistic': [t_stat], 'Degrees of Freedom': [dof], 'p-value': [p_value]})
    except Exception as e:
        print("Error occurred during t-test:", e)
        t_test_results = pd.DataFrame()


    # Perform ANOVA
    try:
        formula = 'DeltaE ~ C(Condition)'
        model = sm.formula.ols(formula, data=long_data).fit()
        anova_results = sm.stats.anova_lm(model, typ=2)
        # print("ANOVA Results:\n", anova_results)
    except Exception as e:
        print("Error occurred during ANOVA:", e)
        anova_results = pd.DataFrame()

    # Run Tukey's HSD test if there are multiple unique conditions
    try:
        if long_data['Condition'].nunique() > 1:
            tukey_results = mc.pairwise_tukeyhsd(long_data['DeltaE'], long_data['Condition'])
            tukey_df = pd.DataFrame(tukey_results.summary().data[1:], columns=tukey_results.summary().data[0])
            # print("\nTukey's HSD Results:\n", tukey_df)
        else:
            print("\nNot enough unique groups for Tukey's HSD test.")
            tukey_df = pd.DataFrame()
    except Exception as e:
        print("Error occurred during Tukey's HSD test:", e)
        tukey_df = pd.DataFrame()

    return long_data, anova_results, tukey_df, t_test_results


def save_stats(data:pd.DataFrame, anova_results:pd.DataFrame, tukey_df:pd.DataFrame, t_test_df:pd.DataFrame, save_path:str, title_name:str)->None:

    if not os.path.exists(os.path.join(save_path, 'Stats')):
        os.makedirs(os.path.join(save_path, 'Stats'))

    data.to_excel(os.path.join(save_path, 'Stats', re.sub(r"[(),: ]", "_", title_name) + "_DATA_for_Stats.xlsx")) if not data.empty else None
    anova_results.to_excel(os.path.join(save_path, 'Stats', re.sub(r"[(),: ]", "_", title_name) + "_ANOVA.xlsx")) if not anova_results.empty else None
    tukey_df.to_excel(os.path.join(save_path, 'Stats', re.sub(r"[(),: ]", "_", title_name) + "_TUKEY.xlsx")) if not tukey_df.empty else None
    t_test_df.to_excel(os.path.join(save_path, 'Stats', re.sub(r"[(),: ]", "_", title_name) + "_T_TEST.xlsx")) if not t_test_df.empty else None


def other_plots(data, save_path=None, title_name='', SHOW_=False):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 1. correlation heatmap between steps to check their effect on DeltaE
    try:
        plt.figure(figsize=(10, 8))
        plt.title(title_name)
        sns.heatmap(data[[col for col in data.columns if '_DeltaE' in col]].corr(), annot=True, cmap="coolwarm", robust=True)
        plt.tight_layout()
        # plt.savefig(os.path.join(plots_dir, "Correlation_Heatmap_DeltaE.svg"), dpi=1200) if SAVE else None
        if save_path is not None:
            plt.savefig(os.path.join(save_path, re.sub(r"[(),: ]", "_", title_name) + "_Correlation_Heatmap_DeltaE.svg"), dpi=1200)
        plt.show() if SHOW_ else None
    except Exception as e:
        print("Error occurred during correlation heatmap:", e)


    # 2. Pairplot of different methods to compare their DeltaE distributions
    try:
        plt.figure(figsize=(15, 10))
        sns.pairplot(data, hue='Method', vars=[col for col in data.columns if '_DeltaE' in col], height=3, aspect=1.2)
        # plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, re.sub(r"[(),: ]", "_", title_name) + "_Pairplot_Methods_DeltaE.svg"), dpi=1200)
        plt.show() if SHOW_ else None
    except Exception as e:
        print("Error occurred during pairplot:", e)


    return None
