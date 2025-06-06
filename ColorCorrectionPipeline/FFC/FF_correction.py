import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures

from ColorCorrectionPipeline.utils.logger_ import log_, ThrowDlg, match_keywords
from ultralytics import YOLO
import torch

import gc

gc.enable()

FLOAT = np.float32
UINT8 = np.uint8
cmaps = ['viridis', 'plasma', 'jet', 'Greys', 'cividis']


class FlatFieldCorrection():
    """
    Class for computing flat-field correction from white image.

    Example usage:
    
    ```
    import cv2
    from FFC.FF_correction import FlatFieldCorrection

    if __name__ == "__main__":
        # Example: Load a white-background image and perform FFC
        path_ = r"Data/Backgrounds/White/Blank.JPG"
        w_img = cv2.imread(path_, cv2.IMREAD_COLOR)

        ffc_params = {
            "model_path": "best_models/PD_trained_512_dauntless-sweep-1/weights/best.pt",
            "manual_crop": False,
            "smooth_window": 11,
            "bins": 50,
            "show": True,
        }

        fit_params = {
            "degree": 5,
            "interactions": True,
            "fit_method": "nn",  # linear, nn, pls, svm
            "max_iter": 1000,
            "tol": 1e-8,
            "verbose": False, 
            "rand_seed": 0,
        }

        ffc = FlatFieldCorrection(img=w_img, **ffc_params)
        multiplier = ffc.compute_multiplier(**fit_params)
        # np.save("mult.npy", multiplier)

        corrected_img = ffc.apply_ffc(img=w_img, multiplier=multiplier, show=True)
        ```

    """
    def __init__(self, img=None, **kwargs):
        self.img = img
        self.model_path = kwargs.get('model_path', '')
        self.manual_crop = kwargs.get('manual_crop', False)
        if self.model_path == '':
            self.manual_crop = True
        self.show = kwargs.get('show', False)
        self.bins = kwargs.get('bins', 50)
        self.smooth_window = kwargs.get('smooth_window', 5)
        self.crop_rect = kwargs.get('crop_rect', None)
        self.model = None
        self.img_cropped = None
        self.cropped_multiplier = None
        self.final_multiplier = None
        self.is_color = self.check_color(self.img) if self.img is not None else None

        if not self.manual_crop:
            self.model = YOLO(self.model_path)


    def check_color(self, img):
        self.is_color = img.ndim == 3 and img.shape[2] == 3
        return self.is_color
    
    def resize_image(self, img, factor=None, size=None):
        img_ = img
        if factor is not None:
            img_ = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        if size is not None:
            img_ = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
        
        return img_
    
    def transform_extremity(self, x: np.ndarray, cut_off: float=1.5, max: float=2.0):
        x_ = x.flatten()
        mask = x_ > cut_off
        x_[mask] = cut_off + (max - cut_off) * np.tanh(max * (x_[mask] - cut_off))
        return x_.reshape(x.shape)
    
    def show_results(self, img_correct, img_original):
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        if len(img_correct.shape) == 3:
            ax[0].imshow(cv2.cvtColor(img_correct, cv2.COLOR_BGR2RGB))
            ax[1].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        else:
            img_correct = cv2.normalize(img_correct, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            img_original = cv2.normalize(img_original, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            ax[0].imshow(img_correct, cmap='gray')
            ax[1].imshow(img_original, cmap='gray')
        ax[0].set_title("FF Corrected Image")
        ax[1].set_title("Original Image")
        plt.show()

        return fig

    def plot_intensity_distribution(self, Z, Z_flat, half=False):

        if half:
            shape = Z.shape
            wh_ = shape[0]//2
            hh_ = shape[1]//2
            Z = Z[0:wh_, :]
            Z_flat = Z_flat[0:wh_, :]

        try:
            w,h,_ = Z.shape
        except:
            w,h = Z.shape

        bins = self.bins
        if w > self.bins or h > bins:
            x = np.linspace(0, w-1, bins)
            y = np.linspace(0, h-1, bins)
            X, Y = np.meshgrid(x, y)
            h_win = int((self.smooth_window-1)/2)

            Z_ = np.zeros_like(X)
            Z_flat_ = np.zeros_like(X)

            for i, x_ in enumerate(x):
                for j, y_ in enumerate(y):
                    x_,y_ = int(x_), int(y_)
                    x_bounds = max(0, x_-h_win), min(w, x_+h_win)
                    y_bounds = max(0, y_-h_win), min(h, y_+h_win)

                    Z_[i, j] = np.mean(Z[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]])
                    Z_flat_[i, j] = np.mean(Z_flat[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]])

            Z_ = np.array(Z_)
            Z_flat_ = np.array(Z_flat_)
        else:
            Z_ = Z
            Z_flat_ = Z_flat

        fig = go.Figure(data=[
            go.Surface(z=Z_-10, opacity=1, colorscale='Viridis'),
            go.Surface(z=Z_flat_, opacity=0.3, colorscale='Jet'),
        ])
        fig.update_layout(title='Intensity distribution', autosize=True,
                          margin=dict(l=65, r=50, b=65, t=90),
                          scene=dict(
                              xaxis_title='X',
                              yaxis_title='Y',
                              zaxis_title='Intensity*'))
        fig.update_scenes(zaxis_range=[0, 256])
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                           highlightcolor="limegreen", project_z=True))
        fig.show()

    def plot_multiplier(self, multiplier):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("Multiplier")
        im = ax.imshow(multiplier, cmap='jet')
        plt.colorbar(im, ax=ax, orientation='vertical')
        plt.show()
        return fig
    
    def show_3d(self, img_list, names=None):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, img in enumerate(img_list):
            x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
            randi = np.random.randint(0, len(cmaps))
            p = ax.plot_surface(x, y, img, cmap=cmaps[randi], alpha=0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        if names is not None:
            ax.legend(names)

        fig.colorbar(p, ax=ax)
        plt.show()
        return fig
    
    def detect_and_crop(self):

        if not self.manual_crop:
            sr = 0.95
            results = self.model.predict(
                source=self.img, 
                half=False,
                show=False,
                save=False,
                save_txt=False,
                conf=0.7,
                iou=0.6
            )
            boxes = []
            probs = []
            for result in results:
                box_cpu = np.round(result.boxes.xyxy.cpu().numpy()).astype(int)
                prob_cpu = result.boxes.conf.cpu().numpy()
                boxes.append(box_cpu[0,:])
                probs.append(prob_cpu[0])
            boxes = np.array(boxes)
            probs = np.array(probs)
            if len(boxes) > 1:
                log_(f'{len(boxes)} objects detected', 'light_yellow', 'italic', 'warning')
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                max_index = np.argmax(areas)
                boxes = boxes[max_index]
                probs = probs[max_index]
                log_(
                    f'biggest object ID: {max_index} BB: {boxes} probability: {probs} selected', 
                    'light_yellow', 
                    'italic', 
                    'warning'
                )

            x1,y1,x2,y2 = boxes[0]
            x1 = int(x1 + (1-sr) * (x2 - x1))
            y1 = int(y1 + (1-sr) * (y2 - y1))
            x2 = int(x2 - (1-sr) * (x2 - x1))
            y2 = int(y2 - (1-sr) * (y2 - y1))

        else: # select ROI manually
            log_('Select ROI manually\nPress "ENTER" when done selecting ROI', 'light_blue', 'italic', 'info')
            cv2.namedWindow("Press 'ENTER' when done", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Press 'ENTER' when done", 1200, 800)
            rect = cv2.selectROI("Press 'ENTER' when done", self.img, True)
            cv2.destroyAllWindows()

            try:
                x1 = int(rect[0])
                y1 = int(rect[1])
                x2 = int(rect[0] + rect[2])
                y2 = int(rect[1] + rect[3])
            except:
                log_('ROI not selected', 'light_yellow', 'italic', 'warning')
                x1 = 0
                y1 = 0
                x2 = self.img.shape[1]
                y2 = self.img.shape[0]

        self.crop_rect = [x1, y1, x2, y2]
        self.img_cropped = self.img[y1:y2, x1:x2]

        if self.show:
            img = self.img.copy()
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
            cv2.namedWindow("Image ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image ROI", 1200, 800)
            cv2.imshow("Image ROI", img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del results, boxes, probs
        gc.collect()

    def get_L(self, img, smooth=False):
        is_color = self.check_color(img)
        img_LAB = None
        if is_color:
            img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L = img_LAB[:, :, 0]
        else:
            L = img
        if smooth:
            L = cv2.GaussianBlur(L, (self.smooth_window, self.smooth_window), 0)

        return L, img_LAB
    
    def polynomial_features(self, X, **kwargs):
        degree = kwargs.get('degree', 5)
        interactions = not(kwargs.get('interactions', False))
        poly = PolynomialFeatures(degree=degree, interaction_only=interactions)
        X_poly = poly.fit_transform(X)
        names = poly.get_feature_names_out(['x', 'y'])
        return X_poly, names, poly

    def fit_model(self, X, y, **kwargs):

        method = kwargs.get('fit_method', 'nn')
        max_iter = kwargs.get('max_iter', 1000)
        tol = kwargs.get('tol', 1e-8)
        verbose = kwargs.get('verbose', False)
        rand_seed = kwargs.get('rand_seed', 0)

        options = ['linear', 'nn', 'pls', 'svm']

        fit_method = match_keywords(method, options)

        model_dict = {
            'linear': LinearRegression(
                fit_intercept=True,
                n_jobs=8,
            ),

            'nn': MLPRegressor(
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                hidden_layer_sizes=(100,),
                max_iter=max_iter,
                shuffle=True,
                random_state=rand_seed,
                tol=tol,
                verbose=verbose,
                nesterovs_momentum=True,
                early_stopping=True,
                n_iter_no_change=int(max_iter * 0.1),
                validation_fraction=0.15,
            ),

            'pls': PLSRegression(
                n_components=np.shape(X)[1]-1,
                max_iter=max_iter,
                tol=tol,
            ),

            'svm': SVR(
                kernel='rbf',
                degree=3,
                verbose=verbose,
                epsilon=0.1,
                tol=tol,
                max_iter=max_iter,
            )
        }
        if fit_method not in options:
            response = ThrowDlg.yesno(f"""Fit method '{fit_method}' is not recognized. Try one of: {options}.
                                      Do you want to continue with the default method (Linear regression)?""")
            if response.lower() == "yes": 
                fit_method = 'linear'
                log_(f"Using the default method '{fit_method}' for fitting", color='orange', font_style = 'bold', level='WARNING')
                return model_dict[fit_method]
            else:
                log_(f"Cancelled fitting using method '{fit_method}'", color='orange', font_style = 'bold', level='WARNING')
                sys.exit(0)
        else:
            log_(f"ffc fitting using method '{fit_method}'...", color='cyan', font_style='italic', level='INFO')
            model = model_dict[fit_method]

        model.fit(X, y)
        return model

    def compute_multiplier(self, **kwargs):

        # 1. Crop the image
        self.detect_and_crop()

        img_full = self.img.copy()
        img_cropped = self.img_cropped.copy()

        L_full, _ = self.get_L(img_full, smooth=True)
        L_cropped, _ = self.get_L(img_cropped, smooth=True)

        L_float = L_cropped.astype(FLOAT)/255
        self.cropped_multiplier = np.max(L_float)/L_float

        flat_cropped = (L_float * self.cropped_multiplier)
        flat_cropped = (255*flat_cropped).astype(UINT8)

        # 2. Compute metrics
        # (omitted for brevity, same as before)

        # 3. Extrapolate multiplier
        if self.crop_rect is None:
            y1, x1, y2, x2 = 0, 0, self.img.shape[1], self.img.shape[0]
        else:
            y1, x1, y2, x2 = self.crop_rect

        x = np.linspace(x1, x2-1, self.bins)
        y = np.linspace(y1, y2-1, self.bins)
        X, Y = np.meshgrid(x, y)

        x_c = np.linspace(0, L_cropped.shape[0]-1, self.bins)
        y_c = np.linspace(0, L_cropped.shape[1]-1, self.bins)
        X_c, Y_c = np.meshgrid(x_c, y_c)

        h_win = int((self.smooth_window-1) / 2)

        Z_m = np.ones_like(X_c)
        for i, x_ in enumerate(x_c):
            for j, y_ in enumerate(y_c):
                x_,y_ = int(x_), int(y_)
                x_l, x_h = [max(0, x_-h_win), min(L_cropped.shape[0]-1, x_+h_win)]
                y_l, y_h = [max(0, y_-h_win), min(L_cropped.shape[1]-1, y_+h_win)]
                Z_m[i, j] = np.mean(self.cropped_multiplier[x_l:x_h, y_l:y_h])

        Z_m = np.array(Z_m)

        x_flat, y_flat = X.flatten(), Y.flatten()
        z_flat = Z_m.flatten()

        min_x, max_x = 0, L_full.shape[0]
        min_y, max_y = 0, L_full.shape[1]
        min_z, max_z = np.min(z_flat), np.max(z_flat)

        eps = 1e-15
        x_flat = (x_flat - min_x) / (max_x - min_x + eps)
        y_flat = (y_flat - min_y) / (max_y - min_y + eps)
        z_flat = (z_flat - min_z) / (max_z - min_z + eps)

        xy_flat = np.stack([x_flat, y_flat], axis=1)
        xy_flat, names, poly = self.polynomial_features(xy_flat, **kwargs)

        model = self.fit_model(xy_flat, z_flat, **kwargs)

        x_full = np.linspace(0, L_full.shape[0]-1, self.bins)
        y_full = np.linspace(0, L_full.shape[1]-1, self.bins)
        X_full, Y_full = np.meshgrid(x_full, y_full)

        X_full_flat = X_full.flatten()
        Y_full_flat = Y_full.flatten()

        x_full_flat = (X_full_flat - min_x) / (max_x - min_x + eps)
        y_full_flat = (Y_full_flat - min_y) / (max_y - min_y + eps)

        xy_full_flat = np.stack([x_full_flat, y_full_flat], axis=1)
        xy_full_flat = poly.transform(xy_full_flat)

        # predict
        f_multiplier = model.predict(xy_full_flat)
        f_multiplier = (f_multiplier*(max_z-min_z) + min_z).reshape(self.bins, self.bins)

        f_multiplier = self.transform_extremity(
            f_multiplier, 
            max=1.8, 
            cut_off=1.3,
        )
        
        f_multiplier = cv2.resize(f_multiplier, (L_full.shape[1], L_full.shape[0]), interpolation=cv2.INTER_CUBIC)
        self.final_multiplier = f_multiplier

        # 4. Apply FFC to the image (via per-pixel multiplication)
        img_corrected = self.apply_ffc(img_full)

        if self.show:
            self.show_3d([flat_cropped,L_cropped], names=['Flat','Original'])
            self.show_3d([self.final_multiplier], names=['Final Multiplier'])
            self.show_results(img_corrected, img_full)

        gc.collect()
        return self.final_multiplier
    

    def apply_ffc(self, img, multiplier=None, show=False):
        """
        Applies flat‐field correction to `img` by multiplying its L channel by self.final_multiplier.
        """
        img_orig = img if not show else img.copy()
        assert img_orig.dtype == UINT8, 'Image must be of type UINT8'

        if multiplier is not None:
            self.final_multiplier = multiplier
            
        w, h = img.shape[:2]
        w_o, h_o = self.final_multiplier.shape[:2]

        if (w, h) != (w_o, h_o):
            log_(
                f'Image size: {w}x{h} | Final multiplier size: {w_o}x{h_o}', 
                'light_yellow', 'italic'
            )
            rtn = ThrowDlg.yesno(
                msg='Image size does not match final multiplier size. Do you want to resize the image?',
            )
            if rtn == 'yes':
                img = self.resize_image(img, size=(w_o, h_o))

        L, img_LAB = self.get_L(img, smooth=False)

        # multiply L channel by final_multiplier
        L_ = (L.astype(FLOAT) * self.final_multiplier).astype(UINT8)

        if self.check_color(img):
            img_LAB[:, :, 0] = L_
            img_corrected = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
        else:
            img_corrected = L_

        if show:
            self.show_results(img_corrected, img_orig)

        return np.clip(img_corrected, 0, 255)

