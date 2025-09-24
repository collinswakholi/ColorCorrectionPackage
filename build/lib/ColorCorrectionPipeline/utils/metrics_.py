from sklearn import metrics
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np
from scipy import stats
import pandas as pd

__all__ = ['Metrics', 'desc_stats']

class metrics_results:
    def __init__(self):
        self.metrics = None
        self.decriptor = None

class Metrics:
    def __init__(self, gt, pred):
        self.gt = np.array(gt)
        self.pred = np.array(pred)
        self.results = metrics_results()

        # check if gt and pred are the same size
        if gt.shape != pred.shape:
            print('gt and pred should be the same size')
            return
    
        

    def get_(self, type_ = 'regression', return_df = True):
        mets = []
        if type_ == 'regression':
            mets = self.regression_metrics()
        elif type_ == 'classification':
            mets = self.classification_metrics()
        elif type_ == 'all':
            class_ = self.classification_metrics()
            reg_ = self.regression_metrics()

            mets = {**class_, **reg_}
        else:
            raise ValueError('type_ should be regression, classification, or all')
        
        descriptor_pred = desc_stats(self.pred)
        
        if return_df:
            descriptor_pred = pd.DataFrame(descriptor_pred, index=[0])
            descriptor_pred.index = ['Descriptor']
            mets = pd.DataFrame(mets, index=[0])
            mets.index = ['Metrics']
        self.results.metrics = mets
        self.results.decriptor = descriptor_pred
        return self.results


    def compute_mae(self, mat1, mat2):

        mat1 = np.asarray(mat1)
        mat2 = np.asarray(mat2)

        assert mat1.shape == mat2.shape, "Matrices must have same shape"

        mat1_flat = mat1.reshape(-1, mat1.shape[-1])
        mat2_flat = mat2.reshape(-1, mat2.shape[-1])

        norms = np.linalg.norm(mat1_flat, axis=1) * np.linalg.norm(mat2_flat, axis=1)
        dots_prods = np.einsum('ij,ij->i', mat1_flat, mat2_flat)

        dot_product = np.clip(dots_prods / norms, -1, 1)
        angular_error = np.degrees(np.arccos(dot_product))
        return angular_error

    def regression_metrics(self):

        metrics_ = {}

        gt_ = self.gt.flatten()
        pred_ = self.pred.flatten()
        m_out = 'uniform_average'
        rmse = metrics.root_mean_squared_error(self.gt, self.pred, multioutput=m_out)
        mae = metrics.mean_absolute_error(self.gt, self.pred, multioutput=m_out)
        mape = metrics.mean_absolute_percentage_error(self.gt, self.pred, multioutput=m_out)
        r2_score = metrics.r2_score(self.gt, self.pred, multioutput=m_out)
        p_dist = metrics.pairwise_distances(self.gt, self.pred, metric='euclidean', n_jobs=16)
        max_error = metrics.max_error(gt_, pred_)
        # gamma_deviance = metrics.mean_gamma_deviance(gt_, pred_)
        try:
            gamma_deviance = metrics.mean_gamma_deviance(gt_, pred_)
        except:
            gamma_deviance = np.nan
        mean_angular_error = np.mean(self.compute_mae(self.gt, self.pred))

        try:
            ssim = structural_similarity(self.gt, self.pred)
        except:
            ssim = np.nan
        try:
            psnr = peak_signal_noise_ratio(self.gt, self.pred)
        except:
            psnr = np.nan

        metrics_['MSE'] = rmse
        metrics_['MAE'] = mae
        metrics_['R2_score'] = r2_score
        metrics_['MAPE'] = mape
        metrics_['Max_error'] = max_error
        metrics_['Gamma_deviance'] = gamma_deviance
        metrics_['pairwise_dist_mean'] = np.mean(p_dist)
        metrics_['pairwise_dist_std'] = np.std(p_dist)
        metrics_['SSIM'] = ssim
        metrics_['PSNR'] = psnr
        metrics_['Mean_angular_error'] = mean_angular_error


        return metrics_
    

    def classification_metrics(self):
        metrics_ = {}

        gt_ = self.gt.flatten()
        pred_ = self.pred.flatten()

        avg = 'micro' # or 'macro', 'weighted', 'samples', 'binary', 'micro'
        zero_div = 0.0

        accuracy = metrics.accuracy_score(gt_, pred_)
        balanced_accuracy = metrics.balanced_accuracy_score(gt_, pred_)
        f1 = metrics.f1_score(gt_, pred_, average=avg, zero_division=zero_div)
        precision = metrics.precision_score(gt_, pred_, average=avg, zero_division=zero_div)
        recall = metrics.recall_score(gt_, pred_, average=avg, zero_division=zero_div)
        jaccard = metrics.jaccard_score(gt_, pred_, average=avg, zero_division=zero_div)
        hamming_loss = metrics.hamming_loss(gt_, pred_)

        metrics_['Accuracy'] = accuracy
        metrics_['Balanced_accuracy'] = balanced_accuracy
        metrics_['F1'] = f1
        metrics_['Precision'] = precision
        metrics_['Recall'] = recall
        metrics_['Jaccard'] = jaccard
        metrics_['Hamming_loss'] = hamming_loss

        return metrics_


def desc_stats(mat: np.ndarray) -> dict:
    """
    Description:
    ------------
    Compute the min, max, mean, variance, skew, kurtosis, median, std, snr, snr_20_log, coef_variation
    """
    descriptor = {}
    
    desc = stats.describe(mat, axis=None)
    min_, max_ = desc.minmax
    mean_ = desc.mean
    variance_ = desc.variance
    skew_ = desc.skewness
    kurtosis_ = desc.kurtosis

    median_ = np.median(mat)
    std_ = np.std(mat)

    snr_20_log = 20 * np.log10(np.mean(mat) / np.std(mat))
    coef_variation = np.std(mat) / np.mean(mat)

    
    descriptor['min'] = min_
    descriptor['max'] = max_
    descriptor['mean'] = mean_
    descriptor['variance'] = variance_
    descriptor['skew'] = skew_
    descriptor['kurtosis'] = kurtosis_
    descriptor['median'] = median_
    descriptor['std'] = std_
    descriptor['snr_20_log'] = snr_20_log
    descriptor['coef_variation'] = coef_variation

    return descriptor