import os
import numpy as np
import pandas as pd
import torch
from itertools import chain
from typing import Tuple
from tqdm.notebook import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score

def lorenz(t, x, sigma=10, beta=2.66667, rho=28):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]

def plot_lorenz(x_train, x_dot_train_measured, x_test, x_dot_test_measured, t_train, t_test, feature_name=["x", "y", "z"]):
    plt.figure(figsize=(20, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(t_train, x_train[:, i], label='train')
        plt.plot(t_test, x_test[:, i], label='test')
        plt.grid(True)
        plt.xlabel("t", fontsize=24)
        plt.ylabel(feature_name[i], fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)

    plt.tight_layout()

    plt.figure(figsize=(20, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(t_train, x_dot_train_measured[:, i], label='train')
        plt.plot(t_test, x_dot_test_measured[:, i], label='test')
        plt.grid(True)
        plt.xlabel("t", fontsize=24)
        plt.ylabel(r"$\dot{" + feature_name[i] + "}$", fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)

    plt.tight_layout()

def arx_process(y_sequence, u_sequence, a_params, b_params, t):
    if a_params.size == 1:
        a_params = a_params[:, np.newaxis]
    if b_params.size == 1:
        b_params = b_params[:, np.newaxis]
    
    n_a, n_b = a_params.shape[0], b_params.shape[0]
    phi_y = np.array([y_sequence[i] if i>=0 else 0.0 for i in range(t-1, t-n_a-1, -1)])[:, np.newaxis]
    phy_u = np.array([u_sequence[i] if i>=0 else 0.0 for i in range(t-1, t-n_b-1, -1)])[:, np.newaxis]
    
    return phi_y.T@a_params + phy_u.T@b_params

def simulate_arx(u_sequence, a_params, b_params, sigma2_y=0.0, seed=0):
    rng = np.random.default_rng(seed=seed)
    n = u_sequence.shape[0]
    y0 = np.zeros(n)
    y = np.zeros(n)
    e_y = np.zeros(n)
    for t in range(n):
        y0[t] = arx_process(y_sequence=y0, u_sequence=u_sequence, a_params=a_params, b_params=b_params, t=t)
        e_y[t] = np.sqrt(sigma2_y)*rng.standard_normal(size=1)
        y[t] = y0[t] + e_y[t]
    return y, y0, e_y

def plot_realization(u0, y0, u, y, t=None, figsize=(10,6)):
    if t is None:
         t = np.arange(0, u.shape[0], 1)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, hspace=0.1)
    axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle('Data')
    axs[0].plot(t, u, label=r'$u(t)$', color='b')
    axs[0].plot(t, u0, label=r'$u_0(t)$', color='r')
    axs[0].set_ylabel('u(t)')
    axs[0].margins(0.0)
    axs[0].legend(loc='upper left')

    axs[1].plot(t, y, label=r'$y(t)$', color='b')
    axs[1].plot(t, y0, label=r'$y_0(t)$', color='r')
    axs[1].set_ylabel('y(t)')
    axs[1].set_xlabel('t')
    axs[1].margins(0.0)
    axs[1].legend(loc='upper left')
    
    plt.show()

def plot_random_realization(u0, y0, u, y, t=None, seed=0, figsize=(10,6)):
    rng = np.random.default_rng(seed=seed)
    idx = rng.integers(low=0, high=u.shape[1])
    if t is None:
         t = np.arange(0, u.shape[0], 1)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, hspace=0.1)
    axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle('Data')
    axs[0].plot(t*1000, u[:,idx], label=r'$u(t)$', color='b')
    axs[0].plot(t*1000, u0[:,idx], label=r'$u_0(t)$', color='r')
    axs[0].set_ylabel('u(t)')
    axs[0].margins(0.0)
    axs[0].legend(loc='upper left')

    axs[1].plot(t*1000, y[:,idx], label=r'$y(t)$', color='b')
    axs[1].plot(t*1000, y0[:,idx], label=r'$y_0(t)$', color='r')
    axs[1].set_ylabel('y(t)')
    axs[1].set_xlabel('t [ms]')
    axs[1].margins(0.0)
    axs[1].legend(loc='upper left')
    
    plt.show()

def create_arx_regressors(u, y, na, nb):
    n = y.shape[0]
    phi = pd.DataFrame(data=zip(u, y), index=range(n), columns=['u(t)', 'y(t)'])
    regressors = []
    for tau_a in range(1, na+1):
        phi[f'y(t-{tau_a})'] = phi['y(t)'].shift(tau_a).fillna(0.0)
        regressors.append(f'y(t-{tau_a})')
    for tau_b in range(1, nb+1):
        phi[f'u(t-{tau_b})'] = phi['u(t)'].shift(tau_b).fillna(0.0)
        regressors.append(f'u(t-{tau_b})')
    return phi[regressors].values

def arx_identification(model, u, y, na, nb):
    Phi = create_arx_regressors(u=u, y=y, na=na, nb=nb)
    model.fit(Phi, y)
    if isinstance(model, GridSearchCV):
        return model.best_estimator_.coef_
    else:
        return model.coef_

def plot_coefficients(theta_df, a0, b0, xlim=(-1,1)):
    na = a0.shape[0]
    nb = b0.shape[0]
    fig, axes = plt.subplots(int(np.ceil((na+nb)/2)), 2, figsize=(12, int(np.ceil((na+nb)/2))*3))
    axes = axes.flat
    for i,(p, t) in enumerate(zip([f'a{i+1}' for i in range(na)] + [f'b{i+1}' for i in range(nb)], a0.tolist() + b0.tolist())):
        axes[i].hist(theta_df[p].values, bins=25)
        axes[i].axvline(theta_df[p].median(), linestyle='-', color='k', label=fr"$\hat{{{p}}}$")
        axes[i].axvline(t, linestyle='--', color='r', label=fr"${{{p}}}$")
        axes[i].set_xlim(xlim)
        axes[i].legend()
    plt.tight_layout()
    plt.show()

def arx_identification_iv(model, u, y, na, nb):
    n = y.shape[0]
    y_hat = np.zeros(n)

    theta_hat = arx_identification(model=LinearRegression(fit_intercept=False), u=u, y=y, na=na, nb=nb)
    
    for t in range(n):
        y_hat[t] = arx_process(y_sequence=y, u_sequence=u, a_params=theta_hat[:na], b_params=theta_hat[na:], t=t)
        
    Phi = create_arx_regressors(u=u, y=y, na=na, nb=nb)
    Z = create_arx_regressors(u=u, y=y_hat, na=na, nb=nb)
    if isinstance(model, MultiRidge):
        model.fit(Phi, y, Z)
    else:
        model.fit(Z.T @ Phi, Z.T @ y)
    if isinstance(model, GridSearchCV) or isinstance(model, RandomizedSearchCV):
        return model.best_estimator_.coef_
    else:
        return model.coef_

class MultiRidge(RegressorMixin, BaseEstimator):
    def __init__(self,
                 alpha=1.0,
                 folds=5,
                 shuffle=True,
                 random_state=None,
                 epochs=100,
                 learning_rate=1.0,
                 history=False,
                 ):
        self.alpha = alpha
        self.folds = folds
        self.shuffle = shuffle
        self.random_state = random_state
        if self.shuffle is True:
            self.kf = KFold(n_splits=self.folds, shuffle=self.shuffle, random_state=self.random_state)
        else:
            self.kf = KFold(n_splits=self.folds, shuffle=self.shuffle)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.Lambda = None
        self.coef_ = None
        self.multioutput = None
        if history:
            self.history = dict()
        else:
            self.history = None
    def fit(self, X, y, Z=None):
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')
        if len(y.shape) == 1:
            X,Y = check_X_y(X, y)
            Y = y[:, np.newaxis]
            self.multioutput = False
        else:
            X,Y = check_X_y(X, y, multi_output=True)
            self.multioutput = True
        n, d = X.shape
        m = Y.shape[1]
        self.Lambda = np.eye(d)*self.alpha
        if isinstance(self.history, dict):
            self.history['lambda'] = np.zeros((self.epochs+1, d))
            self.history['lambda'][0,:] = self.alpha
            self.history['coef'] = np.zeros((self.epochs, d, m)) if m > 1 else np.zeros((self.epochs, d))
        if Z is None:
            for k in range(self.epochs):
            #for k in tqdm(range(self.epochs), total=self.epochs):
                grad_J_v_cv = np.zeros(d)
                for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(self.kf.split(X, Y)):
                    X_t, Y_t = X[train_fold_idx], Y[train_fold_idx]
                    X_v, Y_v = X[valid_fold_idx], Y[valid_fold_idx]
                    n_t, n_v = X_t.shape[0], X_v.shape[0]

                    theta_hat = np.linalg.lstsq(X_t.T @ X_t + n_t * self.Lambda @ self.Lambda, X_t.T @ Y_t, rcond=None)[0]
                    C = X_v @ theta_hat - Y_v
                    B = np.linalg.lstsq((X_t.T @ X_t + n_t * self.Lambda @ self.Lambda).T, X_v.T @ C @ theta_hat.T, rcond=None)[0]
                    grad_J_v = -(n_t/n_v) * np.diagonal(self.Lambda @ B + B @ self.Lambda)
                    grad_J_v_cv += grad_J_v / self.folds

                theta_hat_refit = np.linalg.lstsq(X.T @ X + n * self.Lambda.T @ self.Lambda, X.T @ Y, rcond=None)[0]
                if self.multioutput is True:
                    self.coef_ = theta_hat_refit
                else:
                    self.coef_ = theta_hat_refit.squeeze()

                if isinstance(self.learning_rate, float):
                    self.Lambda = self.Lambda - self.learning_rate*np.diag(grad_J_v_cv)
                else:
                    self.Lambda = self.Lambda - self.learning_rate.learning_rate*np.diag(grad_J_v_cv)
                    self.learning_rate.update_lr(k)
                    
                if isinstance(self.history, dict):
                    self.history['lambda'][k+1,:] = np.diagonal(self.Lambda)
                    self.history['coef'][k,:] = self.coef_
        else:
            for k in range(self.epochs):
            #for k in tqdm(range(self.epochs), total=self.epochs):
                grad_J_v_cv = np.zeros(d)
                for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(self.kf.split(X, Y)):
                    X_t, Y_t, Z_t = X[train_fold_idx], Y[train_fold_idx], Z[train_fold_idx]
                    X_v, Y_v, Z_v = X[valid_fold_idx], Y[valid_fold_idx], Z[valid_fold_idx]
                    n_t, n_v = X_t.shape[0], X_v.shape[0]

                    theta_hat = np.linalg.lstsq(X_t.T @ Z_t @ Z_t.T @ X_t + n_t**2 * self.Lambda @ self.Lambda, X_t.T @ Z_t @ Z_t.T @ Y_t, rcond=None)[0]
                    C = Z_v.T @ (X_v @ theta_hat - Y_v)
                    B = np.linalg.lstsq((X_t.T @ Z_t @ Z_t.T @ X_t + n_t**2 * self.Lambda @ self.Lambda).T, X_v.T @ Z_v @ C @ theta_hat.T, rcond=None)[0]
                    grad_J_v = -(n_t/n_v)**2 * np.diagonal(self.Lambda @ B + B @ self.Lambda)
                    grad_J_v_cv += grad_J_v / self.folds

                theta_hat_refit = np.linalg.lstsq(X.T @ Z @ Z.T @ X + n**2 * self.Lambda.T @ self.Lambda, X.T @ Z @ Z.T @ Y, rcond=None)[0]
                if self.multioutput is True:
                    self.coef_ = theta_hat_refit
                else:
                    self.coef_ = theta_hat_refit.squeeze()

                if isinstance(self.learning_rate, float):
                    self.Lambda = self.Lambda - self.learning_rate*np.diag(grad_J_v_cv)
                else:
                    self.Lambda = self.Lambda - self.learning_rate.learning_rate*np.diag(grad_J_v_cv)
                    self.learning_rate.update_lr(k)
                
                if isinstance(self.history, dict):
                    self.history['lambda'][k+1,:] = np.diagonal(self.Lambda)
                    self.history['coef'][k,:] = self.coef_
            
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        return X @ self.coef_

    def _more_tags(self):
        return {
            'poor_score': True
        }

def simulation_error(u_test, y_test, a_estimated, b_estimated):
    n_test = y_test.shape[0]
    y_test_sim = np.zeros(n_test)
    y_test_sim, _, _ = simulate_arx(u_sequence=u_test, a_params=a_estimated, b_params=b_estimated, sigma2_y=0.0, seed=0)
    if np.isfinite(y_test_sim).all():
        return (1/n_test)*np.linalg.norm(y_test - y_test_sim)**2, np.maximum(0,r2_score(y_test, y_test_sim))
    else:
        return 99,0

class LinearLR():
    def __init__(self, initial_lr, decay):
        self.learning_rate = initial_lr
        self.decay = decay
        
    def update_lr(self, epoch):
        self.learning_rate *= self.decay

class MultiRidge2(RegressorMixin, BaseEstimator):
    def __init__(self,
                 lambda_vector=None,
                 folds=5,
                 shuffle=True,
                 random_state=None,
                 normalize=True,
                 epochs=100,
                 learning_rate=100,
                 scoring='r2',
                 verbose=0,
                 save_history=True,
                 device='cpu',
                 dtype=torch.float32):
        
        self.lambda_vector = lambda_vector
        self.folds = folds
        self.shuffle = shuffle
        if self.shuffle is True:
            if not isinstance(random_state, int):
                self.random_state = 0
            else:
                self.random_state = random_state
        self.normalize = normalize
        self.epochs = epochs
        self.learning_rate = learning_rate
        if isinstance(scoring, str):
            from sklearn.metrics import get_scorer
            self.scoring = {scoring: get_scorer(scoring)._score_func}
        else:
            self.scoring = scoring
        self.verbose = verbose
        self.save_history = save_history
        if self.save_history is True:
            columns = [f'fold{fold+1}_{split}_{metric}' for fold in range(self.folds) for split in ['train', 'valid'] for metric in self.scoring.keys()] + \
                      [f'{stat}_{split}_{metric}' for stat in ['mean', 'std'] for split in ['train', 'valid'] for metric in self.scoring.keys()] + \
                      list(chain.from_iterable((f'test_{metric}_ensemble', f'test_{metric}_refit', f'train_{metric}_refit') for metric in self.scoring.keys()))
            self.history = dict()
            self.history['learning_curves'] = pd.DataFrame(data=0.0, columns=columns, index=range(self.epochs))
        self.device = device
        self.dtype = dtype
        self.coef_ = None
            

    def fit(self, X, y, eval_set=None,  **kwargs):
        
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        #X_train, Y_train = check_X_y(X, y)
        if len(y.shape) == 1:
            X_train, Y_train = check_X_y(X, y)
            Y_train = Y_train[:, np.newaxis]
        else:
            X_train, Y_train = X,y
        X_train_copy, Y_train_copy = np.copy(X_train), np.copy(Y_train)
        self.n_samples_in_  = X_train.shape[0]
        self.n_features_in_ = X_train.shape[1]
        self.n_targets = Y_train.shape[1]

        if eval_set is not None:
            #X_test, Y_test = check_X_y(eval_set[0], eval_set[1])
            if len(eval_set[1].shape) == 1:
                X_test, Y_test = check_X_y(eval_set[0], eval_set[1])
                Y_test = Y_test[:, np.newaxis]
            else:
                X_test, Y_test = eval_set[0], eval_set[1]
            X_test_copy, Y_test_copy = np.copy(X_test), np.copy(Y_test)

        if self.normalize is True:
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_train, Y_train = scaler_x.fit_transform(X_train), scaler_y.fit_transform(Y_train)
            if eval_set is not None:
                X_test, Y_test = scaler_x.transform(X_test), scaler_y.transform(Y_test)
            σx, σy = np.sqrt(scaler_x.var_), np.sqrt(scaler_y.var_)
            if len(y.shape) == 1:
                self.scale_factor = σy / σx
            else:
                self.scale_factor = (σy[:,np.newaxis] / σx).T
        else:
            self.scale_factor = 1.0

        X_train, Y_train = torch.tensor(X_train, device=self.device, dtype=self.dtype), torch.tensor(Y_train, device=self.device, dtype=self.dtype)
        if eval_set is not None:
            X_test, Y_test = torch.tensor(X_test, device=self.device, dtype=self.dtype), torch.tensor(Y_test, device=self.device, dtype=self.dtype)
        
        Id = torch.eye(self.n_features_in_, device=self.device, dtype=self.dtype)
        ones = torch.ones(self.n_features_in_, 1, device=self.device, dtype=self.dtype)
        if self.lambda_vector is None:
            self.lambda_vector = torch.ones(self.n_features_in_, device=self.device, dtype=self.dtype)
        else:
            self.lambda_vector = torch.tensor(self.lambda_vector, device=self.device, dtype=self.dtype)
        
        if self.save_history is True:    
            self.history['lambda'] = np.zeros((self.epochs+1, self.n_features_in_))
            self.history['lambda'][0,:] = self.lambda_vector.cpu().numpy()
            self.history['coef'] = np.zeros((self.epochs, self.n_features_in_, self.n_targets)) if self.n_targets > 1 else np.zeros((self.epochs, self.n_features_in_))

        for k in tqdm(range(self.epochs), total=self.epochs):
            grad_E_cv = 0.0
            if eval_set is not None:
                Y_test_hat_ensemble = torch.zeros_like(Y_test, device=self.device, dtype=self.dtype)
            
            if self.shuffle is True:
                kf = KFold(n_splits=self.folds, shuffle=self.shuffle, random_state=self.random_state*(k+1))
            else:
                kf = KFold(n_splits=self.folds, shuffle=self.shuffle)
    
            for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(kf.split(X_train_copy, Y_train_copy)):

                X_train_fold, Y_train_fold = X_train_copy[train_fold_idx], Y_train_copy[train_fold_idx]
                X_valid_fold, Y_valid_fold = X_train_copy[valid_fold_idx], Y_train_copy[valid_fold_idx]

                if self.normalize is True:
                    scaler_fold_x, scaler_fold_y = StandardScaler(), StandardScaler()
                    X_train_fold, Y_train_fold  = scaler_fold_x.fit_transform(X_train_fold), scaler_fold_y.fit_transform(Y_train_fold)
                    X_valid_fold, Y_valid_fold  = scaler_fold_x.transform(X_valid_fold), scaler_fold_y.transform(Y_valid_fold)
                    σx_fold, σy_fold = np.sqrt(scaler_fold_x.var_), np.sqrt(scaler_fold_y.var_)
                    if len(y.shape) == 1:
                        self.scale_factor_fold = σy_fold / σx_fold
                    else:
                        self.scale_factor_fold = (σy_fold[:,np.newaxis] / σx_fold).T
                else:
                    self.scale_factor_fold = 1.0
                
                X_train_fold, Y_train_fold = torch.tensor(X_train_fold, device=self.device, dtype=self.dtype), torch.tensor(Y_train_fold, device=self.device, dtype=self.dtype)
                X_valid_fold, Y_valid_fold = torch.tensor(X_valid_fold, device=self.device, dtype=self.dtype), torch.tensor(Y_valid_fold, device=self.device, dtype=self.dtype)
                N_train_fold, N_valid_fold = X_train_fold.shape[0], X_valid_fold.shape[0]
                
                Λ = torch.diag(self.lambda_vector)
                theta_fold_hat = torch.linalg.lstsq(X_train_fold.T @ X_train_fold + N_train_fold * Λ@Λ, X_train_fold.T @ Y_train_fold)[0]

                Y_train_fold_hat = X_train_fold @ theta_fold_hat
                Y_valid_fold_hat = X_valid_fold @ theta_fold_hat
                if eval_set is not None:
                    Y_test_hat_ensemble += X_test @ theta_fold_hat
                
                ## gradient computation
                R_fold = (Y_valid_fold_hat - Y_valid_fold)
                B_fold = torch.linalg.lstsq((X_train_fold.T @ X_train_fold + N_train_fold * Λ@Λ).T, X_valid_fold.T @ R_fold @ theta_fold_hat.T)[0]
                grad_E_fold = -(N_train_fold/N_valid_fold) * torch.diagonal(Λ@B_fold + B_fold@Λ)
                grad_E_cv += grad_E_fold/self.folds

                ## save fold metrics
                if self.save_history is True:
                    for metric, metric_func in self.scoring.items():
                        self.history['learning_curves'].loc[k, f'fold{n_fold+1}_train_{metric}'] = metric_func(Y_train_fold.cpu().numpy(),#*self.scale_factor_fold,
                                                                                               Y_train_fold_hat.cpu().numpy())#*self.scale_factor_fold)
                        self.history['learning_curves'].loc[k, f'fold{n_fold+1}_valid_{metric}'] = metric_func(Y_valid_fold.cpu().numpy(),#*self.scale_factor_fold,
                                                                                               Y_valid_fold_hat.cpu().numpy())#*self.scale_factor_fold)

            ## save aggregated statistics
            if self.save_history is True:
                for metric, metric_func in self.scoring.items():
                    self.history['learning_curves'].loc[k, f'mean_train_{metric}'] = self.history['learning_curves'].loc[k, [f'fold{fold+1}_train_{metric}' for fold in range(self.folds)]].mean()
                    self.history['learning_curves'].loc[k, f'std_train_{metric}']  = self.history['learning_curves'].loc[k, [f'fold{fold+1}_train_{metric}' for fold in range(self.folds)]].std(ddof=0)
                    self.history['learning_curves'].loc[k, f'mean_valid_{metric}'] = self.history['learning_curves'].loc[k, [f'fold{fold+1}_valid_{metric}' for fold in range(self.folds)]].mean()
                    self.history['learning_curves'].loc[k, f'std_valid_{metric}']  = self.history['learning_curves'].loc[k, [f'fold{fold+1}_valid_{metric}' for fold in range(self.folds)]].std(ddof=0)
                    if eval_set is not None:
                        self.history['learning_curves'].loc[k, f'test_{metric}_ensemble'] = metric_func(Y_test.cpu().numpy(),#*self.scale_factor,
                                                                                        Y_test_hat_ensemble.cpu().numpy())#*self.scale_factor)

            ## refit an all train and compute test score
            theta_hat_refit = torch.linalg.lstsq(X_train.T @ X_train + self.n_samples_in_ * Λ@Λ, X_train.T @ Y_train)[0]
            Y_train_hat_refit = X_train @ theta_hat_refit
            if eval_set is not None:
                Y_test_hat_refit = X_test @ theta_hat_refit

            if self.save_history is True:
                for metric, metric_func in self.scoring.items():
                    self.history['learning_curves'].loc[k, f'train_{metric}_refit'] = metric_func(Y_train.cpu().numpy(),#*self.scale_factor,
                                                                                  Y_train_hat_refit.cpu().numpy())#*self.scale_factor)
                    if eval_set is not None:
                        self.history['learning_curves'].loc[k, f'test_{metric}_refit'] = metric_func(Y_test.cpu().numpy(),#*self.scale_factor,
                                                                                     Y_test_hat_refit.cpu().numpy())#*self.scale_factor)
            
            self.coef_ = (self.scale_factor * theta_hat_refit.cpu().numpy().squeeze()).T

            ## logging
            if (self.verbose > 0) and (self.save_history is True):
                if k%self.verbose == 0:
                    text = "Epoch {}: ".format(k + 1)
                    for metric, _ in self.scoring.items():
                        text += "Train {}: {:.3f}, ".format(metric, self.history['learning_curves'].loc[k, f'mean_train_{metric}'])
                        text += "Valid {}: {:.3f}, ".format(metric, self.history['learning_curves'].loc[k, f'mean_valid_{metric}'])
                        if eval_set is not None:
                            text += "Test {}: {:.3f}, ".format(metric, self.history['learning_curves'].loc[k, f'test_{metric}_refit'])
                    print(text[:-2])

            if isinstance(self.learning_rate, float):
                self.lambda_vector = self.lambda_vector - self.learning_rate * grad_E_cv
            else:
                self.lambda_vector = self.lambda_vector - self.learning_rate.learning_rate * grad_E_cv
                self.learning_rate.update_lr(k)
            
            if self.save_history is True:
                self.history['lambda'][k+1,:] = self.lambda_vector.cpu().numpy()
                self.history['coef'][k,:] = self.coef_.T
        
        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        return X @ self.coef_.T

    def _more_tags(self):
        return {
            'poor_score': True
        }
    
    
### Figure saving function taken from:
### https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/#saving-figures
def save_fig(
        fig: matplotlib.figure.Figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: Tuple[float, float] = [6.4, 4], 
        save: bool = True, 
        dpi: int = 300,
        transparent_png = True,
    ):
    """This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print("Cannot save figure: {}".format(e))