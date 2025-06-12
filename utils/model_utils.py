import tqdm
import numpy as np
import torch
from torch.distributions.normal import Normal
from scipy.stats import norm as scipy_norm

#########################################################################################
# DETECTION MODEL
#########################################################################################

def detection_loss(beta, y, X, threshold, beta_mean, beta_prec):
    L = X@beta
    Phi_L = Normal(0, 1).cdf(L)*threshold
    Phi_L = torch.clamp(Phi_L, 1e-6, 1 - 1e-6)
    llh = torch.mean(y*torch.log(Phi_L) + (1 - y)*torch.log1p(-Phi_L))
    penalty = -0.5*beta_prec*torch.sum((beta - beta_mean)**2)
    return -(llh + penalty)


def fit_detection_model(y_train, X_train, threshold_train, beta_mean, beta_prec,
                       lr=0.1, num_epochs=10000, tol=0.0001):
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    threshold_train_tensor = torch.tensor(threshold_train, dtype=torch.float32)
    beta_mean_tensor = torch.tensor(beta_mean, dtype=torch.float32)
    
    beta = torch.tensor([-3.0, -3.0, 1.0, 0.0, 0.0], requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    losses = []
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training detection model", mininterval=10):
        optimizer.zero_grad()
        loss = detection_loss(beta, y_train_tensor, X_train_tensor, threshold_train_tensor, beta_mean_tensor, beta_prec)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            losses.append(loss.item())

        if len(losses)>2 and np.abs(losses[-1] - losses[-2])/losses[-2] < tol:
            break
    
    return beta.detach().numpy()


#########################################################################################
# MIGRATION MODEL
#########################################################################################

def m_numpy(lats, days, theta):
    Z1 = (days - (theta[0] + theta[1]*lats))/(theta[4]/2)
    Z2 = (days - (theta[2] + theta[3]*lats))/(theta[5]/2)
    P1 = scipy_norm.cdf(Z1)
    P2 = 1 - scipy_norm.cdf(Z2)
    return np.c_[P1, P2].min(axis=1)


def m_torch(lat, day, theta):
    theta1, theta2, theta3, theta4, theta5, theta6 = theta
    term1 = torch.distributions.Normal(0.0, 1.0).cdf((day - (theta1 + theta2 * lat)) / (theta5/2.0))
    term2 = 1 - torch.distributions.Normal(0.0, 1.0).cdf((day - (theta3 + theta4 * lat)) / (theta6/2.0))
    return torch.min(term1, term2)


def migration_loss(theta, y, lats, days, threshold, theta_mean, theta_prec, grid_vec):
    probs = threshold * m_torch(lats, days, theta)
    probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
    llh = torch.mean(y*torch.log(probs) + (1 - y)*torch.log(1 - probs)) # easier to tune penalty with average
    residual = m_torch(grid_vec[:,0], grid_vec[:,1], theta) - m_torch(grid_vec[:,0], grid_vec[:,1], theta_mean)
    func_penalty = -theta_prec * torch.mean(residual**2)
    return -(llh + func_penalty)


def fit_migration_model(y_train, lat_train, day_train, threshold_train, theta_mean, theta_prec,
                        theta_init=None, theta_prev=None, lr=0.1, num_epochs=10000, tol=0.0001,
                        max_change=np.inf, dtype=torch.float32):
    val_list = [y_train, lat_train, day_train, threshold_train, theta_mean]
    y_train_tensor, lat_train_tensor, day_train_tensor, threshold_train_tensor, theta_mean_tensor = [torch.tensor(val, dtype=dtype) for val in val_list]
    grid_vec = torch.cartesian_prod(torch.arange(60.0, 70.0, 1), torch.arange(1.0, 366.0, 10))
    
    if theta_init is None:
        theta = torch.tensor(theta_mean, requires_grad=True, dtype=dtype)
    else:
        theta = torch.tensor(theta_init, requires_grad=True, dtype=dtype)
        
    optimizer = torch.optim.Adam([theta], lr=lr)
    losses = []
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training migration model", mininterval=10):
        optimizer.zero_grad()
        loss = migration_loss(theta, y_train_tensor, lat_train_tensor, day_train_tensor, 
                              threshold_train_tensor, theta_mean_tensor, theta_prec, grid_vec)
        loss.backward()
        optimizer.step()
        
        if theta_init is not None and max_change < np.inf:
            with torch.no_grad():
                residual = m_torch(grid_vec[:,0], grid_vec[:,1], theta) - m_torch(grid_vec[:,0], grid_vec[:,1], theta_prev)
                change = torch.mean(residual**2)
                if change  > max_change:
                    print(f"Loop breaked at iteration {epoch}")
                    break

        if (epoch + 1) % 100 == 0 or epoch == 0:
            losses.append(loss.item())

        if len(losses)>2 and np.abs(losses[-1] - losses[-2])/losses[-2] < tol:
            break
    
    return theta.detach().numpy()