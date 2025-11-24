
import os
os.environ["OMP_NUM_THREADS"] = '16'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
from torch.profiler import profile, ProfilerActivity, record_function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import copy, random, sys

from preprocess import get_data
from calibrate import calibrate
from recon import create_proj_geom, reconstruct
from models import DnCNN


def set_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)
  if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
        
set_seed(42)


def _show_batch(inp, tgt):
    inp_cpu = inp.detach().cpu().squeeze(1)
    tgt_cpu = tgt.detach().cpu().squeeze(1)

    batch_size = inp_cpu.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    if batch_size == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(2):
        axes[i, 0].imshow(inp_cpu[i], cmap='seismic')
        axes[i, 0].set_title(f'Input {i}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(tgt_cpu[i], cmap='seismic')
        axes[i, 1].set_title(f'Target {i}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()



def _get_batch(x_tilde_torch, batch_size):
  K, W, W = x_tilde_torch.shape
  idx = torch.randint(low=0, high=K, size=(batch_size,), device=device)
  expanded = x_tilde_torch.unsqueeze(0).expand(batch_size, -1, -1, -1)

  mask = torch.ones(batch_size, K, dtype=torch.bool, device=device)
  mask[torch.arange(batch_size), idx] = 0

  inputs = expanded[mask].reshape(batch_size, K-1, W, W).mean(dim=1)
  targets = x_tilde_torch[idx]
  return inputs.unsqueeze(1), targets.unsqueeze(1)


def normalize_tensor(x):
    x_min = x.min()
    x_max = x.max()
    if (x_max - x_min) < 1e-8:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def train_n2i(x_tilde_torch, model, gt, patience=2, batch_size=4, num_epochs=1000, lr=1e-3):
  print("Sarting training of \033[32mnoise2inverse\033[0m algorithm ... ")
  learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'Learnable parameters : {learnable_params/1e6:.1f}M') 
  '''
  for module in model.modules():
    if hasattr(module, 'reset_parameters'):
      module.reset_parameters()
  '''
  K = x_tilde_torch.size(0)
  x_tilde_sum = x_tilde_torch.sum(dim=0)
  x_tilde_in = (x_tilde_sum - x_tilde_torch) / (K - 1)
  loss_history = np.empty(num_epochs)
  ssim_history = np.zeros(num_epochs)

  best_ssim = -np.inf
  epochs_no_improve = 0
  weights = None

  model.train()
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  for epoch in range(1, num_epochs + 1):

    inp_list, tgt_list = _get_batch(x_tilde_torch, batch_size)
    inp = inp_list.to(device)
    tgt = tgt_list.to(device)

    optimizer.zero_grad()
    out = model(inp)
    loss = criterion(out, tgt)
    loss.backward()
    optimizer.step()

    epoch_loss = loss.item()
    loss_history[epoch-1] = epoch_loss

    if epoch % 50 == 0:
      #_show_batch(inp, tgt)
      #_show_batch(out, tgt)
      print(f"Epoch {epoch:02d}/{num_epochs} | loss = {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
      x_star_j = model(x_tilde_in.unsqueeze(1)).squeeze().cpu().numpy()
      x_star = np.mean(x_star_j, axis=0)
      x_star_range = x_star.max() - x_star.min()

      #ssim_centroid = np.empty(K)
      #for j in range(K):
      #  ssim_centroid[j] = ssim(x_star, x_star_j[j], data_range=x_star_range)
      #boost_ssim = np.mean(ssim_centroid)
      boost_ssim = ssim(x_star, gt, data_range=x_star_range)
      ssim_history[epoch - 1] = boost_ssim
      
      if boost_ssim >= best_ssim:
        best_ssim = boost_ssim
        weights = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
      else:
        epochs_no_improve += 1
      model.train()

      if epochs_no_improve == patience:
        continue
        print(f"Early stopping at epoch {epoch} (best SSIM: {best_ssim:.5f})")
        return weights, ssim_history[:epoch], loss_history[:epoch]
  return weights, ssim_history, loss_history


def denoise_n2i(sinogram, slice_num, angles, sdd, sod, pxl_size, K, algo_type, max_iter=15):
  full_proj_geom = create_proj_geom(sinogram, angles, sdd, sod, pxl_size)
  ncp_history, ncp_iter, recon = reconstruct(sinogram, 
                                             full_proj_geom, 
                                             algo=algo_type, 
                                             max_iter=max_iter)
  
  H, W, W = recon.shape
  x_tilde = np.empty((K, H, W, W), dtype=np.float32)
  for j in range(K):
    print(f'j{j} :', end='', flush=True)
    angles_j = angles[j::K]
    sino_j = sinogram[:,j::K,:]

    proj_geom_j = create_proj_geom(sino_j, angles_j, sdd, sod, pxl_size)
    _, _, x_tilde[j] = reconstruct(sinogram=sino_j, 
                                    proj_geom=proj_geom_j, 
                                    algo=algo_type, 
                                    max_iter=max_iter, 
                                    prev_ncp_iter=ncp_iter)
      
  model = DnCNN(1)
  model.to(device)
  x_tilde_torch = torch.from_numpy(x_tilde[:,slice_num]).to(device)
  x_tilde_torch = normalize_tensor(x_tilde_torch)
  weights, ssim_history, loss_history = train_n2i(x_tilde_torch, 
                                                  model)
  
  plt.plot(loss_history)
  plt.xscale('log')
  plt.yscale('log')
  plt.grid(True, which='both', ls='--')
  plt.show()
  plt.plot(ssim_history)
  plt.show()
  
  torch.save(weights, '../weights/n2i.pth')
  #model.load_state_dict(torch.load('../weights/n2i.pth', weights_only=False))
  model.eval()
  with torch.no_grad():
    x_star = model(x_tilde_torch.unsqueeze(1)).squeeze().mean(dim=0).cpu().numpy()
  return x_star



def main():
  gt = np.load('../n2i_data/pomd_fdk_rem22_from0_GT.npy')
  mid_slice = 48

  sdd = 572.0  # mm
  binning = 2*2
  pxl_size = 62e-3  # mm 
  trans_x, trans_y, trans_z, tilt, roll = 3.3563, -11.91, 531.56, np.deg2rad(-0.3817), np.deg2rad(1.1555)
  trans_x += 0
  trans_y *= 0
  tilt *= -1
  roll *= +1
  geom_params = {'sdd':sdd, 
                 'sod':trans_z, 
                 'pxl_size':pxl_size*binning, 
                 'tilt':tilt, 
                 'roll':roll, 
                 'shift_x':trans_x, 
                 'shift_y':trans_y}

  noisy_sino = np.load('../n2i_data/pomd_sino_noise500_22.npy')
  K = 4
  algo = 'FDK_CUDA'
  H, W, W = gt.shape
  angles = np.linspace(0.0, 2*np.pi, 1024, endpoint=False)
  x_tilde = np.empty((K, H, W, W), dtype=np.float32)
  for j in range(K):
    print(f'j{j} :', end='', flush=True)
    angles_j = angles[j::K]
    sino_j = noisy_sino[:,j::K,:]

    proj_geom_j = create_proj_geom(sino_j, angles_j, **geom_params)
    _, _, x_tilde[j] = reconstruct(sinogram=sino_j, 
                                    proj_geom=proj_geom_j, 
                                    algo=algo, 
                                    max_iter=30, 
                                    prev_ncp_iter=30)

  np.save('../n2i_data/x_tilde_fdk500.npy', x_tilde)
  plt.imshow(x_tilde[0,mid_slice])
  plt.show()
  
  x_tilde = np.load('../n2i_data/x_tilde_fdk500.npy')[:,:,50:400,50:400]
  ground_truth = np.load('../n2i_data/pomd_fdk_rem22_from0_GT.npy')[:,50:400,50:400]
  noisy = np.load('../n2i_data/pomd_fdk22_from0_noise500.npy')[:,50:400,50:400]
  mid_slice = 48 #noisy.shape[0] // 2 + 1

  gt_normed = normalize_tensor(torch.from_numpy(ground_truth[mid_slice])).numpy()

  model = DnCNN(1)
  model.to(device)
  x_tilde_torch = torch.from_numpy(x_tilde[:,mid_slice]).to(device)
  x_tilde_torch_norm = normalize_tensor(x_tilde_torch)
  weights, ssim_history, loss_history = train_n2i(x_tilde_torch_norm, 
                                                  model,
                                                  gt=gt_normed, 
                                                  num_epochs=500)
  
  plt.plot(loss_history)
  plt.xscale('log')
  plt.yscale('log')
  plt.grid(True, which='both', ls='--')
  plt.show()
  plt.plot(ssim_history)
  plt.show()
  
  torch.save(weights, '../weights/n2i_fdk500_48_500.pth')
  ground_truth = np.load('../n2i_data/pomd_fdk_rem22_from0_GT.npy')[:,50:400,50:400]
  mid_slice = ground_truth.shape[0] // 2 + 1
  gt_normed = normalize_tensor(torch.from_numpy(ground_truth[mid_slice])).numpy()

  x_tilde_sirt = np.load('../n2i_data/x_tilde_sirt500.npy')[:,:,50:400,50:400]
  noisy_sirt = np.load('../n2i_data/pomd_sirt22_from0_noise500.npy')[:,50:400,50:400]
  noisy_normed_sirt = normalize_tensor(torch.from_numpy(noisy_sirt[mid_slice])).numpy()
  x_tilde_fdk = np.load('../n2i_data/x_tilde_fdk500.npy')[:,:,50:400,50:400]
  noisy_fdk = np.load('../n2i_data/pomd_fdk22_from0_noise500.npy')[:,50:400,50:400]
  noisy_normed_fdk = normalize_tensor(torch.from_numpy(noisy_fdk[mid_slice])).numpy()

  #sirt
  model_sirt = DnCNN(1)
  model_sirt.to(device)
  x_tilde_sirt = torch.from_numpy(x_tilde_sirt[:,mid_slice]).to(device)
  model_sirt.load_state_dict(torch.load('../weights/n2i_sirt500_mid+1_1000.pth', weights_only=False))

  K = x_tilde_sirt.shape[0]
  x_tilde_sums = x_tilde_sirt.sum(dim=0)
  x_tilde_sirt = (x_tilde_sums - x_tilde_sirt) / (K - 1) 
  x_tilde_sirt = normalize_tensor(x_tilde_sirt)

  model_sirt.eval()
  with torch.no_grad():
    x_star_sirt = model_sirt(x_tilde_sirt.unsqueeze(1)).squeeze().mean(dim=0).cpu().numpy()
  
  #fdk
  model_fdk = DnCNN(1)
  model_fdk.to(device)
  x_tilde_fdk = torch.from_numpy(x_tilde_fdk[:,mid_slice]).to(device)
  model_fdk.load_state_dict(torch.load('../weights/n2i_fdk500_mid+1_500.pth', weights_only=False))

  K = x_tilde_fdk.shape[0]
  x_tilde_sumf = x_tilde_fdk.sum(dim=0)
  x_tilde_fdk = (x_tilde_sumf - x_tilde_fdk) / (K - 1) 
  x_tilde_fdk = normalize_tensor(x_tilde_fdk)

  model_fdk.eval()
  with torch.no_grad():
    x_star_fdk = model_fdk(x_tilde_fdk.unsqueeze(1)).squeeze().mean(dim=0).cpu().numpy()


  fig, axes = plt.subplots(1, 5, figsize=(20, 8), dpi=300)
  titles = ['POM-Dh', 'FDK', 'FDK + N2I', 'SIRT', 'SIRT + N2I']
  images = [gt_normed+1e-9, noisy_normed_fdk, x_star_fdk, noisy_normed_sirt, x_star_sirt]

  for ax, img, title in zip(axes, images, titles):
      im = ax.imshow(img, cmap='cubehelix')
      ax.set_title(title, fontsize=12)
      ax.axis('off')

      if title != 'POM-Dh':
          # Compute metrics
          p = psnr(gt_normed, img, data_range=1.0)
          s = ssim(gt_normed, img, data_range=1.0)

          # Text with red background box in bottom-left corner
          ax.text(
              0.4, 0.01,  # bottom-left inside axes
              f"PSNR: {p:.2f} dB\nSSIM: {s:.3f}",
              ha='left', va='bottom',
              color='white',
              fontsize=18,
              transform=ax.transAxes,
              bbox=dict(
                  facecolor='red',
                  edgecolor='black',
                  boxstyle='square,pad=0.3',
                  alpha=0.8
              )
          )

      fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

  plt.tight_layout()
  plt.savefig('../zz_output/mid_row.png', bbox_inches='tight')
  plt.savefig('../zz_output/mid_row.eps', format='eps', bbox_inches='tight')


if __name__ == "__main__":
  main()

