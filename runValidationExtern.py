from contiformer import ContiFormer
import os
import torch
import torch.optim as optim
import hydra
from omegaconf import DictConfig
import numpy as np
from torch.utils.data import DataLoader
from dataset_timeSeries import TimeSeriesDataset_Interpolation_roundedInput
import random
import logging
import tqdm
import matplotlib.pyplot as plt

@hydra.main(config_path="config", config_name="training", version_base=None)
def main(cfg: DictConfig):
    device, log = setup_environment(cfg)
    _, val_dataloader = get_ds_timeSeries(cfg)
    model, optimizer = build_model_and_optimizer(cfg, device, log)

    epochs = 20
    for epoch in range(epochs):
        run_validation(model, val_dataloader, device, epoch, cfg=cfg, log=log, save_visuals=True)





class TqdmCompatibleHandler(logging.StreamHandler):
    """Logging handler that plays nicely with tqdm progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)



def run_validation(model, val_loader, device, epoch, cfg, log, save_visuals=True):
    """
    Run validation and return metrics for early stopping.

    Returns:
        dict: Dictionary containing MAE, RMSE, and other metrics
    """
    model.eval()
    metrics = {'mae': float('inf'), 'rmse': float('inf')}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
            groundTruth = batch["groundTruth"].to(device)
            timeSeries_noisy_original = batch["noisy_TimeSeries"].to(device)
            mask = batch["mask"].to(device)
            time_stamps_original = batch["time_stamps"].to(device)

            div_term_cpu = batch["div_term"].to(device)
            min_value_cpu = batch["min_value"].to(device)

            # mask_indices = torch.where(mask[0] == True)[0]
            # timeSeries_noisy = timeSeries_noisy_original[:, mask_indices].unsqueeze(-1)
            # time_stamps = time_stamps_original[0].detach().clone()[mask_indices]
            # time_stamps = time_stamps.reshape(1, -1, 1).repeat(timeSeries_noisy.size(0), 1, 1).to(device)
            # timeSeries_noisy = torch.cat((timeSeries_noisy.to(device), time_stamps), dim=-1).float()


            pred_x = model(timeSeries_noisy_original.unsqueeze(-1), mask)[0]
            div_term = div_term_cpu.unsqueeze(-1).unsqueeze(-1)
            min_value = min_value_cpu.unsqueeze(-1).unsqueeze(-1)
            groundTruth = groundTruth.unsqueeze(-1)

            # Inverse transform from [0, 1] back to original scale
            pred_x = pred_x * div_term + min_value
            groundTruth = groundTruth * div_term + min_value

            mae = torch.abs(pred_x - groundTruth).mean()
            rmse = torch.sqrt(((pred_x - groundTruth) ** 2).mean())

            # Additional metrics
            mape = torch.mean(torch.abs((pred_x - groundTruth) / (groundTruth + 1e-8))) * 100

            metrics = {'mae': mae.item(), 'rmse': rmse.item(), 'mape': mape.item()}
            log.info('Validation: Epoch: %d, MAE: %.4f, RMSE: %.4f, MAPE: %.2f%%',
                     epoch, mae.item(), rmse.item(), mape.item())

            if save_visuals:
                # Denormalize time for visualization and exports (model still uses [0, 1])
                time_scale = float(cfg.number_x_values) if cfg.number_x_values > 1 else 1.0
                time_stamps_plot = time_stamps_original[0].cpu() * time_scale

                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                ax.plot(time_stamps_plot,
                        (timeSeries_noisy_original[0].detach().cpu() * div_term_cpu[0].detach().cpu()) + min_value_cpu[0].detach().cpu(),
                        'o', markersize=2, alpha=0.5, label="Noisy Input")
                ax.plot(time_stamps_plot,
                        groundTruth[0].detach().cpu().squeeze(-1),
                        linewidth=2, label="Ground Truth")
                ax.plot(time_stamps_plot,
                        pred_x[0].detach().cpu().squeeze(-1),
                        linewidth=2, linestyle='--', label="Prediction")
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title(f'Epoch {epoch} | MAE: {mae.item():.4f} | RMSE: {rmse.item():.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)

                save_path = os.path.join(cfg.val_dir_pictures, f'vis_{epoch}.svg')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                loss_fig, loss_ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
                error_samples = np.arange(len(model.total_loss))
                loss_ax[0].plot(error_samples, model.total_loss, label="Total Loss", alpha=0.7)
                loss_ax[0].set_ylabel('Total Loss')
                loss_ax[1].plot(error_samples, model.L1_loss, label="L1 Loss", color='orange', alpha=0.7)
                loss_ax[1].set_ylabel('L1 Loss')
                loss_ax[2].plot(error_samples, model.gradient_loss, label="Gradient Loss", color='green', alpha=0.7)
                loss_ax[2].set_ylabel('Gradient Loss')
                loss_ax[2].set_xlabel('Training Steps')
                for ax in loss_ax:
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                loss_path = os.path.join(cfg.val_dir_pictures, f'loss_{epoch}.svg')
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                plt.close(loss_fig)

                loss_data_path = os.path.join(cfg.val_dir_data, f'loss_{epoch}.pkl')
                torch.save({
                    'total_loss': model.total_loss,
                    'l1_loss': model.L1_loss,
                    'gradient_loss': model.gradient_loss,
                    'smooth_loss': model.smooth_loss,
                }, loss_data_path)

                save_path = os.path.join(cfg.val_dir_data, f'pred_{epoch}.pkl')
                torch.save({
                    'pred': pred_x,
                    'target': groundTruth,
                    'samp': timeSeries_noisy_original,
                    'time_plot': time_stamps_plot,
                    'metrics': metrics,
                }, save_path)

            break






def get_logger(name):
    logger = logging.getLogger(name)
    filename = f'{name}.log'
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

    # Clear any pre-existing handlers so we do not get duplicate lines when Hydra/root logging is configured.
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    fh = logging.FileHandler(filename, mode='a+', encoding='utf-8')
    ch = TqdmCompatibleHandler()
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

    return logger


def setup_environment(cfg):
    os.makedirs(cfg.train_dir, exist_ok=True)
    os.makedirs(cfg.val_dir_pictures, exist_ok=True)
    os.makedirs(cfg.val_dir_data, exist_ok=True)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() and cfg.gpu >= 0 else 'cpu')
    log = get_logger(os.path.join(cfg.train_dir, 'log'))
    return device, log

def get_ds_timeSeries(cfg):
    train_count = cfg.train_count
    val_count = cfg.val_count
    x_values = np.arange(0, cfg.number_x_values)

    train_ds = TimeSeriesDataset_Interpolation_roundedInput(train_count, x_values, cfg)
    val_ds = TimeSeriesDataset_Interpolation_roundedInput(val_count, x_values, cfg)

    train_dataloader = DataLoader(train_ds, batch_size=cfg.batch_size)
    val_dataloader = DataLoader(val_ds, batch_size=1)

    return train_dataloader, val_dataloader


def build_model_and_optimizer(cfg, device, log):
        obs_dim = 1
        model = ContiFormer(input_size=obs_dim,
                            d_model=getattr(cfg, 'd_model', 256),
                            d_inner=getattr(cfg, 'd_inner', 256),
                            n_layers=getattr(cfg, 'n_layers', 3),
                            n_head=getattr(cfg, 'n_head', 4),
                            d_k=getattr(cfg, 'd_k', 64),
                            d_v=getattr(cfg, 'd_v', 64),
                            dropout=getattr(cfg, 'dropout', 0.1),
                            actfn_ode=getattr(cfg, 'actfn', "softplus"),
                            layer_type_ode=getattr(cfg, 'layer_type_ode', "concat"),
                            zero_init_ode=getattr(cfg, 'zero_init_ode', True),
                            atol_ode=getattr(cfg, 'atol', 1e-6),
                            rtol_ode=getattr(cfg, 'rtol', 1e-6),
                            method_ode=getattr(cfg, 'method', "rk4"),
                            linear_type_ode=getattr(cfg, 'linear_type_ode', "inside"),
                            regularize=getattr(cfg, 'regularize', 256),
                            approximate_method=getattr(cfg, 'approximate_method', "last"),
                            nlinspace=getattr(cfg, 'nlinspace', 3),
                            interpolate_ode=getattr(cfg, 'interpolate_ode', "linear"),
                            itol_ode=getattr(cfg, 'itol_ode', 1e-2),
                            add_pe=getattr(cfg, 'add_pe', False),
                            normalize_before=getattr(cfg, 'normalize_before', False),
                            # max_length=getattr(cfg, 'max_length', 100),
                            ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

        if cfg.train_dir is not None:
            ckpt_path = os.path.join(cfg.train_dir, f'ckpt_{cfg.model_name}.pth')
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log.info('Loaded ckpt from {}'.format(ckpt_path))

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f'Model: {cfg.model_name} | Total params: {total_params:,} | Trainable: {trainable_params:,}')
        log.info(f'Architecture: d_model={getattr(cfg, "d_model", 64)}, n_layers={getattr(cfg, "n_layers", 3)}, n_head={getattr(cfg, "n_head", 4)}')

        return model, optimizer



if __name__ == '__main__':
    main()