"""Joint flow-matching losses and exponential moving-average weights."""

from __future__ import annotations

from copy import deepcopy

import pytorch_lightning as pl
import torch

from spagc.reconstruction.transport import Sampler, create_transport


class DeepSpatialModule(pl.LightningModule):
    """
    DeepSpatial Module for Training & Inference.

    Parameters
    ----------
    args : dict or argparse.Namespace
        Configuration dictionary containing hyperparameters such as learning rate,
        path type, and sampling settings.
    model : torch.nn.Module
        The core neural network architecture (e.g., the GiT model) that predicts
        the velocity fields.
    """

    def __init__(self, args, model, histology_runtime=None):
        """
        Initializes an LightningModule instance for DeepSpatial.
        """
        super().__init__()
        self.save_hyperparameters(args)
        self.model = model
        self.histology = histology_runtime
        if self.histology and (
            self.hparams.path_type != "Linear" or self.hparams.prediction != "velocity"
        ):
            raise ValueError(
                "Histology mode currently supports Linear molecular/cell paths and velocity prediction only"
            )
        self.ema_decay = self.hparams.get("ema_decay", 0.999)
        self.ema_model = deepcopy(self.model)
        self._freeze(self.ema_model)
        self.transport = create_transport(
            path_type=self.hparams.path_type,
            prediction=self.hparams.prediction,
            train_eps=self.hparams.get("train_eps", 0.02),
            sample_eps=self.hparams.get("sample_eps", 0.02),
        )
        self.sampler = Sampler(self.transport)

    def _freeze(self, module):
        """Freeze model parameters for EMA or evaluation."""
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def configure_optimizers(self):
        """Initialize AdamW optimizer with weight decay."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.get("weight_decay", 1e-05),
        )

    def _shared_step(self, batch):
        """
        Computes joint Flow Matching losses across spatial and molecular dimensions.
        """
        (x0, x1) = (batch["x0"], batch["x1"])
        (g0, g1) = (batch["g0"], batch["g1"])
        (c0, c1) = (batch["c0"], batch["c1"])
        (z0, z1) = (batch["z0"], batch["z1"])
        delta_z = batch["delta_z"]
        (t, _, _) = self.transport.sample(x1)
        (_, xt, ux_t) = self.transport.path_sampler.plan(t, x0, x1)
        if self.histology and self.histology.config.use_morphology_path:
            (xt, ux_t) = self.histology.plan(batch["path_id"], t)
        (_, gt, ug_t) = self.transport.path_sampler.plan(t, g0, g1)
        (_, ct, uc_t) = self.transport.path_sampler.plan(t, c0, c1)
        (_, zt, _) = self.transport.path_sampler.plan(t, z0, z1)
        (vx_pred, vg_pred, vc_pred) = self.model(
            xt=xt, gt=gt, t=t, zt=zt, delta_z=delta_z, ct=ct
        )
        loss_x = self.transport.loss_fn(vx_pred, x0, xt, t, ux_t).mean()
        loss_g = self.transport.loss_fn(vg_pred, g0, gt, t, ug_t).mean()
        loss_c = self.transport.loss_fn(vc_pred, c0, ct, t, uc_t).mean()
        if not self.hparams.get("use_celltype", True):
            loss_c = loss_c * 0
        lambda_g = self.hparams.get("lambda_g", 0.1)
        lambda_c = self.hparams.get("lambda_c", 10.0)
        loss_total = loss_x + lambda_g * loss_g + lambda_c * loss_c
        return {
            "loss": loss_total,
            "loss_x": loss_x,
            "loss_g": loss_g,
            "loss_c": loss_c,
        }

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("loss", loss["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("loss_x", loss["loss_x"], on_epoch=True)
        self.log("loss_g", loss["loss_g"], on_epoch=True)
        self.log("loss_c", loss["loss_c"], on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA parameters after each optimizer step."""
        self._update_ema()

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA model weights with exponential decay."""
        for ema_param, model_param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                model_param.data, alpha=1 - self.ema_decay
            )

    def on_load_checkpoint(self, checkpoint):
        """Ensure EMA model is also loaded from checkpoint."""
        self._update_ema()

    @torch.no_grad()
    def sample(self, batch, mode="ODE", steps=20):
        """
        Integrates the learned flow field to reconstruct intermediate biological states.


        Parameters
        ----------
        batch : dict
            A dictionary containing the initial states (`x0`, `g0`, `c0`), the physical
            Z-depth conditions (`z0`, `z1`, `delta_z`), and other necessary tensors.
        mode : str, optional
            The integration mode, either `"ODE"` (Ordinary Differential Equation) or
            `"SDE"` (Stochastic Differential Equation). By default `"ODE"`.
        steps : int, optional
            The number of integration steps from the source to the target slice.
            Higher values yield more accurate but slower trajectories. By default 20.

        Returns
        -------
        dict
            A dictionary containing the full integration trajectories:
            - `'x_traj'` : torch.Tensor of shape `(Steps, Batch, 2)`
            - `'g_traj'` : torch.Tensor of shape `(Steps, Batch, Gene_Dim)`
            - `'c_traj_discrete'` : torch.Tensor of shape `(Steps, Batch)` containing discrete cell type labels.
        """
        self.ema_model.eval()
        if self.histology and mode != "ODE":
            raise ValueError(
                "Morphology-guided paths support ODE sampling, not linear-path SDE score conversion"
            )
        if steps < 2:
            raise ValueError("steps must be at least 2")
        sample_config = {
            "num_steps": steps,
            "sampling_method": self.hparams.get("sampling_method", "dopri5"),
            "atol": self.hparams.get("atol", 1e-05),
            "rtol": self.hparams.get("rtol", 1e-05),
        }
        if self.histology and sample_config["sampling_method"] in (
            "dopri5",
            "dopri8",
            "bosh3",
            "adaptive_heun",
            "fehlberg2",
        ):
            sample_config["options"] = {"step_t": [1.0], "first_step": 0.01}
        sampler_fn = (
            self.sampler.sample_ode(**sample_config)
            if mode == "ODE"
            else self.sampler.sample_sde(**sample_config)
        )
        (x0, g0, c0) = (batch["x0"], batch["g0"], batch["c0"])
        (x_dim, g_dim) = (x0.shape[-1], g0.shape[-1])
        (z0, z1, delta_z) = (batch["z0"], batch["z1"], batch["delta_z"])
        init_state = torch.cat([x0, g0, c0], dim=-1)

        def velocity_field_wrapper(joint_state_t, t):
            xt = joint_state_t[..., :x_dim]
            gt = joint_state_t[..., x_dim : x_dim + g_dim]
            ct = joint_state_t[..., x_dim + g_dim :]
            if torch.is_tensor(t):
                t_val = t.item() if t.dim() == 0 else t[0].item()
            else:
                t_val = t
            t_tensor = torch.full(
                (xt.shape[0],), t_val, device=xt.device, dtype=xt.dtype
            )
            condition_t = t_tensor.clamp(0, 1) if self.histology else t_tensor
            (_, zt, _) = self.transport.path_sampler.plan(condition_t, z0, z1)
            reverse = batch.get("reverse", False) and self.histology is not None
            model_t = 1 - t_tensor if reverse else t_tensor
            model_gap = -delta_z if reverse else delta_z
            if not self.hparams.get("use_celltype", True):
                ct = torch.zeros_like(ct)
            (vx, vg, vc) = self.ema_model(
                xt=xt, gt=gt, t=model_t, zt=zt, delta_z=model_gap, ct=ct
            )
            if not self.hparams.get("use_celltype", True):
                vc = torch.zeros_like(vc)
            if reverse:
                (vx, vg, vc) = (-vx, -vg, -vc)
            return torch.cat([vx, vg, vc], dim=-1)

        trajectory = sampler_fn(init_state, velocity_field_wrapper)
        if isinstance(trajectory, (list, tuple)):
            trajectory = torch.stack(trajectory, dim=0)
        x_traj = trajectory[..., :x_dim]
        g_traj = trajectory[..., x_dim : x_dim + g_dim]
        c_traj_cont = trajectory[..., x_dim + g_dim :]
        return {
            "x_traj": x_traj,
            "g_traj": g_traj,
            "c_traj_cont": c_traj_cont,
            "c_traj_discrete": torch.argmax(c_traj_cont, dim=-1),
        }
