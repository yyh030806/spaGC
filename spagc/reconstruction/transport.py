"""Interpolation paths and numerical flow samplers."""

from __future__ import annotations

import enum

import numpy as np
import torch as th
from torchdiffeq import odeint


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))


def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


class ICPlan:
    """Linear Coupling Plan"""

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return (t, 1)

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return (1 - t, -1)

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization;"""
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        (sigma_t, d_sigma_t) = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * sigma_t**2 - sigma_t * d_sigma_t
        return (-drift, diffusion)

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
            "inccreasing-decreasing": norm * th.sin(np.pi * t) ** 2,
        }
        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        (alpha_t, d_alpha_t) = self.compute_alpha_t(t)
        (sigma_t, d_sigma_t) = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        (alpha_t, d_alpha_t) = self.compute_alpha_t(t)
        (sigma_t, d_sigma_t) = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        (drift, var) = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        (alpha_t, _) = self.compute_alpha_t(t)
        (sigma_t, _) = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt

    def compute_ut(self, t, x0, x1, xt):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        (_, d_alpha_t) = self.compute_alpha_t(t)
        (_, d_sigma_t) = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return (t, xt, ut)


class VPCPlan(ICPlan):
    """class for VP path flow matching"""

    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = lambda t: (
            -0.25 * (1 - t) ** 2 * (self.sigma_max - self.sigma_min)
            - 0.5 * (1 - t) * self.sigma_min
        )
        self.d_log_mean_coeff = lambda t: (
            0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min
        )

    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = self.log_mean_coeff(t)
        alpha_t = th.exp(alpha_t)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return (alpha_t, d_alpha_t)

    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = th.sqrt(1 - th.exp(p_sigma_t))
        d_sigma_t = th.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return (sigma_t, d_sigma_t)

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        """Compute the drift term of the SDE"""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return (-0.5 * beta_t * x, beta_t / 2)


class GVPCPlan(ICPlan):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)

    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = th.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return (alpha_t, d_alpha_t)

    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        sigma_t = th.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return (sigma_t, d_sigma_t)

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * th.tan(t * np.pi / 2))


class sde:
    """SDE solver class"""

    def __init__(self, drift, diffusion, *, t0, t1, num_steps, sampler_type):
        assert t0 < t1, "SDE sampler has to be in forward time"
        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return (x, mean_x)

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return (xhat + 0.5 * self.dt * (K1 + K2), xhat)

    def __forward_fn(self):
        """Select the requested stochastic integration step."""
        sampler_dict = {"Euler": self.__Euler_Maruyama_step, "Heun": self.__Heun_step}
        try:
            sampler = sampler_dict[self.sampler_type]
        except KeyError as exc:
            raise NotImplementedError(
                f"Unknown SDE sampler: {self.sampler_type}"
            ) from exc
        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                (x, mean_x) = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)
        return samples


class ode:
    """ODE solver class"""

    def __init__(
        self, drift, *, t0, t1, sampler_type, num_steps, atol, rtol, options=None
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"
        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type
        self.options = options

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t = (
                th.ones(x[0].size(0)).to(device) * t
                if isinstance(x, tuple)
                else th.ones(x.size(0)).to(device) * t
            )
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol,
            options=self.options,
        )
        return samples


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps):
        path_options = {
            PathType.LINEAR: ICPlan,
            PathType.GVP: GVPCPlan,
            PathType.VP: VPCPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched
        """
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])

        def _fn(x):
            return -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0

        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [VPCPlan]:
            t1 = 1 - eps if not sde or last_step_size == 0 else 1 - last_step_size
        elif type(self.path_sampler) in [ICPlan, GVPCPlan] and (
            self.model_type != ModelType.VELOCITY or sde
        ):
            t0 = (
                eps
                if diffusion_form == "SBDM"
                and sde
                or self.model_type != ModelType.VELOCITY
                else 0
            )
            t1 = 1 - eps if not sde or last_step_size == 0 else 1 - last_step_size
        if reverse:
            (t0, t1) = (1 - t0, 1 - t1)
        return (t0, t1)

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """
        x0 = th.randn_like(x1)
        (t0, t1) = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return (t, x0, x1)

    def loss_fn(self, model_output, x0, xt, t, ut):
        if self.model_type == ModelType.VELOCITY:
            loss = mean_flat((model_output - ut) ** 2)
        else:
            (_, drift_var) = self.path_sampler.compute_drift(xt, t)
            (sigma_t, _) = self.path_sampler.compute_sigma_t(expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / sigma_t**2
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            if self.model_type == ModelType.NOISE:
                loss = mean_flat(weight * (model_output - x0) ** 2)
            else:
                loss = mean_flat(weight * (model_output * sigma_t + x0) ** 2)
        return loss

    def training_losses(self, model, x1, model_kwargs=None):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs is None:
            model_kwargs = {}
        (t, x0, x1) = self.sample(x1)
        (t, xt, ut) = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        (B, *_, C) = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)
        terms = {}
        terms["pred"] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = mean_flat((model_output - ut) ** 2)
        else:
            (_, drift_var) = self.path_sampler.compute_drift(xt, t)
            (sigma_t, _) = self.path_sampler.compute_sigma_t(expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / sigma_t**2
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * (model_output - x0) ** 2)
            else:
                terms["loss"] = mean_flat(weight * (model_output * sigma_t + x0) ** 2)
        return terms

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            (drift_mean, drift_var) = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_ode(x, t, model, **model_kwargs):
            (drift_mean, drift_var) = self.path_sampler.compute_drift(x, t)
            (sigma_t, _) = self.path_sampler.compute_sigma_t(expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, (
                "Output shape from ODE solver must match input shape"
            )
            return model_output

        return body_fn

    def get_score(self):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:

            def score_fn(x, t, model, **kwargs):
                return (
                    model(x, t, **kwargs)
                    / -self.path_sampler.compute_sigma_t(expand_t_like_x(t, x))[0]
                )
        elif self.model_type == ModelType.SCORE:

            def score_fn(x, t, model, **kwagrs):
                return model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:

            def score_fn(x, t, model, **kwargs):
                return self.path_sampler.get_score_from_velocity(
                    model(x, t, **kwargs), x, t
                )
        else:
            raise NotImplementedError()
        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(self, transport):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def step(self, zt, t, dt, model, **model_kwargs):
        """
        Performs a single reverse ODE step using the Euler method.
        This is the "easy way" to call one step.

        Args:
            zt: The current latent tensor at time t.
            t: The current timestep (as a tensor, e.g., from [0, 1]).
            dt: The step size for this step (e.g., 1 / num_steps).
            model: The diffusion model to use for prediction (e.g., the EMA model).
            **model_kwargs: Additional arguments for the model (like class labels).

        Returns:
            The predicted latent tensor at time t-dt.
        """
        drift_pred = self.drift(zt, t, model, **model_kwargs)
        return zt + drift_pred * dt

    def __get_sde_diffusion_and_drift(
        self, *, diffusion_form="SBDM", diffusion_norm=1.0
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )
            return diffusion

        def sde_drift(x, t, model, **kwargs):
            return self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(
                x, t, model, **kwargs
            )

        sde_diffusion = diffusion_fn
        return (sde_drift, sde_diffusion)

    def __get_last_step(self, sde_drift, *, last_step, last_step_size):
        """Get the last step function of the SDE solver"""
        if last_step is None:

            def last_step_fn(x, t, model, **model_kwargs):
                return x
        elif last_step == "Mean":

            def last_step_fn(x, t, model, **model_kwargs):
                return x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t

            def last_step_fn(x, t, model, **model_kwargs):
                return x / alpha(t)[0][0] + sigma(t)[0][0] ** 2 / alpha(t)[0][
                    0
                ] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":

            def last_step_fn(x, t, model, **model_kwargs):
                return x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()
        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
        t_forward=None,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """
        if last_step is None:
            last_step_size = 0.0
        (sde_drift, sde_diffusion) = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm
        )
        (t0, t1) = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )
        if t_forward:
            t0 = t_forward
        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )
        last_step_fn = self.__get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size
        )

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            assert len(xs) == num_steps, "Samples does not match the number of steps"
            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-06,
        rtol=0.001,
        reverse=False,
        t_forward=None,
        options=None,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:

            def drift(x, t, model, **kwargs):
                return self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift
        (t0, t1) = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )
        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            options=options,
        )
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-06,
        rtol=0.001,
        t_forward=None,
    ):
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        def _likelihood_drift(x, t, model, **model_kwargs):
            (x, _) = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(
                    th.sum(self.drift(x, t, model, **model_kwargs) * eps), x
                )[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        (t0, t1) = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )
        if t_forward:
            t0 = t_forward
        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            (drift, delta_logp) = _ode.sample(input, model, **model_kwargs)
            (drift, delta_logp) = (drift[-1], delta_logp[-1])
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return (logp, drift)

        return _sample_fn


def create_transport(
    path_type="Linear",
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    """
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY
    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE
    path_choice = {"Linear": PathType.LINEAR, "GVP": PathType.GVP, "VP": PathType.VP}
    path_type = path_choice[path_type]
    if path_type in [PathType.VP]:
        train_eps = 1e-05 if train_eps is None else train_eps
        sample_eps = 0.001 if train_eps is None else sample_eps
    elif (
        path_type in [PathType.GVP, PathType.LINEAR]
        and model_type != ModelType.VELOCITY
    ):
        train_eps = 0.001 if train_eps is None else train_eps
        sample_eps = 0.001 if train_eps is None else sample_eps
    else:
        train_eps = 0
        sample_eps = 0
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )
    return state
