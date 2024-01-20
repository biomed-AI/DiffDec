import torch
import torch.nn.functional as F
import numpy as np
import math

from src import utils
from src.egnn import Dynamics
from src.noise import GammaNetwork, PredefinedNoiseSchedule
from typing import Union

class EDM(torch.nn.Module):
    def __init__(
            self,
            dynamics: Union[Dynamics],
            in_node_nf: int,
            n_dims: int,
            timesteps: int = 1000,
            noise_schedule='learned',
            noise_precision=1e-4,
            loss_type='vlb',
            norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.),
    ):
        super().__init__()
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned with a vlb objective'
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)

        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases

    def forward(self, x, h, x_batch_new, h_batch_new, node_mask_batch_new, scaffold_mask_batch_new, 
                scaffold_mask_ori_batch_new, rgroup_mask, rgroup_mask_batch_new, rgroup_mask_ori_batch_new, 
                edge_mask, context, center_of_mass_mask, batch_new_len_tensor):

        x_batch_new = utils.remove_partial_mean_with_mask(x_batch_new, node_mask_batch_new, center_of_mass_mask)
        utils.assert_partial_mean_zero_with_mask(x_batch_new, node_mask_batch_new, center_of_mass_mask)

        # ori start
        # Normalization and concatenation
        # x, h = self.normalize(x, h)
        # xh = torch.cat([x, h], dim=2)

        # Volume change loss term
        delta_log_px = self.delta_log_px(rgroup_mask).mean()

        # Sample t
        t_int = torch.randint(0, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1
        t = t_int / self.T
        s = s_int / self.T

        # Masks for t=0 and t>0
        t_is_zero = (t_int == 0).squeeze().float()
        t_is_not_zero = 1 - t_is_zero

        # Compute gamma_t and gamma_s according to the noise schedule
        gamma_t = self.inflate_batch_array(self.gamma(t), x)
        gamma_s = self.inflate_batch_array(self.gamma(s), x)

        # Compute alpha_t and sigma_t from gamma
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample noise
        # Note: only for rgroup
        eps_t = self.sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), mask=rgroup_mask)

        eps_t_batch_new = torch.repeat_interleave(eps_t, batch_new_len_tensor, dim=0).to(rgroup_mask.device)

        x_batch_new, h_batch_new = self.normalize(x_batch_new, h_batch_new)
        xh_batch_new = torch.cat([x_batch_new, h_batch_new], dim=2)


        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        # Note: keep scaffold unchanged
        # z_t = alpha_t * xh + sigma_t * eps_t
        # z_t = xh * scaffold_mask + z_t * rgroup_mask

        alpha_t_batch_new = torch.repeat_interleave(alpha_t, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
        sigma_t_batch_new = torch.repeat_interleave(sigma_t, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
        t_batch_new = torch.repeat_interleave(t, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
        gamma_t_batch_new = torch.repeat_interleave(gamma_t, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
        gamma_s_batch_new = torch.repeat_interleave(gamma_s, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
        if len(t_is_zero.shape) != 0:
            t_is_zero_batch_new = torch.repeat_interleave(t_is_zero, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
        else:
            t_is_zero_batch_new = t_is_zero
        if len(t_is_not_zero.shape) != 0:
            t_is_not_zero_batch_new = torch.repeat_interleave(t_is_not_zero, batch_new_len_tensor, dim=0).to(rgroup_mask.device)
        else:
            t_is_not_zero_batch_new = t_is_not_zero

        z_t = alpha_t_batch_new * xh_batch_new + sigma_t_batch_new * eps_t_batch_new
        z_t = xh_batch_new * scaffold_mask_ori_batch_new + z_t * rgroup_mask_ori_batch_new

        # Neural net prediction
        eps_t_hat = self.dynamics.forward(
            xh=z_t,
            t=t_batch_new,
            node_mask=node_mask_batch_new,
            rgroup_mask=rgroup_mask_batch_new,
            context=context,
            edge_mask=edge_mask,
        )

        eps_t_hat = eps_t_hat * rgroup_mask_batch_new

        eps_t_batch_new = eps_t_batch_new * rgroup_mask_batch_new

        # Computing basic error (further used for computing NLL and L2-loss)
        error_t = self.sum_except_batch((eps_t_batch_new - eps_t_hat) ** 2)

        # Computing L2-loss for t>0
        normalization = (self.n_dims + self.in_node_nf) * self.numbers_of_nodes(rgroup_mask_batch_new)
        l2_loss = error_t / normalization
        l2_loss = l2_loss.mean()

        # The KL between q(z_T | x) and p(z_T) = Normal(0, 1) (should be close to zero)
        kl_prior = self.kl_prior(xh_batch_new, rgroup_mask_batch_new).mean()

        # Computing NLL middle term
        SNR_weight = (self.SNR(gamma_s_batch_new - gamma_t_batch_new) - 1).squeeze(1).squeeze(1)
        loss_term_t = self.T * 0.5 * SNR_weight * error_t
        loss_term_t = (loss_term_t * t_is_not_zero_batch_new).sum() / t_is_not_zero_batch_new.sum()

        # Computing noise returned by dynamics
        noise = torch.norm(eps_t_hat, dim=[1, 2])
        noise_t = (noise * t_is_not_zero_batch_new).sum() / t_is_not_zero_batch_new.sum()

        if t_is_zero_batch_new.sum() > 0:
            # The _constants_ depending on sigma_0 from the
            # cross entropy term E_q(z0 | x) [log p(x | z0)]
            neg_log_constants = -self.log_constant_of_p_x_given_z0(x_batch_new, rgroup_mask_batch_new)

            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and selected only relevant via masking
            loss_term_0 = -self.log_p_xh_given_z0_without_constants(h_batch_new, z_t, gamma_t_batch_new, eps_t_batch_new, eps_t_hat, rgroup_mask_batch_new)
            loss_term_0 = loss_term_0 + neg_log_constants
            loss_term_0 = (loss_term_0 * t_is_zero_batch_new).sum() / t_is_zero_batch_new.sum()

            # Computing noise returned by dynamics
            noise_0 = (noise * t_is_zero_batch_new).sum() / t_is_zero_batch_new.sum()
        else:
            loss_term_0 = 0.
            noise_0 = 0.

        return delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0

    @torch.no_grad()
    def sample_chain(self, x, h, node_mask, scaffold_mask, rgroup_mask, edge_mask, context, keep_frames=None):
        n_samples = x.size(0)
        n_nodes = x.size(1)

        # Normalization and concatenation
        x, h, = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        # Initial rgroup sampling from N(0, I)
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, mask=rgroup_mask)
        z = xh * scaffold_mask + z * rgroup_mask

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Sample p(z_s | z_t)
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt_only_rgroup(
                s=s_array,
                t=t_array,
                z_t=z,
                node_mask=node_mask,
                scaffold_mask=scaffold_mask,
                rgroup_mask=rgroup_mask,
                edge_mask=edge_mask,
                context=context,
            )
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z)

        # Finally sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0_only_rgroup(
            z_0=z,
            node_mask=node_mask,
            scaffold_mask=scaffold_mask,
            rgroup_mask=rgroup_mask,
            edge_mask=edge_mask,
            context=context,
        )
        chain[0] = torch.cat([x, h], dim=2)

        return chain

    def sample_p_zs_given_zt_only_rgroup(self, s, t, z_t, node_mask, scaffold_mask, rgroup_mask, edge_mask, context):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only rgroup features and coords"""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_t)
        sigma_s = self.sigma(gamma_s, target_tensor=z_t)
        sigma_t = self.sigma(gamma_t, target_tensor=z_t)

        # Neural net prediction.
        eps_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            node_mask=node_mask,
            rgroup_mask=rgroup_mask,
            context=context,
            edge_mask=edge_mask,
        )
        eps_hat = eps_hat * rgroup_mask

        # Compute mu for p(z_s | z_t)
        mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat

        # Compute sigma for p(z_s | z_t)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample z_s given the parameters derived from zt
        z_s = self.sample_normal(mu, sigma, rgroup_mask)
        z_s = z_t * scaffold_mask + z_s * rgroup_mask

        return z_s

    def sample_p_xh_given_z0_only_rgroup(self, z_0, node_mask, scaffold_mask, rgroup_mask, edge_mask, context):
        """Samples x ~ p(x|z0). Samples only rgroup features and coords"""
        zeros = torch.zeros(size=(z_0.size(0), 1), device=z_0.device)
        gamma_0 = self.gamma(zeros)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        eps_hat = self.dynamics.forward(
            t=zeros,
            xh=z_0,
            node_mask=node_mask,
            rgroup_mask=rgroup_mask,
            edge_mask=edge_mask,
            context=context
        )
        eps_hat = eps_hat * rgroup_mask

        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=rgroup_mask)
        xh = z_0 * scaffold_mask + xh * rgroup_mask

        x, h = xh[:, :, :self.n_dims], xh[:, :, self.n_dims:]
        x, h = self.unnormalize(x, h)
        h = F.one_hot(torch.argmax(h, dim=2), self.in_node_nf) * node_mask

        return x, h

    def compute_x_pred(self, eps_t, z_t, gamma_t):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.sigma(gamma_t, target_tensor=eps_t)
        alpha_t = self.alpha(gamma_t, target_tensor=eps_t)
        x_pred = 1. / alpha_t * (z_t - sigma_t * eps_t)
        return x_pred

    def kl_prior(self, xh, mask):
        """
        Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss.
        However, you compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part)
        sigma_T_x = self.sigma(gamma_T, mu_T_x).view(-1)  # Remove inflate, only keep batch dimension for x-part
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = self.gaussian_kl(mu_T_h, sigma_T_h, zeros, ones)

        # Compute KL for x-part
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        d = self.dimensionality(mask)
        kl_distance_x = self.gaussian_kl_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=d)

        return kl_distance_x + kl_distance_h

    def log_constant_of_p_x_given_z0(self, x, mask):
        batch_size = x.size(0)
        degrees_of_freedom_x = self.dimensionality(mask)
        zeros = torch.zeros((batch_size, 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0)
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_p_xh_given_z0_without_constants(self, h, z_0, gamma_0, eps, eps_hat, mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_0
        z_h = z_0[:, :, self.n_dims:]

        # Take only part over x
        eps_x = eps[:, :, :self.n_dims]
        eps_hat_x = eps_hat[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0) * self.norm_values[1]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'
        log_p_x_given_z_without_constants = -0.5 * self.sum_except_batch((eps_x - eps_hat_x) ** 2)

        # Categorical features
        # Compute delta indicator masks
        h = h * self.norm_values[1] + self.norm_biases[1]
        estimated_h = z_h * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded
        centered_h = estimated_h - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=centered_h_cat, stdev=sigma_0_cat)
        log_p_h_proportional = torch.log(
            self.cdf_standard_gaussian((centered_h + 0.5) / sigma_0) -
            self.cdf_standard_gaussian((centered_h - 0.5) / sigma_0) +
            epsilon
        )

        # Normalize the distribution over the categories
        log_Z = torch.logsumexp(log_p_h_proportional, dim=2, keepdim=True)
        log_probabilities = log_p_h_proportional - log_Z

        # Select the log_prob of the current category using the onehot representation
        log_p_h_given_z = self.sum_except_batch(log_probabilities * h * mask)

        # Combine log probabilities for x and h
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, mask):
        z_x = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=mask.device,
            node_mask=mask
        )
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=mask.device,
            node_mask=mask
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sample_normal(self, mu, sigma, node_mask):
        """Samples from a Normal distribution."""
        eps = self.sample_combined_position_feature_noise(mu.size(0), mu.size(1), node_mask)
        return mu + sigma * eps

    def normalize(self, x, h):
        new_x = x / self.norm_values[0]
        new_h = (h.float() - self.norm_biases[1]) / self.norm_values[1]
        return new_x, new_h

    def unnormalize(self, x, h):
        new_x = x * self.norm_values[0]
        new_h = h * self.norm_values[1] + self.norm_biases[1]
        return new_x, new_h

    def unnormalize_z(self, z):
        assert z.size(2) == self.n_dims + self.in_node_nf
        x, h = z[:, :, :self.n_dims], z[:, :, self.n_dims:]
        x, h = self.unnormalize(x, h)
        return torch.cat([x, h], dim=2)

    def delta_log_px(self, mask):
        return -self.dimensionality(mask) * np.log(self.norm_values[0])

    def dimensionality(self, mask):
        return self.numbers_of_nodes(mask) * self.n_dims

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -self.expm1(self.softplus(gamma_s) - self.softplus(gamma_t)),
            target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    @staticmethod
    def numbers_of_nodes(mask):
        return torch.sum(mask.squeeze(2), dim=1)

    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,),
        or possibly more empty axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    @staticmethod
    def sum_except_batch(x):
        return x.view(x.size(0), -1).sum(-1)

    @staticmethod
    def expm1(x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def gaussian_kl(q_mu, q_sigma, p_mu, p_sigma):
        """
        Computes the KL distance between two normal distributions.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
        kl = torch.log(p_sigma / q_sigma) + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2) - 0.5
        return EDM.sum_except_batch(kl)

    @staticmethod
    def gaussian_kl_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
        """
        Computes the KL distance between two normal distributions taking the dimension into account.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
            d: dimension
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
        mu_norm_2 = EDM.sum_except_batch((q_mu - p_mu) ** 2)
        return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma ** 2 + mu_norm_2) / (p_sigma ** 2) - 0.5 * d