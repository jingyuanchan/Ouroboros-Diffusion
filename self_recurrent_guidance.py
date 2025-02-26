import torch

class DDIM_self_recurrent_guide(object):
    def __init__(self):
        pass
    
    # Main function for the DDIM self-recurrent guidance
    def fifo_onestep_self_recurrent_guide(self, cond, shape, latents=None, timesteps=None, indices=None,
                                          unconditional_guidance_scale=1., unconditional_conditioning=None,
                                          **kwargs):
        """
        Performs a single step of DDIM with self-recurrent guidance.
        
        Args:
            cond: Conditioning tensor for DDIM.
            shape: Shape of the latent tensors.
            latents: Latent tensors to be processed.
            timesteps: Timesteps to be applied.
            indices: Indices of timesteps to consider.
            unconditional_guidance_scale: Scaling factor for unconditional guidance.
            unconditional_conditioning: Conditioning tensor for unconditional guidance.
            **kwargs: Additional arguments.

        Returns:
            Updated latents and predicted x_0 after applying DDIM with self-recurrent guidance.
        """
        device = self.model.betas.device        
        b, _, f, _, _ = shape

        ts = torch.Tensor(timesteps.copy()).to(device=device, dtype=torch.long)
        noise_pred, score = self.unet_self_recurrrent_guide(latents, cond, ts,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    **kwargs)
        # If the guidance gradient is not None, apply self-recurrent guidance
        if score is not None:
            latents, pred_x0 = self.ddim_step_self_recurrent_guide(latents, noise_pred, indices, score)
        else:
            latents, pred_x0 = self.ddim_step(latents, noise_pred, indices)
        return latents.detach(), pred_x0.detach()
    
    # Compute the subject loss and noise
    def unet_self_recurrrent_guide(self, x, c, t, unconditional_guidance_scale=1.,
                                   unconditional_conditioning=None, window=None, guide_windows=None, spatial_weights=[0.0, 0.0, 0.0]):
        """
        Applies U-Net with self-recurrent guidance for computing subject loss and noise.

        Args:
            x: Input latent tensor.
            c: Conditioning tensor.
            t: Timesteps tensor.
            unconditional_guidance_scale: Scaling factor for unconditional guidance.
            unconditional_conditioning: Conditioning tensor for unconditional guidance.
            window: Current window index.
            guide_windows: List of windows that require guidance.
            spatial_weights: Weights for spatial dimensions.

        Returns:
            Predicted noise tensor and the computed score for guidance.
        """
        reference_window = 0  # Reference window for computing long-term memory

        # If the current window does not require self-recurrent guidance
        if window not in guide_windows:
            return self.unet(x, c, t, unconditional_guidance_scale, unconditional_conditioning), None
        else:
            # Enable gradient computation for the input
            x = x.detach()
            x.requires_grad = True
            e_t = self.model.apply_model(x, t, c, is_cfg=False)

            # Retrieve the key and mask for the current window to compute short-term memory K'_{t}^{\tau+t}
            attention_map = self.model.model.diffusion_model.attention_map[window]
            cur_key = attention_map.cur_key
            cur_avg_mask = attention_map.cur_avg_mask

            # Retrieve the clean reference key and mask to compute long-term memory(LTM) K'_\text{ltm} as in Eq. 5
            clean_attention_map = self.model.model.diffusion_model.attention_map[reference_window]
            clean_reference_k = clean_attention_map.moving_avg_key # Moving avg is applied to LTM as in Eq. 5
            clean_avg_mask = clean_attention_map.moving_avg_mask

            # Compute the weighted sum of losses across different layers
            spatial_weights = torch.tensor(spatial_weights).to(x.dtype).to(x.device)
            weight_each_layer = torch.repeat_interleave(spatial_weights / 100.0, repeats=3).detach()

            # Compute the subject loss for the current layer
            loss = compute_subj_loss(cur_key, clean_reference_k, cur_avg_mask, clean_avg_mask,  # type: ignore
                                     weight_each_layer, calc_background=False)  # Simplified for brevity
            
            # Compute the gradient of the subject loss with respect to the input x
            score = torch.autograd.grad(loss, x, allow_unused=True)[0]
            assert score is not None, 'score is None'
            score = score.detach()

            # Apply U-Net to the input x for condition-free guidance
            with torch.no_grad():
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, is_cfg=True)

            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            return e_t, score
    
    # Helper function to generate noise
    def noise_like(self, shape, device, repeat=False):
        """
        Generates random noise of a given shape.

        Args:
            shape: Shape of the noise tensor.
            device: Device to create the noise tensor on.
            repeat: Whether to repeat the noise for each batch.

        Returns:
            A noise tensor of the specified shape.
        """
        repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
        noise = lambda: torch.randn(shape, device=device)
        return repeat_noise() if repeat else noise()
    
    # Apply DDIM with self-recurrent guidance
    def ddim_step_self_recurrent_guide(self, sample, noise_pred, indices, score, guide_scale=1.0):
        """
        Performs a single DDIM step with self-recurrent guidance.

        Args:
            sample: Latent sample tensor.
            noise_pred: Predicted noise tensor.
            indices: Indices of the timesteps.
            score: Guidance score tensor.
            guide_scale: Scaling factor for the guidance.

        Returns:
            Updated latent tensor and predicted x_0 tensor.
        """
        b, _, f, *_, device = *sample.shape, sample.device

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        # Select parameters corresponding to the currently considered timestep
        size = (b, 1, 1, 1, 1)
        x_prevs = []
        pred_x0s = []

        for i, index in enumerate(indices):
            x = sample[:, :, [i]]
            e_t = noise_pred[:, :, [i]]
            s_t = score[:, :, [i]]
            a_t = torch.full(size, alphas[index], device=device)
            a_prev = torch.full(size, alphas_prev[index], device=device)
            sigma_t = torch.full(size, sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device)

            # Compute predicted x_0 and adjust noise prediction based on guidance score
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            # Compute the direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

            # Compute the guidance term
            guidance = (1. - a_prev - sigma_t**2).sqrt() * (guide_scale * (1 - a_t).sqrt() * s_t)
            
            # Apply DDIM with self-recurrent guidance as in Eq. 6
            noise = sigma_t * self.noise_like(x.shape, device)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise - guidance

            x_prevs.append(x_prev)
            pred_x0s.append(pred_x0)

        x_prev = torch.cat(x_prevs, dim=2)
        pred_x0 = torch.cat(pred_x0s, dim=2)

        return x_prev, pred_x0