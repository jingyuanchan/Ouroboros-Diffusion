import torch
import torch.fft as fft
import math

class QueryManager:
    def __init__(self):
        pass

    @staticmethod
    def gaussian_low_pass_filter(shape, d_s=0.25, d_t=1.0):
        """
        Compute the gaussian low pass filter mask.

        Args:
            shape: shape of the filter (volume)
            d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
            d_t: normalized stop frequency for temporal dimension (0.0-1.0)
        """
        T, H, W = shape[-3], shape[-2], shape[-1]
        mask = torch.zeros(shape)
        if d_s == 0 or d_t == 0:
            return mask
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                    mask[..., t, h, w] = math.exp(-1/(2*d_s**2) * d_square)
        return mask
    
    @staticmethod
    def freq_mix_3d(x, noise, LPF):
        """
        Noise reinitialization.

        Args:
            x: diffused latent
            noise: randomly sampled noise
            LPF: low pass filter
        """
        # FFT
        x_freq = fft.fftn(x, dim=(-3, -2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
        noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
        noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

        # Frequency mix
        HPF = 1 - LPF
        x_freq_low = x_freq * LPF
        noise_freq_high = noise_freq * HPF
        x_freq_mixed = x_freq_low + noise_freq_high  # Mix in freq domain

        # IFFT
        x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
        x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real
        
        return x_mixed
    
    @staticmethod
    def shift_w_coherent_tail_sampling(latents, sampler):
        """
        Apply coherent tail sampling to the latents.

        Args:
            latents: tensor of latents
            sampler: the sampler object that contains the necessary attributes
        """
        # Retieve DDPM timestep for the second-to-last latent
        last_timestep = sampler.ddim_timesteps[-2]
        # Get the second to the last latent in previous queue
        sec2last_latent = latents[:, :, [-1]].clone()  # index is -1 since the clear latents has not popped yet
        # Apply DDPM Forward to the second-to-last latent till the last timestep
        for t in range(last_timestep+1, sampler.ddpm_num_timesteps):
            beta_t = sampler.betas[t]
            sec2last_latent = (1 - beta_t)**0.5 * sec2last_latent + beta_t**0.5 * torch.randn_like(sec2last_latent)
        # Shift the latent(same as popping the first latent)
        latents[:, :, :-1] = latents[:, :, 1:].clone()

        # eta_rand is the \eta in Eq. 3
        eta_rand = torch.randn_like(sec2last_latent)
        # freq_filter is the \mathcal{F}^r_low in Eq. 3
        freq_filter = QueryManager.gaussian_low_pass_filter(sec2last_latent.shape)
        # Mix the noise in frequency domain
        latents[:, :, -1:] = QueryManager.freq_mix_3d(sec2last_latent, eta_rand, freq_filter)
        # Return the updated latents
        return latents
