import torch
import tqdm

from modules.encoder import SvcEncoder
from utils.hparams import hparams

class RectifiedFlow(torch.nn.Module):
    def __init__(self, phone_encoder, out_dims, denoise_fn, spec_min, spec_max, timesteps=hparams.get("timesteps", 1000), loss_type=hparams.get("diff_loss_type", "l2")):
        super().__init__()
        self.velocity_function = denoise_fn
        self.fs2 = SvcEncoder(phone_encoder, out_dims)
        self.mel_bins = out_dims
        self.loss_type = loss_type

        self.timesteps = timesteps

        self.register_buffer("spec_min", torch.FloatTensor(spec_min)[None, None, :hparams["keep_bins"]])
        self.register_buffer("spec_max", torch.FloatTensor(spec_max)[None, None, :hparams["keep_bins"]])
    
    def loss(self, x_1, t, condition):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_prediction = self.velocity_function(x_t, self.timesteps * t, condition)

        target_velocity = x_1 - x_0

        if self.loss_type == "l1":
            loss = (target_velocity - v_prediction).abs().mean()
        elif self.loss_type == "l2":
            loss = torch.nn.functional.mse_loss(target_velocity, v_prediction)
        else:
            raise NotImplementedError(f"Loss type '{self.loss_type}' is not implemented.")
        
        return loss

    @torch.no_grad()
    def sample_euler(self, x, t, dt, condition):
        x += self.velocity_function(x, self.timesteps * t, condition) * dt
        t += dt

        return x, t
    
    @torch.no_grad()
    def sample_heun(self, x, t, dt, condition=None):
        k_1 = self.velocity_function(x, self.timesteps * t, condition)
        x_pred = x + k_1 * dt
        t_pred = t + dt
        k_2 = self.velocity_function(x_pred, self.timesteps * t_pred, condition)
        x += (k_1 + k_2) / 2 * dt
        t += dt

        return x, t
    
    @torch.no_grad()
    def sample_rk2(self, x, t, dt, condition):
        k_1 = self.velocity_function(x, self.timesteps * t, condition)
        k_2 = self.velocity_function(x + 0.5 * k_1 * dt, self.timesteps * (t + 0.5 * dt), condition)
        x += k_2 * dt
        t += dt

        return x, t
    
    @torch.no_grad()
    def sample_rk4(self, x, t, dt, condition):
        k_1 = self.velocity_function(x, self.timesteps * t, condition)
        k_2 = self.velocity_function(x + 0.5 * k_1 * dt, self.timesteps * (t + 0.5 * dt), condition)
        k_3 = self.velocity_function(x + 0.5 * k_2 * dt, self.timesteps * (t + 0.5 * dt), condition)
        k_4 = self.velocity_function(x + k_3 * dt, self.timesteps * (t + dt), condition)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt

        return x, t
    
    @torch.no_grad()
    def sample_rk5(self, x, t, dt, condition):
        k_1 = self.velocity_function(x, self.timesteps * t, condition)
        k_2 = self.velocity_function(x + 0.25 * k_1 * dt, self.timesteps * (t + 0.25 * dt), condition)
        k_3 = self.velocity_function(x + 0.125 * (k_2 + k_1) * dt, self.timesteps * (t + 0.25 * dt), condition)
        k_4 = self.velocity_function(x + 0.5 * (-k_2 + 2 * k_3) * dt, self.timesteps * (t + 0.5 * dt), condition)
        k_5 = self.velocity_function(x + 0.0625 * (3 * k_1 + 9 * k_4) * dt, self.timesteps * (t + 0.75 * dt), condition)
        k_6 = self.velocity_function(x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt / 7, self.timesteps * (t + dt), condition)
        
        x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt / 90
        t += dt

        return x, t
    
    def forward(self, hubert, mel2ph=None, spk_embed_id=None, ref_mels=None, f0=None, energy=None, aux_mel_pred=None, infer=False):
        ret = self.fs2(hubert, mel2ph, spk_embed_id, f0, energy)
        condition = ret["decoder_inp"].transpose(1, 2)
        b = hubert.shape[0]
        device = hubert.device

        if not infer:
            x_1 = self.normalize_spectrogram(ref_mels)
            x_1 = x_1.transpose(1, 2)[:, None, :, :]

            T_start = hparams.get("T_start", 0.4)
            random_T = torch.rand(b, device=device, dtype=torch.float32)
            t = random_T * (1.0 - T_start) + T_start
            t = torch.clamp(t, 1e-7, 1.0 - 1e-7)

            loss = self.loss(x_1, t, condition)
            ret["diff_loss"] = loss

            return ret
        else:
            K_step = hparams.get("K_step_infer", 20)
            T_start = hparams.get("T_start_infer", 0.4)
            sampling_algorithm = hparams.get("sampling_algorithm", "rk4")

            try:
                sampler_function = getattr(self, f"sample_{sampling_algorithm}")
            except AttributeError:
                raise NotImplementedError(f"Sampling algorithm '{sampling_algorithm}' not found.")
            
            shape = (condition.shape[0], 1, self.mel_bins, condition.shape[2])

            if T_start > 0.0 and ref_mels is not None:
                x_1 = self.normalize_spectrogram(ref_mels)
                x_1 = x_1.transpose(1, 2)[:, None, :, :]

                x_0 = torch.randn(shape, device=device)

                x = x_0 + T_start * (x_1 - x_0)
                t = torch.full((b,), T_start, device=device, dtype=torch.float32)
                dt = (1.0 - T_start) / K_step

                description = "Sample time step (ReFlow Shallow)"
            else:
                x = torch.randn(shape, device=device)
                t = torch.zeros((b,), device=device, dtype=torch.float32)
                dt = 1.0 / K_step

                description = "Sample time step (ReFlow)"
            
            for _ in tqdm.tqdm(range(K_step), desc=description + f" ({sampling_algorithm})", total=K_step):
                x, t = sampler_function(x, t, dt, condition)
            
            x = x.squeeze(1).transpose(1, 2)

            if mel2ph is not None:
                ret["mel_out"] = self.denormalize_spectrogram(x) * ((mel2ph > 0).float()[:, :, None])
            else:
                ret["mel_out"] = self.denormalize_spectrogram(x)
            
            return ret

    def normalize_spectrogram(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1
    
    def denormalize_spectrogram(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min