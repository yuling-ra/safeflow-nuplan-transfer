import torch, numpy as np
import torchdiffeq

def sample_full_vector(model, cond_np, y_scaler, cfg, device):
    model.eval()
    with torch.no_grad():
        cond_norm = cond_np.astype(np.float32)
        cond = torch.from_numpy(cond_norm).unsqueeze(0).to(device)  # (1,cond_dim)
        D = y_scaler.min_.shape[0]
        x0 = torch.randn(1, D, device=device)

        def ode_fun(t, x_flat):
            x = x_flat.view(1, D)
            t_tensor = torch.tensor([t.item()], device=device, dtype=torch.float32)
            v = model(x, t_tensor, condition=cond)
            return v.view(-1)

        t_span = torch.tensor([0., 1.], device=device)
        res = torchdiffeq.odeint(ode_fun, x0.view(-1), t_span,
                                 atol=cfg.infer.ode.atol, rtol=cfg.infer.ode.rtol,
                                 method=cfg.infer.ode.method)
        y_hat = res[-1].view(D).cpu().numpy()
        return y_hat

def split_full_vector(y_vec, data_spec):
    s = data_spec["slices"]
    ta, ts = data_spec["T_actions"], data_spec["T_states"]
    ad, sd = data_spec["action_dim"], data_spec["state_dim"]
    a0, a1 = s["actions"]
    s0, s1 = s["states"]
    actions = y_vec[a0:a1].reshape(ta, ad)
    states  = y_vec[s0:s1].reshape(ts, sd)
    return actions, states
