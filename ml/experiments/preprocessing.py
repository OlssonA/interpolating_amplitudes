import numpy as np

def inner_p(p1, p2):
    return np.log(
        p1[:, 0] * p2[:, 0]
        - p1[:, 1] * p2[:, 1]
        - p1[:, 2] * p2[:, 2]
        - p1[:, 3] * p2[:, 3]
    )


def preprocess_particles_mlp(particles_raw, mean, std, eps_std=1e-2):
    p_grouped = np.transpose(particles_raw, (1, 0, 2))
    p_single_array = particles_raw.reshape(particles_raw.shape[0], -1)
    for i in range(p_grouped.shape[0]):
        for j in range(i + 1, p_grouped.shape[0]):
            p_single_array = np.concatenate(
                (p_single_array, inner_p(p_grouped[i], p_grouped[j])[:, None]), axis=1
            )

    for k in range(len(std)):
        if std[k] != 0:
            p_single_array[:,k] = (p_single_array[:,k] - mean[k]) / std[k]

    return p_single_array
    
def preprocess_particles_gatr(particles_raw, std, eps_std=1e-2):
    return particles_raw / std

def preprocess_amplitude_gggh(amplitude, mean=None, std=None):
    log_amplitude = np.log(amplitude)
    if std is None:
        mean = log_amplitude.mean()
        std = log_amplitude.std()
    prepd_amplitude = (log_amplitude - mean) / std
    #assert np.isfinite(prepd_amplitude).all()
    return prepd_amplitude, mean, std

def undo_preprocess_amplitude_gggh(prepd_amplitude, mean, std):
    assert mean is not None and std is not None
    amplitude = prepd_amplitude * std + mean
    return np.exp(prepd_amplitude)
