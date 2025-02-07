import math
import numpy as np


def kallen(a, b, c):
    return (a - (np.sqrt(b) + np.sqrt(c)) ** 2) * (a - (np.sqrt(b) - np.sqrt(c)) ** 2)


def par2p(evaluation_point, mt2=1, mh2=12 / 23):
    beta2, fracstt, thetah, thetat, phit = tuple(evaluation_point)
    if fracstt == 0 or fracstt == 1:
        return 0
    costhetah, costhetat, cosphit, sinthetah, sinthetat = (
        np.cos(thetah),
        np.cos(thetat),
        np.cos(phit),
        np.sin(thetah),
        np.sin(thetat),
    )
    sinthetah = np.sqrt(1 - costhetah**2)
    sinthetat = np.sqrt(1 - costhetat**2)
    sinphit = np.sqrt(1 - cosphit**2)
    s = (2 * np.sqrt(mt2) + np.sqrt(mh2)) ** 2 / (1 - beta2)
    stt = fracstt * (np.sqrt(s) - np.sqrt(mh2)) ** 2 + (1 - fracstt) * 4 * mt2
    qh = np.sqrt(kallen(s, stt, mh2) / 4 / s)
    pt = np.sqrt(stt / 4 - mt2)
    Eh = np.sqrt(qh**2 + mh2)
    qt1 = np.array(
        (
            np.sqrt(stt / 4),
            pt * sinthetat * cosphit,
            pt * sinthetat * sinphit,
            pt * costhetat,
        )
    )
    qt2 = np.array(
        (
            np.sqrt(stt / 4),
            -pt * sinthetat * cosphit,
            -pt * sinthetat * sinphit,
            -pt * costhetat,
        )
    )
    invrot = np.array(
        (
            (1, 0, 0, 0),
            (0, -costhetah, 0, -sinthetah),
            (0, 0, 1, 0),
            (0, sinthetah, 0, -costhetah),
        )
    )
    invboost = np.array(
        (
            (np.sqrt(1 + qh**2 / stt), 0, 0, qh / np.sqrt(stt)),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (qh / np.sqrt(stt), 0, 0, np.sqrt(1 + qh**2 / stt)),
        )
    )
    q1 = np.array((1.0, 0.0, 0.0, +1.0)) * np.sqrt(s) / 2
    q2 = np.array((1.0, 0.0, 0.0, -1.0)) * np.sqrt(s) / 2
    ph = np.array((Eh, qh * sinthetah, 0, qh * costhetah))
    p1 = invrot @ invboost @ qt1
    p2 = invrot @ invboost @ qt2

    return q1, q2, p1, p2, ph


if __name__ == "__main__":

    def save(array, path):
        print(f"  {path} ({array.nbytes/1024**3:.2f} GB)")
        array.tofile(path)

    import scipy
    import sys

    sys.path.append("../../")
    import testfunctions

    f1_map = testfunctions.f1_map
    dim = 5
    n_log2 = 10  # 2**n_log2 points
    # Uniform points from a low-discrepancy sobol sequence
    points = np.array(
        scipy.stats.qmc.Sobol(dim, scramble=True, bits=53).random_base2(n_log2)
    )
    amps, weights = f1_map(points)
    amps = np.array(amps)
    amps.tofile("f1-amp.f64")
    points.tofile("f12-pts.f64")

    file_path_amp_train = "f1-amp.f64"
    data_amp = np.fromfile(file_path_amp_train, dtype=np.float64)

    file_path_sm = "f1-amp.f64"
    data_sm = np.fromfile(file_path_sm, dtype=np.float64)

    file_path_p = "f12-pts.f64"
    data = np.fromfile(file_path_p, dtype=np.float64).reshape(-1, 5)

    beta2 = 0.1 + 0.86 * data[:, 0]
    fracstt = data[:, 1]
    theta_H = math.pi * data[:, 2]
    theta_T = math.pi * data[:, 3]
    phi_T = 2 * math.pi * data[:, 4]

    inputs = np.column_stack((beta2, fracstt, theta_H, theta_T, phi_T))

    p = np.array([par2p(inp, mt2=1, mh2=12 / 23) for inp in inputs])
    p = np.reshape(p, (len(data), -1))

    np.save(
        "tth_dataset.npy",
        np.concatenate((p, data_sm[:, None], data_amp[:, None]), axis=-1),
    )
