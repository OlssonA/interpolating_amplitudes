import math
import os
import random
import struct
import subprocess

import numpy as np
from numpy.linalg import inv
import scipy.stats.qmc
import multiprocessing


# General functions


def inner_p(p1, p2):
    return (
        p1[:, 0] * p2[:, 0]
        - p1[:, 1] * p2[:, 1]
        - p1[:, 2] * p2[:, 2]
        - p1[:, 3] * p2[:, 3]
    )


def kallen(a, b, c):
    return (a - (np.sqrt(b) + np.sqrt(c)) ** 2) * (a - (np.sqrt(b) - np.sqrt(c)) ** 2)


def angle_between(p1, p2):
    """Calculate the angle between two 4-vectors."""
    dot_product = p1[:, 0] * p2[:, 0] + p1[:, 1] * p2[:, 1] + p1[:, 2] * p2[:, 2]
    norms = np.sqrt(
        p1[:, 0] * p1[:, 0] + p1[:, 1] * p1[:, 1] + p1[:, 2] * p1[:, 2]
    ) * np.sqrt(p2[:, 0] * p2[:, 0] + p2[:, 1] * p2[:, 1] + p2[:, 2] * p2[:, 2])
    cos_theta = dot_product / norms
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return theta


def rotate_func(p1, p2):
    p1_uni = p1 / np.sqrt(np.dot(p1, p1))
    p2_uni = p2 / np.sqrt(np.dot(p2, p2))
    axis = np.cross(p2_uni, p1_uni)
    rot_basis = np.array(
        (
            (0, 0, 0, 0),
            (0, 0, -axis[2], axis[1]),
            (0, axis[2], 0, -axis[0]),
            (0, -axis[1], axis[0], 0),
        )
    )
    cos_theta = np.clip(np.dot(p1_uni, p2_uni), -1.0, 1.0)
    sin_theta = np.sqrt(1 - cos_theta**2)

    if sin_theta == 0:
        rot = np.eye(4)
    elif cos_theta < -0.9999:
        rot = np.array(
            (
                (1, 0, 0, 0),
                (0, -1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, -1),
            )
        )
    else:
        rot = (
            np.eye(4)
            + rot_basis
            + (1 - cos_theta) / (sin_theta**2) * rot_basis @ rot_basis
        )

    return rot, axis, cos_theta, p1_uni, p2_uni


def boost_func(p1, p2):
    beta_cm = (p1[3] + p2[3]) / (p1[0] + p2[0])
    gamma_cm = 1.0 / np.sqrt(1.0 - beta_cm**2)

    boost = np.array(
        (
            (gamma_cm, 0, 0, -gamma_cm * beta_cm),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (-gamma_cm * beta_cm, 0, 0, gamma_cm),
        )
    )

    return boost


# $qq/gg \to t\bat{t}H process$


def x_to_p_tth(data):

    beta2, fracstt, thetah, thetat, phit = (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        data[:, 4],
    )
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

    p = np.concatenate((q1, q2, ph, p1, p2), axis=-1)

    return p


def p_to_x_tth(data):

    p = data.reshape((-1, 5, 4))

    p_tt = p[:, 2] + p[:, 3]
    p_fin = p[:, 2] + p[:, 3] + p[:, 4]

    s_tt = inner_p(p_tt, p_tt)
    s_hat = inner_p(p_fin, p_fin)

    beta2 = 1 - (2 * MT + MH) ** 2 / s_hat
    frac_stt = (s_tt - 4 * MT**2) / ((np.sqrt(s_hat) - MH) ** 2 - 4 * MT**2)

    theta_H = angle_between(p[:, 0, 1:], p[:, 4, 1:])

    qh = np.sqrt(kallen(s_hat, s_tt, MH**2) / 4 / s_hat)

    p1_ini = p[:, 0]
    p2_ini = p[:, 1]
    p1_cm = p[:, 2]
    p2_cm = p[:, 3]
    ph_cm = p[:, 4]

    boost = np.zeros((p.shape[0], 4, 4))
    rot = np.zeros((p.shape[0], 4, 4))

    p1_rot = np.zeros((p.shape[0], 4))
    p2_rot = np.zeros((p.shape[0], 4))
    ptt_rot = np.zeros((p.shape[0], 4))
    p1_boost = np.zeros((p.shape[0], 4))
    p2_boost = np.zeros((p.shape[0], 4))

    for i in range(p.shape[0]):
        rot[i], axis, cos_theta, p1_uni, p2_uni = rotate_func(
            p2_ini[i, 1:], ph_cm[i, 1:]
        )
        p1_rot[i] = rot[i] @ p1_cm[i]
        p2_rot[i] = rot[i] @ p2_cm[i]
        ptt_rot[i] = rot[i] @ (p1_cm[i] + p2_cm[i])

        boost[i] = boost_func(p1_rot[i], p2_rot[i])
        p1_boost[i] = boost[i] @ p1_rot[i]
        p2_boost[i] = boost[i] @ p2_rot[i]

    theta_T = angle_between(p1_boost[:, 1:], ptt_rot[:, 1:])

    cosphit = p1_boost[:, 1] / (np.sin(theta_T) * np.sqrt(s_tt / 4 - MT**2))
    phi_T = np.arccos(np.clip(cosphit, -1.0, 1.0))

    x = np.concatenate((beta2, frac_stt, theta_H, theta_T, phi_T), axis=-1)

    return x


# $gg \to gH process$


def x_to_p_gggh(data):

    mh2 = 125.35**2

    beta2, theta_h = data[:, 0], data[:, 1]
    s = mh2 / (1 - beta2)

    q1 = np.zeros((len(data), 4))
    q2 = np.zeros((len(data), 4))
    p1 = np.zeros((len(data), 4))
    p2 = np.zeros((len(data), 4))

    for i in range(data.shape[0]):
        q1[i] = np.array((1.0, 0.0, 0.0, +1.0)) * np.sqrt(s[i]) / 2
        q2[i] = np.array((1.0, 0.0, 0.0, -1.0)) * np.sqrt(s[i]) / 2
        p1[i] = (
            np.array((1.0, 0.0, -np.sin(theta_h[i]), -np.cos(theta_h[i])))
            * (s[i] - mh2)
            / (2 * np.sqrt(s[i]))
        )
        p2[i] = np.array(
            (
                s[i] + mh2,
                0.0,
                (s[i] - mh2) * np.sin(theta_h[i]),
                (s[i] - mh2) * np.cos(theta_h[i]),
            )
        ) / (2 * np.sqrt(s[i]))

    p = np.concatenate((q1, q2, p1, p2), axis=-1)

    return p


def p_to_x_gggh(data):

    mh2 = 125.35**2

    p = data.reshape((-1, 4, 4))
    s = inner_p(p[:, 0] + p[:, 1], p[:, 0] + p[:, 1])

    beta2 = 1 - 1 / (s - mh2)
    theta = angle_between(p[:, 0, 1:], p[:, 3, 1:])

    x = np.concatenate((beta2, theta), axis=-1)

    return p
