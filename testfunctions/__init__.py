import numpy as np
import math
from . import tth
from . import gggh

def pptth_ps_weight(beta2, fstt, theta_h, theta_t, phi_t):
    def kallen(a, sqrt_b, sqrt_c):
        return (a-(sqrt_b+sqrt_c)**2)*(a-(sqrt_b-sqrt_c)**2)
    cos_theta_h = np.cos(theta_h)
    sin_theta_h = np.sin(theta_h)
    cos_theta_t = np.cos(theta_t)
    sin_theta_t = np.sin(theta_t)
    cos_phi_t = np.cos(phi_t)
    sin_phi_t = np.sin(phi_t)
    top_mass2 = 1
    higgs_mass2 = 12/23
    top_mass = 1
    higgs_mass = np.sqrt(higgs_mass2)
    s = (2*top_mass + higgs_mass)**2/(1-beta2)
    sqrt_s = np.sqrt(s)
    stt = 4*top_mass2*(1-fstt) + (sqrt_s - higgs_mass)**2*fstt
    sqrt_stt = np.sqrt(stt)
    kallen_stt_mt2_mt2 = stt*(stt-4*top_mass2)
    kallen_stt_s_mh2 = kallen(stt, sqrt_s, higgs_mass)
    return ((sqrt_s - higgs_mass)**2 - 4*top_mass2)*np.sqrt(kallen_stt_mt2_mt2*kallen_stt_s_mh2)/(1024*np.pi**4*s*stt)*sin_theta_h*sin_theta_t

def gggh_ps_weight(beta2, theta_h):
    sin_theta_h = np.sin(theta_h)
    return sin_theta_h*beta2/(16*np.pi)

def f1_x_to_par(x1, x2, x3, x4, x5):
    return (0.1 + 0.86*x1, x2, np.pi*x3, np.pi*x4, 2*np.pi*x5)

f1_eval = None
def f1(x1, x2, x3, x4, x5, bounds = [0.96, 1, np.pi, np.pi, 2*np.pi]):
    global f1_eval
    if f1_eval is None: f1_eval = tth.Evaluator("qq-0l")
    beta2 = 0.1 + 0.86*x1
    beta2_factor = (1 - 1.0132*beta2)**2/((1 - 0.9943*beta2)*(1 - 0.3506*beta2))
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = f1_eval(beta2, bounds[1]*x2, bounds[2]*x3, bounds[3]*x4, bounds[4]*x5)
    assert math.isfinite(w)
    assert math.isfinite(amp0)
    return amp0, w*beta2_factor

def f1_map(points, bounds = [0.96, 1, np.pi, np.pi, 2*np.pi]):
    points = np.asarray(points)
    x1, x2, x3, x4, x5 = points.T
    beta2 = 0.1 + 0.86*x1
    beta2_factor = (1 - 1.0132*beta2)**2/((1 - 0.9943*beta2)*(1 - 0.3506*beta2))
    raw_points = np.array([beta2, bounds[1]*x2, bounds[2]*x3, bounds[3]*x4, bounds[4]*x5])
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = np.asarray(tth.evaluate_many(raw_points.T, "qq-0l")).T
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(amp0))
    return amp0, w*beta2_factor

def f1_normal(x1, x2, x3, x4, x5):
    if x5 >= 0.5: x5 = 1-x5
    if x4 >= 0.5: x3 = 1-x3; x4 = 1-x4
    if x3 >= 0.5: x3 = 1-x3; x5 = x5+0.5
    if x5 >= 1:   x5 = x5-1
    if x5 >= 0.5: x5 = 1-x5
    return (x1, x2, x3, x4, x5)

def f1_weight(x1, x2, x3, x4, x5):
    beta2, fstt, theta_h, theta_t, phi_t = f1_x_to_par(x1, x2, x3, x4, x5)
    psw = pptth_ps_weight(beta2, fstt, theta_h, theta_t, phi_t)
    beta2_factor = (1 - 1.0132*beta2)**2/((1 - 0.9943*beta2)*(1 - 0.3506*beta2))
    return psw*beta2_factor

f2_eval = None
def f2(x1, x2, x3, x4, x5, bounds = [0.96, 1, np.pi, np.pi, 2*np.pi]):
    global f2_eval
    if f2_eval is None: f2_eval = tth.Evaluator("qq-1l")
    beta2 = 0.1 + 0.86*x1
    beta2_factor = (1 - 1.0132*beta2)**2/((1 - 0.9943*beta2)*(1 - 0.3506*beta2))
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = f2_eval(beta2, bounds[1]*x2, bounds[2]*x3, bounds[3]*x4, bounds[4]*x5)
    assert math.isfinite(w)
    assert math.isfinite(amp0)
    assert math.isfinite(amp1)
    assert math.isfinite(amp1coulomb)
    return amp1-amp1coulomb, w*beta2_factor

def f2_map(points, bounds = [0.96, 1, np.pi, np.pi, 2*np.pi]):
    points = np.asarray(points)
    x1, x2, x3, x4, x5 = points.T
    beta2 = 0.1 + 0.86*x1
    beta2_factor = (1 - 1.0132*beta2)**2/((1 - 0.9943*beta2)*(1 - 0.3506*beta2))
    raw_points = np.array([beta2, bounds[1]*x2, bounds[2]*x3, bounds[3]*x4, bounds[4]*x5])
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = np.asarray(tth.evaluate_many(raw_points.T, "qq-1l")).T
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(amp0))
    assert np.all(np.isfinite(amp1))
    assert np.all(np.isfinite(amp1coulomb))
    return amp1-amp1coulomb, w*beta2_factor

def f2_normal(x1, x2, x3, x4, x5):
    if x4 >= 0.5: x3 = 1-x3; x4 = 1-x4
    if x5 >= 0.5: x5 = 1-x5
    return (x1, x2, x3, x4, x5)

def f2_weight(x1, x2, x3, x4, x5):
    return f1_weight(x1, x2, x3, x4, x5)

def f3_x_to_par(x1, x2, x3, x4, x5):
    return (0.1 + 0.86*x1, x2, np.pi*x3, np.pi*x4, 2*np.pi*x5)

f3_eval = None
def f3(x1, x2, x3, x4, x5, bounds = [0.96, 1, np.pi, np.pi, 2*np.pi]):
    global f3_eval
    if f3_eval is None: f3_eval = tth.Evaluator("gg-0l")
    beta2 = 0.1+0.86*x1
    beta2_factor = (1 - 1.0134*beta2)**2/((1 - 0.7344*beta2)*(1 - 0.0987*beta2))
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = f3_eval(beta2, bounds[1]*x2, bounds[2]*x3, bounds[3]*x4, bounds[4]*x5)
    assert math.isfinite(w)
    assert math.isfinite(amp0)
    return amp0, w*beta2_factor

def f3_map(points, bounds = [0.96, 1, np.pi, np.pi, 2*np.pi]):
    points = np.asarray(points)
    x1, x2, x3, x4, x5 = points.T
    beta2 = 0.1 + 0.86*x1
    beta2_factor = (1 - 1.0134*beta2)**2/((1 - 0.7344*beta2)*(1 - 0.0987*beta2))
    raw_points = np.array([beta2, bounds[1]*x2, bounds[2]*x3, bounds[3]*x4, bounds[4]*x5])
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = np.asarray(tth.evaluate_many(raw_points.T, "gg-0l")).T
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(amp0))
    return amp0, w*beta2_factor

def f3_normal(x1, x2, x3, x4, x5):
    return f1_normal(x1, x2, x3, x4, x5)

def f3_weight(x1, x2, x3, x4, x5):
    beta2, fstt, theta_h, theta_t, phi_t = f3_x_to_par(x1, x2, x3, x4, x5)
    psw = pptth_ps_weight(beta2, fstt, theta_h, theta_t, phi_t)
    beta2_factor = (1 - 1.0134*beta2)**2/((1 - 0.7344*beta2)*(1 - 0.0987*beta2))
    return psw*beta2_factor

f4_eval = None
def f4(x1, x2, x3, x4, x5, bounds = [0.96, 1, np.pi, np.pi, 2*np.pi]):
    global f4_eval
    if f4_eval is None: f4_eval = tth.Evaluator("gg-1l")
    beta2 = 0.1+0.86*x1
    beta2_factor = (1 - 1.0134*beta2)**2/((1 - 0.7344*beta2)*(1 - 0.0987*beta2))
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = f4_eval(beta2, bounds[1]*x2, bounds[2]*x3, bounds[3]*x4, bounds[4]*x5)
    assert math.isfinite(w)
    assert math.isfinite(amp0)
    assert math.isfinite(amp1)
    assert math.isfinite(amp1coulomb)
    return amp1-amp1coulomb, w*beta2_factor

def f4_map(points, bounds = [0.96, 1, np.pi, np.pi, 2*np.pi]):
    points = np.asarray(points)
    x1, x2, x3, x4, x5 = points.T
    beta2 = 0.1 + 0.86*x1
    beta2_factor = (1 - 1.0134*beta2)**2/((1 - 0.7344*beta2)*(1 - 0.0987*beta2))
    raw_points = np.array([beta2, bounds[1]*x2, bounds[2]*x3, bounds[3]*x4, bounds[4]*x5])
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = np.asarray(tth.evaluate_many(raw_points.T, "gg-1l")).T
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(amp0))
    assert np.all(np.isfinite(amp1))
    assert np.all(np.isfinite(amp1coulomb))
    return amp1-amp1coulomb, w*beta2_factor

def f4_normal(x1, x2, x3, x4, x5):
    return f1_normal(x1, x2, x3, x4, x5)

def f4_weight(x1, x2, x3, x4, x5):
    return f3_weight(x1, x2, x3, x4, x5)

f5_eval = None
def f5(x1, x2, bounds = [0.99, np.pi]): #keep bounds at exactly this for now
    global f5_eval
    if f5_eval is None: f5_eval = gggh.Evaluator()
    min_beta2, max_beta2 = 0.33, bounds[0]
    beta2 = min_beta2 + (max_beta2 - min_beta2)*x1
    pt_over_mh = 1/(2*math.sqrt(1 - min_beta2)/min_beta2)
    theta_0 = math.asin(pt_over_mh*(2*math.sqrt(1-beta2)/beta2))
    if bounds[1] == np.pi:
        theta_H = theta_0 + (bounds[1] - 2*theta_0)*x2
    elif bounds[1] == np.pi/2:
        theta_H = theta_0 + (bounds[1] - theta_0)*x2
    beta2_factor = (1 - 1.0012*beta2)**2/((1 - 0.9802*beta2)*(1 - 0.3357*beta2))
    if bounds[1] == np.pi:
        J_factor = bounds[1] - 2*theta_0
    elif bounds[1] == np.pi/2:
        J_factor = bounds[1] - theta_0
    w, amp1, amp1pole, amp1pole2 = f5_eval(beta2, theta_H)
    assert math.isfinite(w)
    assert math.isfinite(amp1)
    return amp1, w*beta2_factor*J_factor

f5_eval = None
def f5_sin2(x1, x2, bounds = [0.99, np.pi]): #keep bounds at exactly this for now
    global f5_eval
    if f5_eval is None: f5_eval = gggh.Evaluator()
    min_beta2, max_beta2 = 0.33, bounds[0]
    beta2 = min_beta2 + (max_beta2 - min_beta2)*x1
    pt_over_mh = 1/(2*math.sqrt(1 - min_beta2)/min_beta2)
    theta_0 = math.asin(pt_over_mh*(2*math.sqrt(1-beta2)/beta2))
    if bounds[1] == np.pi:
        theta_H = theta_0 + (bounds[1] - 2*theta_0)*x2
    elif bounds[1] == np.pi/2:
        theta_H = theta_0 + (bounds[1] - theta_0)*x2
    beta2_factor = (1 - 1.0012*beta2)**2/((1 - 0.9802*beta2)*(1 - 0.3357*beta2))
    if bounds[1] == np.pi:
        J_factor = bounds[1] - 2*theta_0
    elif bounds[1] == np.pi/2:
        J_factor = bounds[1] - theta_0
    w, amp1, amp1pole, amp1pole2 = f5_eval(beta2, theta_H)
    assert math.isfinite(w)
    assert math.isfinite(amp1)
    return amp1, w*beta2_factor*J_factor*np.sin(theta_H)**2

def f5_map(points, bounds = [0.99, np.pi]):
    points = np.asarray(points)
    x1, x2 = points.T
    min_beta2, max_beta2 = 0.33, bounds[0]
    beta2 = min_beta2 + (max_beta2 - min_beta2)*x1
    pt_over_mh = 1/(2*math.sqrt(1 - min_beta2)/min_beta2)
    theta_0 = np.arcsin(pt_over_mh*(2*np.sqrt(1-beta2)/beta2))
    if bounds[1] == np.pi:
        theta_H = theta_0 + (bounds[1] - 2*theta_0)*x2
    elif bounds[1] == np.pi/2:
        theta_H = theta_0 + (bounds[1] - theta_0)*x2
    beta2_factor = (1 - 1.0012*beta2)**2/((1 - 0.9802*beta2)*(1 - 0.3357*beta2))
    if bounds[1] == np.pi:
        J_factor = bounds[1] - 2*theta_0
    elif bounds[1] == np.pi/2:
        J_factor = bounds[1] - theta_0
    raw_points = np.array([beta2, theta_H])
    w, amp1, amp1pole, amp1pole2 = np.asarray(gggh.evaluate_many(raw_points.T)).T
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(amp1))
    return amp1, w*beta2_factor*J_factor

def f5_normal(x1, x2):
    if x2 >= 0.5: x2 = 1-x2
    return (x1, x2)

def f5_weight(x1, x2):
    min_beta2, max_beta2 = 0.33, 0.99
    beta2 = min_beta2 + (max_beta2 - min_beta2)*x1
    pt_over_mh = 1/(2*math.sqrt(1 - min_beta2)/min_beta2)
    theta_0 = math.asin(pt_over_mh*(2*math.sqrt(1-beta2)/beta2))
    theta_H = theta_0 + (math.pi - 2*theta_0)*x2
    psw = gggh_ps_weight(beta2, theta_H)
    beta2_factor = (1 - 1.0012*beta2)**2/((1 - 0.9802*beta2)*(1 - 0.3357*beta2))
    J_factor = math.pi - 2*theta_0
    return psw*beta2_factor*J_factor
