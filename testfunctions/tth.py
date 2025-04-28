import math
import os
import random
import struct
import subprocess

from . import partransform

class Evaluator:
    def __init__(self, ampname):
        assert ampname in ("qq-0l", "qq-1l", "gg-0l", "gg-1l")
        dirname = os.path.dirname(os.path.abspath(__file__))
        self.p = subprocess.Popen([os.path.join(dirname, "tth-eval"), "-B", ampname],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    def __del__(self):
        self.p.kill()
        self.p.wait()
    def __call__(self, beta2, fracstt, theta_h, theta_t, phi_t):
        self.p.stdin.write(struct.pack("ddddd", beta2, fracstt, theta_h, theta_t, phi_t))
        self.p.stdin.flush()
        result = self.p.stdout.read(8*6)
        return struct.unpack("dddddd", result)

class NoisyEvaluator:
    def __init__(self, ampname, epsrel=3e-3, seed=None):
        self.ev = Evaluator(ampname)
        self.rng = random.Random(seed if seed is not None else os.urandom(8))
        self.epsrel = epsrel
    def __call__(self, beta2, fracstt, theta_h, theta_t, phi_t):
        w, amp0, amp1, amp1pole1, amp1pole2, amp1coulomb = self.ev(beta2, fracstt, theta_h, theta_t, phi_t)
        amp0      += self.rng.gauss(0.0, abs(     amp0)*self.epsrel)
        amp1      += self.rng.gauss(0.0, abs(     amp1)*self.epsrel)
        amp1pole1 += self.rng.gauss(0.0, abs(amp1pole1)*self.epsrel)
        amp1pole2 += self.rng.gauss(0.0, abs(amp1pole2)*self.epsrel)
        return (w, amp0, amp1, amp1pole1, amp1pole2)

def evaluate_many(points, ampname):
    dirname = os.path.dirname(os.path.abspath(__file__))
    eval_command = [os.path.join(dirname, "tth-eval"), "-B", ampname]
    packets = (struct.pack("ddddd", *point) for point in points)
    return [
        struct.unpack("dddddd", packet)
        for packet in partransform.partransform(eval_command, packets, 8*6)
    ]

def rescale(x):
    import numpy as np
    parametrisation_bounds = [[0.1, 0.95], [0,1], [0, np.pi], [0, np.pi], [0, 2*np.pi]]
    return [(parametrisation_bounds[jj][1]-parametrisation_bounds[jj][0])*\
            point + parametrisation_bounds[jj][0] 
            for jj, point in enumerate(x)]

if __name__ == "__main__":
    ev = Evaluator("gg-1l")
    def gg_1L(x): return ev(*tuple(rescale(x)))
    
    print(gg_1L([0.7183125, 0.3, 0.2, 0.49999, 0.1])[2])
    print(gg_1L([0.7183125, 0.3, 0.2, 0.50000, 0.1])[2])
    print(gg_1L([0.7183125, 0.3, 0.2, 0.50001, 0.1])[2])

    exit()

    

    print("qq-1l at (β², fₛₜₜ, θₕ, θₜ, φₜ) = (0.1, 0.2, 0.3, 0.4, 0.5):")
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = ev(0.1, 0.2, 0.3, 0.4, 0.5)
    print("- weight:", w)
    print("- tree:", amp0)
    print("- 1loop:", amp1)
    print("- 1loop, 1/ep:", amp1pole)
    print("- 1loop, 1/ep^2:", amp1pole2)
    print("- 1loop, Columb:", amp1coulomb)
    print("- 1loop, w/o Coulomb:", amp1 - amp1coulomb)

    print("qq-1l at (β², fₛₜₜ, θₕ, θₜ, φₜ) = (0.2, 0.4, 0.8, 1.6, 3.2):")
    w, amp0, amp1, amp1pole, amp1pole2, amp1coulomb = ev(0.2, 0.4, 0.8, 1.6, 3.2)
    print("- weight:", w)
    print("- tree:", amp0)
    print("- 1loop:", amp1)
    print("- 1loop, 1/ep:", amp1pole)
    print("- 1loop, 1/ep^2:", amp1pole2)
    print("- 1loop, Columb:", amp1coulomb)
    print("- 1loop, w/o Coulomb:", amp1 - amp1coulomb)
