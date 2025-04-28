import os
import random
import struct
import subprocess
import numpy as np

from . import partransform
# import partransform

class Evaluator:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        self.p = subprocess.Popen([os.path.join(dirname, "gggh-eval"), "-B"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    def __del__(self):
        self.p.kill()
        self.p.wait()
    def __call__(self, beta2, theta_H):
        self.p.stdin.write(struct.pack("dd", beta2, theta_H))
        self.p.stdin.flush()
        result = self.p.stdout.read(8*4)
        return struct.unpack("dddd", result)

class NoisyEvaluator:
    def __init__(self, epsrel=3e-3, seed=None):
        self.ev = Evaluator()
        self.rng = random.Random(seed if seed is not None else os.urandom(8))
        self.epsrel = epsrel
    def __call__(self, beta2, theta_H):
        w, amp1, amp1pole1, amp1pole2 = self.ev(beta2, theta_H)
        amp1      += self.rng.gauss(0.0, abs(     amp1)*self.epsrel)
        amp1pole1 += self.rng.gauss(0.0, abs(amp1pole1)*self.epsrel)
        amp1pole2 += self.rng.gauss(0.0, abs(amp1pole2)*self.epsrel)
        return (w, amp1, amp1pole1, amp1pole2)

def evaluate_many(points):
    dirname = os.path.dirname(os.path.abspath(__file__))
    eval_command = [os.path.join(dirname, "gggh-eval"), "-B"]
    packets = (struct.pack("dd", *point) for point in points)
    return [
        struct.unpack("dddd", packet)
        for packet in partransform.partransform(eval_command, packets, 8*4)
    ]

if __name__ == "__main__":

    ev = Evaluator()

    print("gggh at (β², θₕ,) = (0.1, 0.2):")
    w, amp1, amp1pole, amp1pole2 = ev(0.1, 0.2)
    print("- weight:", w)
    print("- 1loop:", amp1)
    print("- 1loop, 1/ep:", amp1pole)
    print("- 1loop, 1/ep^2:", amp1pole2)

    print("gggh at (β², θₕ,) = (0.2, 0.4):")
    w, amp1, amp1pole, amp1pole2 = ev(0.2, 0.4)
    print("- weight:", w)
    print("- 1loop:", amp1)
    print("- 1loop, 1/ep:", amp1pole)
    print("- 1loop, 1/ep^2:", amp1pole2)
