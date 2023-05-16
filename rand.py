import torch
import numpy as np
from robust_speech.adversarial.defenses.smoothing import SpeechNoiseAugmentation
from robust_speech.adversarial.attacks.attacker import Attacker


class GaussianAttack(Attacker):
    def __init__(self, asr_brain, sigma=0, **kwargs):
        self.asr_brain = asr_brain
        self.smoother = SpeechNoiseAugmentation(sigma=sigma)

    def perturb(self, batch):
        wav = self.smoother.forward(batch.sig[0], batch.sig[1])
        return wav
