from robust_speech.adversarial.attacks.pgd import SNRPGDAttack,ASRLinfPGDAttack
from sb_whisper_binding import WhisperASR
from whisper_with_gradients import detect_language_with_gradients
from whisper.audio import CHUNK_LENGTH, SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
import robust_speech as rs
import torch.nn as nn
import torch
from robust_speech.adversarial.utils import TargetGenerator
import copy 
from robust_speech.adversarial.attacks.attacker import TrainableAttacker
from lang_attack import WhisperLangID
from robust_speech.adversarial.utils import (
    l2_clamp_or_normalize,
    linf_clamp,
    rand_assign,
)

from tqdm import tqdm

MAXLEN=16000*30
class UniversalWhisperLanguageAttack(TrainableAttacker,ASRLinfPGDAttack):
    def __init__(self,asr_brain,*args,language="es",targeted_for_language=True,nb_epochs=10,eps_item=0.001,success_every=10,univ_perturb=None,epoch_counter=None,**kwargs):
        self.language = "<|"+language.strip("<|>")+"|>"
        self.lang_token = torch.LongTensor(asr_brain.tokenizer.encode(self.language)).to(asr_brain.device)
        ASRLinfPGDAttack.__init__(self,WhisperLangID(asr_brain,self.lang_token),*args,targeted=targeted_for_language,**kwargs)
        self.univ_perturb = univ_perturb
        if self.univ_perturb is None:
            self.univ_perturb = rs.adversarial.utils.TensorModule(size=(MAXLEN,))
        self.nb_epochs=nb_epochs
        self.eps_item=eps_item
        self.success_every=success_every
        self.epoch_counter = epoch_counter
        if self.epoch_counter is None:
            self.epoch_counter = range(100)

    def fit(self, loader):
        return self._compute_universal_perturbation(loader)

    def _compute_universal_perturbation(self, loader):
        self.univ_perturb=self.univ_perturb.to(self.asr_brain.device)
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()

        delta = self.univ_perturb.tensor.data
        success_rate = 0

        best_success_rate = -100
        for epoch in self.epoch_counter:
            # print(f'{epoch}s epoch')
            # epoch += 1
            # GENERATE CANDIDATE FOR UNIVERSAL PERTURBATION
            for idx, batch in enumerate(loader):
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig
                # Slice or Pad to match the shape with data point x
                if wav_init.shape[1] <= delta.shape[0]:
                    begin = torch.randint(delta.shape[0]-wav_init.shape[1]-1,size=(1,))
                    delta_x = delta[begin:begin+wav_init.shape[1]].detach()
                    #delta_x[:wav_init.shape[1]] = delta[: wav_init.shape[1]].detach()
                else:
                    delta_x = torch.zeros_like(wav_init[0])
                    delta_x[: delta.shape[0]] = delta.detach()
                delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())

                #r = (torch.rand_like(delta_x)-0.5) *self.rel_eps_iter * self.eps_item*0.1
                r = torch.zeros_like(delta_x)
                r.requires_grad_()


                for i in range(self.nb_iter):
                    r_batch = r.unsqueeze(0).expand(delta_batch.size())

                    batch.sig = wav_init + delta_batch + r_batch, wav_lens
                    predictions = self.asr_brain.compute_forward(
                        batch, rs.Stage.ATTACK)
                    # loss = 0.5 * r.norm(dim=1, p=2) - self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK)
                    lang_loss = self.asr_brain.compute_objectives(
                            predictions, batch, rs.Stage.ATTACK)
                    l2_norm = r.norm(p=2).to(
                        self.asr_brain.device)
                    # print(idx,lang_loss)
                    if lang_loss.max()<0.1:
                        break
                    loss = 0.5 * l2_norm + lang_loss
                    loss.backward()
                    # print(l2_norm,ctc,CER)
                    grad_sign = r.grad.data.sign()
                    r.data = r.data - self.rel_eps_iter * self.eps_item * grad_sign
                    # r.data = r.data - 0.1 * r.grad.data
                    r.data = linf_clamp(r.data, self.eps_item)
                    r.data = linf_clamp(
                        delta_x + r.data, self.eps) - delta_x

                    # print("delta's mean : ", torch.mean(delta_x).data)
                    # print("r's mean : ",torch.mean(r).data)
                    r.grad.data.zero_()

                delta_x = linf_clamp(delta_x + r.data, self.eps)

                if delta.shape[0] <= delta_x.shape[0]:
                    delta = delta_x[:delta.shape[0]].detach()
                else:
                    delta[begin:begin+wav_init.shape[1]] = delta_x.detach()
                    #delta[:delta_x.shape[0]] = delta_x.detach()

            # print(f'MAX OF INPUT WAVE IS {torch.max(wav_init).data}')
            # print(f'AVG OF INPUT WAVE IS {torch.mean(wav_init).data}')
            # print(f'MAX OF DELTA IS {torch.max(delta).data}')
            # print(f'AVG OF DELTA IS {torch.mean(delta).data}')
            if (epoch % self.success_every)==0:
                print(f'Check success rate after epoch {epoch}')
                # TO CHECK SUCCESS RATE OVER ALL TRAINING SAMPLES
                total_sample = 0.
                fooled_sample = 0.
                loss=0

                for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
                    batch = batch.to(self.asr_brain.device)
                    wav_init, wav_lens = batch.sig

                    delta_x = torch.zeros_like(wav_init[0])
                    if wav_init.shape[1] <= delta.shape[0]:
                        begin = torch.randint(delta.shape[0]-wav_init.shape[1]-1,size=(1,))
                        delta_x = delta[begin:begin+wav_init.shape[1]]
                        #delta_x = delta[:wav_init.shape[1]]
                    else:
                        delta_x[:delta.shape[0]] = delta
                    delta_batch = delta_x.unsqueeze(0).expand(wav_init.size()).to(self.asr_brain.device)
                    batch.sig = wav_init + delta_batch, wav_lens
                    predictions = self.asr_brain.compute_forward(
                            batch, rs.Stage.ATTACK)
                    language_tokens_pred,_,_ = predictions
                    total_sample+=batch.batchsize
                    fooled_sample+= (language_tokens_pred==self.lang_token).sum()
                    loss += self.asr_brain.compute_objectives(
                            predictions, batch, rs.Stage.ATTACK).item()

                success_rate = fooled_sample/total_sample
                print(f'SUCCESS RATE IS {success_rate:.4f}')
                print(f'LOSS IS {(loss/(idx+1)):.4f}')
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    self.univ_perturb.tensor.data = delta.detach()
                    self.checkpointer.save_and_keep_only()
                    #print(
                    #    f"Perturbation vector with best success rate saved. Success rate:{(100*best_success_rate):.2f}%")
        print(
            f"Training finisihed. Best success rate: {best_success_rate:.2f}%") 

    def perturb(self, batch):
        """
        Compute an adversarial perturbation
        Arguments
        ---------
        batch : sb.PaddedBatch
           The input batch to perturb

        Returns
        -------
        the tensor of the perturbed batch
        """
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()

        save_device = batch.sig[0].device
        batch = batch.to(self.asr_brain.device)
        save_input = batch.sig[0]
        wav_init = torch.clone(save_input)

        delta = self.univ_perturb.tensor.data.to(self.asr_brain.device)

        if wav_init.shape[1] <= delta.shape[0]:
            delta_x = delta[:wav_init.shape[1]]
        else:
            delta_x = torch.zeros_like(wav_init[0])
            delta[:delta.shape[0]] = delta
        delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())
        wav_adv = wav_init + delta_batch
        # self.eps = 1.0
        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()
        return wav_adv.data.to(save_device)

