import torch, torch.nn as nn, random
from torchaudio import transforms
import torchaudio.compliance.kaldi as kaldi

# def augment_spec(feat,is_spec_aug,mask_time,mask_freq):
#     if is_spec_aug:
#         if random.sample(['time','freq'],1)[0]== 'time':
#             param = random.randint(mask_time[0],mask_time[1])
#             masking = transforms.TimeMasking(time_mask_param=param)
#         else:
#             param = random.randint(mask_freq[0],mask_freq[1])
#             masking = transforms.FrequencyMasking(freq_mask_param=4)
#         return masking(feat)
#      else:
#         return feat

class logFbankCal(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=40, spec_mask_time=[5,10], spec_mask_freq=[5,10]):
        super(logFbankCal, self).__init__()
        self.fbankCal = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=n_fft,
                                                  win_length=win_length,
                                                  hop_length=hop_length,
                                                  n_mels=n_mels)

        self.spec_mask_time = spec_mask_time
        self.spec_mask_freq = spec_mask_freq
#         self.dropout_f = SharedDimScaleDropout(alpha=0.1,dim=0)
#         self.dropout_t = SharedDimScaleDropout(alpha=0.1,dim=1)
    def forward(self, x, is_spec_aug=[]):
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        #out = out - out.mean(axis=2).unsqueeze(dim=2)
        
#         for i in range(len(is_spec_aug)):
#             out[i] = augment_spec(out[i],is_spec_aug[i],self.spec_mask_time,self.spec_mask_freq)
        if len(is_spec_aug) != 0:        
            for i in range(len(is_spec_aug)):
                if is_spec_aug[i]:
                    rn = out[i].mean()
                    for n in range(random.randint(2, 5)):
                        offset = random.randint(5, 6)
                        start = random.randrange(0, out.shape[1] - offset)
                        out[i][start : start+offset] = rn    
        out = out.transpose(1,2)  #B,tlen,dim         
        speech_lengths = torch.tensor([out.shape[1]])                           
        return out, speech_lengths

class kaldiFbank(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80, spec_mask_time=[5,10], spec_mask_freq=[5,10]):
        super(kaldiFbank, self).__init__()
        self.f_length = int(win_length/sample_rate*1000) # 25
        self.f_shift = int(hop_length/sample_rate*1000) #10
        self.sample_rate = sample_rate
        self.n_fft =  n_fft
        self.n_mels = n_mels
        self.spec_mask_time = spec_mask_time
        self.spec_mask_freq = spec_mask_freq
    def forward(self, x, is_spec_aug=[]):  
        with torch.no_grad():
            fea = []
            for i in range(x.shape[0]):          
                wavform = x[i] * (1 << 15)
                fea.append(kaldi.fbank(wavform.unsqueeze(0),
                                num_mel_bins=self.n_mels,
                                frame_length=self.f_length,
                                frame_shift=self.f_shift,
                                dither=0.0,
                                energy_floor=0.0,
                                sample_frequency=self.sample_rate))   
        out = torch.stack(fea).transpose(1,2) 
        # out = out - out.mean(axis=2).unsqueeze(dim=2)
        if len(is_spec_aug) != 0:               
            for i in range(len(is_spec_aug)):
                if is_spec_aug[i]:
                    rn = out[i].mean()
                    for n in range(random.randint(2, 5)):
                        offset = random.randint(5, 6)
                        start = random.randrange(0, out.shape[1] - offset)
                        out[i][start : start+offset] = rn
        out = out.transpose(1,2)  #B,tlen,dim         
        speech_lengths = torch.tensor([out.shape[1]])       
        return out, speech_lengths                           
    
class SharedDimScaleDropout(nn.Module):
    def __init__(self, alpha: float = 0.5, dim=1):
        '''
        Continuous scaled dropout that is const over chosen dim (usually across time)
        Multiplies inputs by random mask taken from Uniform([1 - 2\alpha, 1 + 2\alpha])
        '''
        super(SharedDimScaleDropout, self).__init__()
        if alpha > 0.5 or alpha < 0:
            raise ValueError("alpha must be between 0 and 0.5")
        self.alpha = alpha
        self.dim = dim
        self.register_buffer('mask', torch.tensor(0.))
         
    def forward(self, X):
        if self.training:
            if self.alpha != 0.:
                # sample mask from uniform dist with dim of length 1 in self.dim and then repeat to match size
                tied_mask_shape = list(X.shape)
                tied_mask_shape[self.dim] = 1
                repeats = [1 if i != self.dim else X.shape[self.dim] for i in range(len(X.shape))]
                return X * self.mask.repeat(tied_mask_shape).uniform_(1 - 2*self.alpha, 1 + 2*self.alpha).repeat(repeats)
                # expected value of dropout mask is 1 so no need to scale outputs like vanilla dropout
        return X
