import yaml
from asr_model import init_asr_model
import torch
import torch.nn.functional as F
import os
import re
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import numpy as np
from wenet.dataset.feats import kaldiFbank
import time
import pickle



torch.manual_seed(666)
stream = False
#device = torch.device(0)
device = torch.device("cpu")
wfile = '1.wav'



waveform, fs = torchaudio.load(wfile)#1,tlen
#import ipdb; ipdb.set_trace()
if fs != 16000:
    waveform = torchaudio.transforms.Resample(
        orig_freq=fs, new_freq=16000)(waveform)
waveform = waveform * (1 << 15)
mat = kaldi.fbank(waveform,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0,
                  energy_floor=0.0,
                  sample_frequency=16000)#fxdim
speech_lengths = torch.tensor([mat.shape[0]])
speech = mat.unsqueeze(0)
print(speech.shape)
#speech = speech.repeat(1,3,1)
#print(speech.shape)
#import ipdb; ipdb.set_trace()
######## model with 6 layers #############
#model_path = './exp/small1/26.pt' #29M
#config = './exp/small1/train.yaml'
#model_path = './exp/small2/31.pt'#26M
#config = './exp/small2/train.yaml'
######## model with 7 layers #############
model_path = './exp/stream_wenet_giga/33.pt'
config = './exp/stream_wenet_giga/train.yaml'
#model_path = './exp/ft_center/99.pt' # 7 layers, dict:601, ft the above model using center loss
#config = './exp/ft_center/train.yaml'
######## model with 6 layers, softmax + center loss #############
#model_path = './exp/center/51.pt'# trained using softmax + center loss, dict:601
#config = './exp/center/train.yaml'
######## model with 12 layers #############
#model_path = './exp/stream_wenet_giga_layer12/37.pt'
#config = './exp/stream_wenet_giga_layer12/train.yaml'
with open(config, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)



# Init asr model from configs
model = init_asr_model(configs)
# print(model)
num_params = sum(p.numel() for p in model.parameters())
print('the number of model params: {}'.format(num_params))
if torch.cuda.is_available():
    checkpoint = torch.load(model_path)
else:
    checkpoint = torch.load(model_path, map_location='cpu')
#model.load_state_dict(checkpoint)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() if k in
        model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model = model.to(device)
model.eval()

ce_para = {}
for name, para in model.named_parameters():
    if name.startswith('ce') and name.endswith('weight'):
        ce_para['w'] = para.detach().numpy()
        print(f"{name}: {para.size()}")
    if name.startswith('ce') and name.endswith('bias'):
        ce_para['b'] = para.detach().numpy()
        print(f"{name}: {para.size()}")
f = open('ce_layer.pkl','wb')
pickle.dump(ce_para, f)
f.close()
import ipdb; ipdb.set_trace()
exit()

#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)
with torch.no_grad():
    for i in range(1):
        start = time.time()
        ppg, logits = model.extract(speech.to(device), speech_lengths.to(device), stream=stream)#speech: (B, 206, 80)
        end = time.time()
        print(f"The wav file is {(speech_lengths/100).item()}s, infer time is {end - start}s\n")
exit()
ppg = ppg.to('cpu').detach().squeeze().numpy()
import ipdb; ipdb.set_trace()
np.save('1.npy',ppg)
print(ppg.shape)
exit()
logits_soft = F.softmax(logits,dim=1)
logits_soft = logits_soft.detach().numpy()
print(logits_soft.shape)
savef = wfile.replace('wav','pos')
#np.savetxt(savef,logits_soft,fmt='%.6f',newline='\n')

savef = wfile.replace('wav','phn2pdf')
maxv = np.max(logits_soft, axis=1)
#import ipdb; ipdb.set_trace()
maxi = np.argmax(logits_soft, axis=1)
with open(savef, 'w') as f:
    for i, (phn, pdf) in enumerate(zip(maxi.tolist(), maxv.tolist())):
        f.write(f"phn_id: {phn}, pdf: {pdf:.6f}\n")

