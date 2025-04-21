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
#from torch.multiprocessing import Process
from multiprocessing import Process
import time


def init_model(config):
    with open(config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # Init asr model from configs
    model = init_asr_model(configs)
    return model
def extract_ppg(file_lst, model,featCal,model_path,gpu_id=None):
    if gpu_id is not None:
        device = torch.device(gpu_id)
    else:
        device = torch.device("cpu")
    featCal = featCal.to(device)
    featCal.eval()
    #import ipdb; ipdb.set_trace()
    #checkpoint = torch.load(model_path)
    checkpoint = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in
            model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()
    #out_dir = '/apdcephfs_cq2/share_1297902/speech_user/nenali/corpus/bbk/wngiga_ppg_utt_center_noise'
    #out_dir = '/apdcephfs_cq2/share_1297902/speech_user/nenali/corpus/bbk/wngiga_ppg_utt_center_aug'
    #out_dir = '/apdcephfs_cq2/share_1297902/speech_user/nenali/corpus/bbk/wngiga_ppg_utt_center'
#    out_dir = '/apdcephfs_cq10/share_1297902/data/speech_data/single_data/tts/AutoPrepARK/ppg/'
    #out_dir = '/apdcephfs_cq10/share_1297902/cq2/speech_user/nenali/corpus/bbk/wngiga_ppg_utt_21pt_lib'
    out_dir = '/apdcephfs_cq10/share_1297902/cq2/speech_train/speech_tts_data/nenali/corpus/bigbbk_data_48k/wngiga_ppg_utt_33pt'
    #out_dir = '/apdcephfs_cq10/share_1297902/cq2/speech_train/speech_tts_data/nenali/corpus/vctk/ppg_33pt'
    #out_dir = '/apdcephfs_cq10/share_1297902/cq2/speech_train/speech_tts_data/nenali/corpus/jili/ppg_21pt'
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for i, wfile in enumerate(file_lst):
            if i % 10 == 0:
                print(f'gpu_{gpu_id}: {i}/{len(file_lst)}')
            #save_path = wfile.replace('/wav_48k/','/ppg/')
            #save_path = wfile.replace('/LibriTTS/','/LibriTTS_ppg/')
            #save_path = wfile.replace('/LibriTTS/','/LibriTTS_logits/')
            #save_path = wfile.replace('/LibriTTS_wav_aug/','/LibriTTS_ppg_aug/')
            #save_path = wfile.replace('/wav_mono_24k_16b_norm-6db/','/wngiga_ppg_utt_dither0/')
            #save_path = wfile.replace('/wav_mono_24k_16b_norm-6db/','/wngiga_ppg_utt_lchunk1_21pt/')
            #save_path = wfile.replace('/wav_mono_24k_16b_norm-6db/','/wngiga_ppg_utt_lchunkv_21pt/')
            #save_path = wfile.replace('/wav_mono_24k_16b_norm-6db/','/wngiga_ppg_utt_lchunkv_small2/')
#            save_name = wfile.replace('/apdcephfs_cq7/share_1297902/common/nenali/wav_zip/',
#                    out_dir)
#            save_file1 =  save_name[:-3] + "npy"
            #import ipdb; ipdb.set_trace()
            #import ipdb; ipdb.set_trace()
            save_name = os.path.basename(wfile)[:-3] + "npy"
            save_file1 = os.path.join(out_dir, save_name)
            #save_path = wfile.replace('/wav_mono_24k_16b_norm-6db/','/wngiga_logit_utt_softmax_21pt/')
            #save_path = wfile.replace('/same_con_diff_spk_utt_conv/','/wngiga_ppg_utt_21pt_aug/')
            #save_file1 = save_path.replace('.wav','.npy')        
#            save_path = wfile.replace('/same_con_diff_spk_utt_conv/','/wngiga_ppg_utt_softmax_21pt_aug/')
#            save_file2 = save_path.replace('.wav','.npy')        
            #save_path = wfile.replace('/wav_mono_24k_16b_norm-6db/','/wngiga_logit_utt_21pt/')
            #save_file2 = save_path.replace('.wav','.npy')        
#            save_path = wfile.replace('/wav_mono_24k_16b_norm-6db/','/wngiga_ppg_utt_21pt/')
#            save_file1 = save_path.replace('.wav','.npy')        
            #save_path = wfile.replace('/wav_mono_24k_16b_norm-6db/','/wngiga_ppg_stream/')
            #save_file3 = save_path.replace('.wav','.npy')        
            if os.path.exists(save_file1) and os.path.getsize(save_file1):
                         continue
            os.makedirs(os.path.dirname(save_file1), exist_ok=True)
#            os.makedirs(os.path.dirname(save_file2), exist_ok=True)
            #try:
            waveform, fs = torchaudio.load(wfile)
            #except:
            #    continue
            if fs != 16000:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=fs, new_freq=16000)(waveform)
                fs = 16000
            waveform = waveform.to(device)
            feats, lengths = featCal(waveform) 
            lengths= lengths.to(device)
            #import ipdb; ipdb.set_trace()
            try:
                ppg, logits = model.extract(feats, lengths, stream=False)#speech: (B, 206, 80)
            except:
                continue    
            ppg = ppg.to('cpu').squeeze().numpy()#1,T,256
#            print(ppg.shape)
            np.save(save_file1,ppg)
            """
            logits_soft = F.softmax(logits,dim=1)
            logits_soft = logits_soft.to('cpu').numpy()
#            print(logits_soft.shape)
            #np.save(save_file1,logits_soft)
            """
            """    
            maxi = np.argmax(logits_soft, axis=1)#(T,)
            save_phn_id = save_file2.replace('.npy','_phn.npy')
            np.save(save_phn_id, maxi)
            """
            """
            logits = logits.to('cpu').numpy()#T,dict_size(601)
#            print(logits.shape)
            np.save(save_file2,logits)
            """
#            if i % 1000 == 0:
#                print(f'gpu_{gpu_id}: {i}/{len(file_lst)}')

######## model with 6 layers #############
#model_path = './exp/small1/26.pt' #29M
#config = './exp/small1/train.yaml'
#model_path = './exp/small2/31.pt'#26M
#config = './exp/small2/train.yaml'
######## model with 6 layers, softmax + center loss #############
#model_path = './exp/center/51.pt'# trained using softmax + center loss, dict:601
#config = './exp/center/train.yaml'
######## model with 7 layers #############
model_path = './exp/stream_wenet_giga/33.pt'#dict:601
config = './exp/stream_wenet_giga/train.yaml'
#model_path = './exp/ft_center/99.pt' # 7 layers, dict:601, ft the above model using center loss
#config = './exp/ft_center/train.yaml'
######## model with 12 layers #############
#model_path = './exp/stream_wenet_giga_layer12/37.pt'
#config = './exp/stream_wenet_giga_layer12/train.yaml'
######## model with 7 layers #############
#model_path = './exp/stream_wenet/51.pt' #dict:357
#config = './exp/stream_wenet/train.yaml'
############################################
#wav_scp_file = './lst/soundlover_48k.scp'
#wav_scp_file = './lst/Yjournal_48k_ori.scp'
#wav_scp_file = './lst/LibriTTS.scp'
#wav_scp_file = './lst/mt_data.scp'
#wav_scp_file = './lst/usm_data.scp'
#wav_scp_file = './lst/mt_mix2.scp'
#wav_scp_file = './lst/bbk_data.scp'
#wav_scp_file = './lst/bbk_data_new3.scp'
#wav_scp_file = './lst/bbk_data_new_189.scp'
#wav_scp_file = './lst/bbk_data_new_aug3.scp'
#wav_scp_file = './lst/wangzhe.scp'
#wav_scp_file = './lst/conv_6spk_aug.scp'
#wav_scp_file = './lst/bbk_data_new_aug_all.scp'
#wav_scp_file = './lst/bbk_data_new_noise_part.scp'
#wav_scp_file = './lst/LibriTTS_wav_aug.scp'
#wav_scp_file = './lst/same_con_diff_spk_utt.scp'
#wav_scp_file = './lst/daji_wav.lst'
#wav_scp_file = './lst/vox2_wav_enh.scp'
#wav_scp_file = './lst/CN-Celeb2_enh.scp'
#wav_scp_file = './lst/CN-Celeb_enh.scp'
#wav_scp_file = './lst/train-clean-360-29spk-300utts.scp'
wav_scp_file = './lst/bbk_data_48k.scp'
#wav_scp_file = './lst/vctk.scp'
#wav_scp_file = './lst/jili.scp'
world_size = 8
model = init_model(config)
#wav_files = [line.strip() for line in open(wav_scp_file,"r")]
wav_files = [line.split()[1] for line in open(wav_scp_file,"r", encoding="utf-8")]
#wav_files = [line.split()[1] for line in open(wav_scp_file,"r",encoding='utf-8')]
wav_file_lsts = np.array_split(wav_files, world_size)
gpu_ids = list(range(world_size))
#gpu_ids = GPUtil.getAvailable(maxMemory=0.02, limit=world_size)
featCal = kaldiFbank()
gpu_id = 7

lsts = np.array_split(wav_file_lsts[7], 8)
extract_ppg(lsts[gpu_id], model, featCal, model_path, gpu_id=gpu_id)
exit()

extract_ppg(wav_file_lsts[gpu_id], model, featCal, model_path, gpu_id=gpu_id)
exit()
t0 = time.time()
processes = []
for rank, gpu_id in enumerate(gpu_ids):
    p = Process(target=extract_ppg, args=(wav_file_lsts[rank], model,featCal,model_path))
    p.start()
    print(f'process {rank} has started')
    processes.append(p)

for p in processes:
    p.join()
print(f'total time is {(time.time() - t0) / 60}')



# torch.manual_seed(666)
# stream = True
# device = torch.device("cpu")
# wfile = '1.wav'



# waveform, fs = torchaudio.load(wfile)
# waveform = waveform * (1 << 15)
# mat = kaldi.fbank(waveform,
#                   num_mel_bins=80,
#                   frame_length=25,
#                   frame_shift=10,
#                   dither=0.1,
#                   energy_floor=0.0,
#                   sample_frequency=fs)#fxdim
# speech_lengths = torch.tensor([mat.shape[0]])
# speech = mat.unsqueeze(0)
# print(speech.shape)




# # print(model)
# num_params = sum(p.numel() for p in model.parameters())
# print('the number of model params: {}'.format(num_params))
# if torch.cuda.is_available():
#     checkpoint = torch.load(model_path)
# else:
#     checkpoint = torch.load(model_path, map_location='cpu')
# model.load_state_dict(checkpoint)

# model = model.to(device)
# ppg, logits = model.extract(speech, speech_lengths, stream=stream)#speech: (B, 206, 80)
# ppg = ppg.detach().numpy()
# #import ipdb; ipdb.set_trace()
# print(ppg.shape)
# exit()
# logits_soft = F.softmax(logits,dim=1)
# logits_soft = logits_soft.detach().numpy()
# print(logits_soft.shape)
# savef = wfile.replace('wav','pos')
# #np.savetxt(savef,logits_soft,fmt='%.6f',newline='\n')

# savef = wfile.replace('wav','phn2pdf')
# maxv = np.max(logits_soft, axis=1)
# #import ipdb; ipdb.set_trace()
# maxi = np.argmax(logits_soft, axis=1)
# with open(savef, 'w') as f:
#     for i, (phn, pdf) in enumerate(zip(maxi.tolist(), maxv.tolist())):
#         f.write(f"phn_id: {phn}, pdf: {pdf:.6f}\n")

