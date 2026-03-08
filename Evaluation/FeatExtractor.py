import numpy as np
import os
import librosa
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io.wavfile as wavfile
import pickle
import model_det
import model_effect
import panns_models

class EmbedExtractor:
    def __init__(self, config=None):
        self.config = config
        self.get_vggish_model()
        self.get_panns_model()

    def get_vggish_model(self):
        use_pca = self.config['vggish_config']['use_pca']
        use_activation = self.config['vggish_config']['use_activation']
        model = torch.hub.load("harritaylor/torchvggish", "vggish")
        if use_pca:
            model.postprocess = False
        if not use_activation:
            model.embeddings = nn.Sequential(
                *list(model.embeddings.children())[:-1])
        model.postprocess = False
        model.embeddings = nn.Sequential(
            *list(model.embeddings.children())[:-1])
        model.eval()

        self.vggish_model = model

    def get_panns_model(self):
        '''use the panns pretrained audiotagging model'''
        sample_rate = self.config['panns_config']['sample_rate']
        window_size = self.config['panns_config']['window_size']
        hop_size = self.config['panns_config']['hop_size']
        mel_bins = self.config['panns_config']['mel_bins']
        fmin = self.config['panns_config']['fmin']
        fmax = self.config['panns_config']['fmax']
        model = panns_models.Cnn14(sample_rate=sample_rate,
                                  window_size=window_size,
                                  hop_size=hop_size,
                                  mel_bins=mel_bins,
                                  fmin=fmin,
                                  fmax=fmax,
                                  classes_num=527)
        checkpoint_path = self.config['model_path']['pretrained_panns_path']
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        self.panns_model = model

    def get_vggish_embed(self, audio_filename_list):
        for audio_id, audio_filename in enumerate(audio_filename_list):
            if audio_id % 100 == 0:
                print(f'Processing {audio_id}/{len(audio_filename_list)}')
            (waveform, _) = librosa.core.load(
                audio_filename, sr=16000, mono=True)

            embed = self.vggish_model.forward(waveform, 16000)
            embed = embed.cpu().detach().numpy().squeeze()
            embed_filename = audio_filename.replace('.wav', '_vggish_embed.npy')
            np.save(embed_filename, embed)

    def get_panns_embed(self, audio_filename_list):
        for audio_id, audio_filename in enumerate(audio_filename_list):
            if audio_id % 100 == 0:
                print(f'Processing {audio_id}/{len(audio_filename_list)}')
            (waveform, _) = librosa.core.load(
                audio_filename, sr=16000, mono=True)
            device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
            waveform = torch.from_numpy(waveform).to(torch.float32).to(device)
            waveform = waveform.unsqueeze(0)

            output_dict = self.panns_model(waveform, None)
            embed = output_dict['embedding'].detach().cpu().numpy().squeeze()

            embed_filename = audio_filename.replace('.wav', '_panns_embed.npy')
            np.save(embed_filename, embed)

    def get_embedding(self, audio_dir, embed_type=['vggish', 'panns']):
        audio_filename_list = glob.glob(os.path.join(audio_dir, '*.wav'))
        if 'vggish' in embed_type:
            self.get_vggish_embed(audio_filename_list)
        if 'panns' in embed_type:
            self.get_panns_embed(audio_filename_list)

        print('Done')
        
class SEDFeatExtractor:
    def __init__(self, config=None, device=None):
        self.config = config
        self.device = device
        self.launch_det_model()

    def launch_det_model(self):
        model = model_det.Cnn14_DecisionLevelMax(sample_rate=16000,
                                    window_size=1024, 
                                    hop_size=320, 
                                    mel_bins=64, 
                                    fmin=50,
                                    fmax=8000,
                                    det_num=110)

        pretrained_model = self.config['model_path']['det110_model_path']
        checkpoint = torch.load(pretrained_model, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model = model.to(device=self.device)
        model = model.eval()       
        self.sed_model = model

    def get_det_score(self, audio_dir):
        audio_filename_list = glob.glob(os.path.join(audio_dir, '*.wav'))
        for audio_id, audio_filename in enumerate(audio_filename_list):
            if audio_id % 100 == 0:
                print(f'Processing {audio_id}/{len(audio_filename_list)}')
            audio_data = wavfile.read(audio_filename)[1].astype(np.float32) / 32768.0
            audio_data_points = self.config['sample_rate'] * self.config['audio_len_sec']
            if audio_data.shape[0] < audio_data_points:
                audio_data = np.pad(audio_data, (0, audio_data_points - audio_data.shape[0]), 'constant')
            elif audio_data.shape[0] > audio_data_points:   
                audio_data = audio_data[:audio_data_points]
            
            audio_data = np.expand_dims(audio_data, axis=0)
            audio_data = torch.from_numpy(audio_data).to(torch.float32).to(self.device)
            with torch.no_grad():
                output_dict = self.sed_model(audio_data)
            
            det_score = F.sigmoid(torch.squeeze(output_dict['segmentwise_det_logits'])).cpu().detach().numpy().squeeze()
            
            audio_basename = os.path.basename(audio_filename)
            audio_save_dir = os.path.dirname(audio_filename)
            
            det_save_filename = os.path.join(audio_save_dir, audio_basename.replace('.wav', '_det.pkl'))
            
            with open(det_save_filename, 'wb') as f:
                pickle.dump({'det_score': det_score}, f, protocol=pickle.HIGHEST_PROTOCOL)


class EffectFeatExtractor:
    def __init__(self, config=None, device = None):
        self.config = config
        self.device = device
        self.launch_effect_model()
        
    def launch_effect_model(self):
        model = model_effect.Cnn14_DecisionLevelMax(sample_rate=16000,
                                    window_size=1024, 
                                    hop_size=320, 
                                    mel_bins=64, 
                                    fmin=50,
                                    fmax=8000,
                                    cls_num=7)

        pretrained_model = self.config['model_path']['effect7_model_path']
        checkpoint = torch.load(pretrained_model, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model = model.to(device=self.device)
        
        model = model.eval()

        self.effect_model = model

    def run_effect_classify(self, audio_data):
        """classify the effect class of the input one audio data"""
        audio_data_points = self.config['sample_rate'] * self.config['audio_len_sec']
        if audio_data.shape[0] < audio_data_points:
            audio_data = np.pad(audio_data, (0, audio_data_points - audio_data.shape[0]), 'constant')
        elif audio_data.shape[0] > audio_data_points:   
            audio_data = audio_data[:audio_data_points]
    
        audio_data = np.expand_dims(audio_data, axis=0)
        audio_data = torch.from_numpy(audio_data).to(torch.float32).to(self.device)
        with torch.no_grad():
            output_dict = self.effect_model(audio_data)

        cls_score = F.softmax(torch.squeeze(output_dict['cls_logits']), dim=-1).cpu().detach().numpy().squeeze()

        return cls_score

    def get_effect_classify_score(self, audio_dir):
        audio_filename_list = glob.glob(os.path.join(audio_dir, '*.wav'))
        for audio_id, audio_filename in enumerate(audio_filename_list):
            if audio_id % 100 == 0:
                print(f'Processing {audio_id}/{len(audio_filename_list)}')
            audio_data = wavfile.read(audio_filename)[1].astype(np.float32) / 32768.0
            audio_data_points = self.config['sample_rate'] * self.config['audio_len_sec']
            if audio_data.shape[0] < audio_data_points:
                audio_data = np.pad(audio_data, (0, audio_data_points - audio_data.shape[0]), 'constant')
            elif audio_data.shape[0] > audio_data_points:   
                audio_data = audio_data[:audio_data_points]
        
            audio_data = np.expand_dims(audio_data, axis=0)
            audio_data = torch.from_numpy(audio_data).to(torch.float32).to(self.device)
            with torch.no_grad():
                output_dict = self.effect_model(audio_data)

            cls_score = F.softmax(torch.squeeze(output_dict['cls_logits']), dim=-1).cpu().detach().numpy().squeeze()


            audio_basename = os.path.basename(audio_filename)
            audio_save_dir = os.path.dirname(audio_filename)
                
            det_save_filename = os.path.join(audio_save_dir, audio_basename.replace('.wav', '_cls.pkl'))
                
            with open(det_save_filename, 'wb') as f:
                pickle.dump({'cls_score': cls_score}, f, protocol=pickle.HIGHEST_PROTOCOL)