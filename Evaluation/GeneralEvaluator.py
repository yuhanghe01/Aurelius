import numpy as np
import os
import json
from scipy.linalg import sqrtm
from numpy.linalg import inv, det


"""
Compute general evaluation metrics for TTA tasks:
KL divergence, FD, FAD
"""
class GeneralEvaluator(object):
    def __init__(
            self,
            ref_dir,
            pred_dir,
            ref_data_filename=None,
            feat_embed_map = None,
            predaudio_key = 'tangoflux'):
        self.ref_dir = ref_dir
        self.pred_dir = pred_dir
        self.feat_embed_map = feat_embed_map
        self.predaudio_key = predaudio_key
        with open(ref_data_filename, 'rb') as f:
            self.ref_data_info = json.load(f)

    def get_ref_pred_embeddings(self, main_cate_name=None, feat_embed_key = None, arity = None):
        if main_cate_name is not None and arity is not None:
            raise ValueError(
                'main_cate_name and arity should not be both set. Please set one of them.')
        ref_embed_list = list()
        pred_embed_list = list()

        feat_embed_key = '_{}_embed.npy'.format(feat_embed_key)
        for main_cate in self.ref_data_info.keys():
            if main_cate_name is not None and main_cate != main_cate_name:
                continue
            if main_cate in ['time', 'author']:
                continue
            for sub_cate in self.ref_data_info[main_cate].keys(): 
                for data_tmp in self.ref_data_info[main_cate][sub_cate]:
                    ref_audio_filename = data_tmp['reference_audio']
                    current_arity = len(data_tmp['audio_label_list'])
                    if arity is not None and current_arity != arity:
                        continue
                    ref_audio_filenames = data_tmp['reference_audio']
                    pred_audio_basename = os.path.basename(ref_audio_filenames[0]).replace('.wav', '_{}.wav'.format(self.predaudio_key))
                    pred_audio_filename = os.path.join(self.pred_dir, pred_audio_basename)
                    assert os.path.exists(pred_audio_filename)
                    pred_audio_embed = np.load(pred_audio_filename.replace('.wav',feat_embed_key))
                    if len(ref_audio_filenames) > 1:
                        ref_audio_embeds = list()
                        for ref_audio_basename in ref_audio_filenames:
                            ref_audio_filename = os.path.join(self.ref_dir, ref_audio_basename)
                            assert os.path.exists(ref_audio_filename)
                            ref_audio_embed = np.load(ref_audio_filename.replace('.wav',feat_embed_key))
                            ref_audio_embeds.append(ref_audio_embed)
                        ref_embed = np.mean(ref_audio_embeds, axis=0)
                        l2_dist = np.sqrt(np.sum(np.square(pred_audio_embed - ref_embed)))
                        min_index = np.argmin(l2_dist)
                        ref_audio_embed = ref_audio_embeds[min_index]
                    else:
                        ref_audio_filename = os.path.join(self.ref_dir, ref_audio_filenames[0])
                        assert os.path.exists(ref_audio_filename)
                        ref_audio_embed = np.load(ref_audio_filename.replace('.wav',feat_embed_key))

                    if len(ref_audio_embed.shape) == 1:
                        ref_audio_embed = np.expand_dims(ref_audio_embed, axis=0)
                    if len(pred_audio_embed.shape) == 1:
                        pred_audio_embed = np.expand_dims(pred_audio_embed, axis=0)
                    ref_embed_list.append(ref_audio_embed)
                    pred_embed_list.append(pred_audio_embed)

        return np.concatenate(ref_embed_list, axis=0), np.concatenate(pred_embed_list, axis=0)

    def calculate_embd_statistics(self, embed_list):
        if isinstance(embed_list, list):
            embed_list = np.array(embed_list, dtype=np.float32)
        mu = np.mean(embed_list, axis=0)
        sigma = np.cov(embed_list, rowvar=False)

        return mu, sigma

    def kl_divergence(self, mu1, sigma1, mu2, sigma2):
        """Compute KL divergence between two multivariate Gaussians."""
        sigma2_inv = inv(sigma2)
        mu_diff = mu2 - mu1

        # Compute the terms of the KL divergence
        trace_term = np.trace(sigma2_inv @ sigma1)
        mean_term = mu_diff.T @ sigma2_inv @ mu_diff
        det_term = np.log(det(sigma2) / det(sigma1))
        d = mu1.shape[0]

        # Final KL divergence value
        return float(0.5 * (trace_term + mean_term - d + det_term))
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """Calculate the Frechet Distance between two multivariate Gaussians."""
        mu_diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Numerical stability: Remove imaginary components if they exist
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Frechet Distance formula
        return float(np.sum(mu_diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean))
    
    def get_score_wrt_maincate(self, maincate_list = None, score_type = 'KL'):
        assert maincate_list is not None, 'maincate_list should not be None'
        assert score_type in ['KL', 'FAD', 'FD'], 'score_type should be KL, FAD or FD'
        score_maincate = dict()
        for main_cate in maincate_list:
            embed_ref, embed_pred = self.get_ref_pred_embeddings(main_cate_name=main_cate,
                                                                 feat_embed_key=self.feat_embed_map[score_type])
            mu_ref, sigma_ref = self.calculate_embd_statistics(embed_ref)
            mu_pred, sigma_pred = self.calculate_embd_statistics(embed_pred)
            if score_type == 'KL':
                score = self.kl_divergence(mu_ref, sigma_ref, mu_pred, sigma_pred)
            elif score_type in ['FAD', 'FD']:
                score = self.calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)
            else:
                raise ValueError('score_type should be KL, FAD or FD')
            score_maincate[main_cate] = float(score)
            
        return score_maincate
    
    def get_score_wrt_arity(self, relarity_dict = None, score_type = 'KL'):
        assert relarity_dict is not None, 'relativity_dict should not be None'
        assert score_type in ['KL', 'FAD', 'FD'], 'score_type should be KL, FAD or FD'
        score_arity = dict()
        for arity in relarity_dict.keys():
            score_arity[arity] = dict()
            embed_ref, embed_pred = self.get_ref_pred_embeddings(arity=arity,
                                                                feat_embed_key=self.feat_embed_map[score_type])
            mu_ref, sigma_ref = self.calculate_embd_statistics(embed_ref)
            mu_pred, sigma_pred = self.calculate_embd_statistics(embed_pred)
            if score_type == 'KL':
                score = self.kl_divergence(mu_ref, sigma_ref, mu_pred, sigma_pred)
            elif score_type == 'FAD':
                score = self.calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)
            else:
                score = self.calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)
            score_arity[arity] = score

        return score_arity

    def get_score(self, score_type = 'KL'):
        assert score_type in ['KL', 'FAD', 'FD'], 'score_type should be KL, FAD or FD'
        if score_type == 'KL':
            embed_ref, embed_pred = self.get_ref_pred_embeddings(feat_embed_key=self.feat_embed_map['KL'])
            mu_ref, sigma_ref = self.calculate_embd_statistics(embed_ref)
            mu_pred, sigma_pred = self.calculate_embd_statistics(embed_pred)
            kl_score = self.kl_divergence(mu_ref, sigma_ref, mu_pred, sigma_pred)
            
            return kl_score
        
        elif score_type == 'FAD':
            embed_ref, embed_pred = self.get_ref_pred_embeddings(feat_embed_key=self.feat_embed_map['FAD'])
            mu_ref, sigma_ref = self.calculate_embd_statistics(embed_ref)
            mu_pred, sigma_pred = self.calculate_embd_statistics(embed_pred)
            fad_score = self.calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)

            return fad_score
        
        elif score_type == 'FD':
            embed_ref, embed_pred = self.get_ref_pred_embeddings(feat_embed_key=self.feat_embed_map['FD'])
            mu_ref, sigma_ref = self.calculate_embd_statistics(embed_ref)
            mu_pred, sigma_pred = self.calculate_embd_statistics(embed_pred)
            fd_score = self.calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)

            return fd_score
        else:
            raise ValueError('score_type should be KL, FAD or FD')
    
    def get_score_report(self, main_cate_list, relarity_dict):
        score_report = {'general': dict(), 'main_cate': dict(), 'arity': dict()}
        score_report['general']['KL'] = self.get_score(score_type='KL')
        score_report['general']['FAD'] = self.get_score(score_type='FAD')
        score_report['general']['FD'] = self.get_score(score_type='FD')
        score_report['main_cate']['KL'] = self.get_score_wrt_maincate(maincate_list=main_cate_list, score_type='KL')
        score_report['main_cate']['FAD'] = self.get_score_wrt_maincate(maincate_list=main_cate_list, score_type='FAD')
        score_report['main_cate']['FD'] = self.get_score_wrt_maincate(maincate_list=main_cate_list, score_type='FD')
        score_report['arity']['KL'] = self.get_score_wrt_arity(relarity_dict=relarity_dict, score_type='KL')
        score_report['arity']['FAD'] = self.get_score_wrt_arity(relarity_dict=relarity_dict, score_type='FAD')
        score_report['arity']['FD'] = self.get_score_wrt_arity(relarity_dict=relarity_dict, score_type='FD')
        
        return score_report
        