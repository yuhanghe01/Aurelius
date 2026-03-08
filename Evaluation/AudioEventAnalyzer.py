import numpy as np
import itertools
import torch
import torch.nn.functional as F

import FeatExtractor

class AudioEventAnalyzer(object):
    """Various audio events analyzing methods
    """
    def __init__(self, config, 
                 device = 'cuda:0' ):
        self.releval_config = config['REL_EVAL_CONFIG']
        self.device = device
        self.config = config
        self.sample_rate = config['sample_rate']
        self.init_effect_classifier()

    def init_effect_classifier(self):
        """initialize the effect classifier
        """
        self.effect_classifier = FeatExtractor.EffectFeatExtractor(self.config, device=self.device)

    def run_effect_classifier(self, input_audio, target_class_name):
        """class ID:
        0: Approaching
        1: Departuring
        2: Timestretching
        3: Attenuation
        4: Amplification
        5: Reverberation
        6: Normal
        """
        name_id_map = {'Approaching': 0, 'Departuring': 1, 'TimeStretching': 2,
                        'Attenuation': 3, 'Amplification': 4, 'Reverberation': 5,
                        'Normal': 6}

        tag_class_score = self.effect_classifier.run_effect_classify(input_audio)
        pred_class_id = np.argmax(tag_class_score)
        target_class_id = name_id_map[target_class_name]

        return pred_class_id == target_class_id

    def get_all_det_audioevents(self, det_score, conf_thrd = 0.5,
                    min_event_lensec = 1):
        """extract potential audio events from the detection score
        """
        det_score = det_score >= conf_thrd
        det_score = det_score.astype(np.int32)
        min_event_len = int(min_event_lensec/0.5)
        event_list = list()

        for time_step in range(det_score.shape[0]):
            potential_event_ids = np.where(det_score[time_step, :]==1)[0]
            if len(potential_event_ids) == 0:
                continue
            for event_id in potential_event_ids:
                start_time = time_step
                end_time = time_step + 1
                while end_time < det_score.shape[0] and det_score[end_time, event_id] == 1:
                    end_time += 1
                if end_time - start_time >= min_event_len:
                    event_list.append([int(event_id), start_time, end_time]) #TODO: need to check if event_id need to be +1
                    assert event_id < det_score.shape[1]
                    for i in range(start_time, min(end_time+1, det_score.shape[0])): #remove the detected event from the detection score
                        det_score[i, event_id] = 0

        return event_list
    
    def get_all_after_audioevents(self, ref_event_list, target_event):
        '''Get all the audio events that are after the target event,
        Each event is a list of [label, start_time, end_time]
        '''
        after_event_list = list()
        target_event_end_time = target_event[2]
        for ref_event in ref_event_list:
            if ref_event[1] > target_event_end_time:
                after_event_list.append(ref_event)
        
        return after_event_list
    
    def get_all_before_audioevents(self, ref_event_list, target_event):
        '''Get all the audio events that are before the target event,
        Each event is a list of [label, start_time, end_time]
        '''
        before_event_list = list()
        target_event_start_time = target_event[1]
        for ref_event in ref_event_list:
            if ref_event[2] < target_event_start_time:
                before_event_list.append(ref_event)
        
        return before_event_list
    
    def get_all_together_audioevents(self, ref_event_list, target_event):
        '''Get all the audio events that overlap with the target event,
        Each event is a list of [label, start_time, end_time]
        '''
        together_event_list = list()
        target_event_start_time = target_event[1]
        target_event_end_time = target_event[2]
        for ref_event in ref_event_list:
            ref_event_start_time= ref_event[1]
            ref_event_end_time = ref_event[2]
            min_start_time = min(ref_event_start_time, target_event_start_time)
            max_end_time = max(ref_event_end_time, target_event_end_time)
            if max_end_time - min_start_time + 1 < ref_event_end_time - ref_event_start_time + 1 + target_event_end_time - target_event_start_time + 1:
                together_event_list.append(ref_event)
        
        return together_event_list

    def check_all_include(self, ref_label_list, label2check_list):
        '''if all labels in ref_label_list are in pred_label_list, return 1, else return 0'''
        exist_ids = list()
        for label2check in label2check_list:
            for ref_id, ref_label in enumerate(ref_label_list):
                if label2check == ref_label and ref_id not in exist_ids:
                    exist_ids.append(ref_id)

        return  len(exist_ids) == len(label2check_list)
    
    def check_any_include(self, ref_label_list, label2check_list):
        """check if any label in label2check_list is in ref_label_list
        """
        for label2check in label2check_list:
            if label2check in ref_label_list:
                return True
            
        return False

    def check_not_include(self, ref_label_list, label2check_list):
        for label2check in label2check_list:
            if label2check in ref_label_list:
                return False
            
        return True

    def get_det_result_with_timestep(self, det_score, conf_thrd = 0.5, min_event_lensec = 1):
        """get all possible detected events from the detection score
        """
        det_score = det_score >= conf_thrd
        det_score = det_score.astype(np.int32)
        min_event_len = int(min_event_lensec/0.5)
        event_list = list() # list of detected events

        for time_step in range(det_score.shape[0]):
            potential_event_ids = np.where(det_score[time_step, :]==1)[0]
            if len(potential_event_ids) == 0:
                continue
            parallel_events = []
            for event_id in potential_event_ids:
                start_time = time_step
                #get the end time
                end_time = time_step + 1
                while end_time < det_score.shape[0] and det_score[end_time, event_id] == 1:
                    end_time += 1
                if end_time - start_time >= min_event_len:
                    parallel_events.append([event_id + 1, start_time, end_time])
                    for i in range(start_time, end_time+1): #remove the detected event from the detection score
                        det_score[i, event_id] = 0
            event_list.append(parallel_events)

        #get the final event list, get all posible combinations
        det_label_list = list()

        all_combinations = list(itertools.product(*event_list))
        for combination in all_combinations:
            det_label_list.append(list(combination))

        return det_label_list

    def get_dettagging_result(self, det_filename, confidence_threshold = 0.5,
                    min_event_lensec = 1):
        """
        det_filename: npy file containing the detection score
        output: a list of ordered detected audio events labels
        """
        det_score = np.load(det_filename) #[20, 25]
        det_score = det_score >= confidence_threshold
        det_score = det_score.astype(np.int32)
        min_event_len = int(min_event_lensec/0.5)
        event_list = list() # list of detected events

        for time_step in range(det_score.shape[0]):
            potential_event_ids = np.where(det_score[time_step, :]==1)[0]
            if len(potential_event_ids) == 0:
                continue
            parallel_events = []
            for event_id in potential_event_ids:
                start_time = time_step
                #get the end time
                end_time = time_step + 1
                while end_time < det_score.shape[0] and det_score[end_time, event_id] == 1:
                    end_time += 1
                if end_time - start_time >= min_event_len:
                    parallel_events.append(event_id + 1)
                    for i in range(start_time, end_time+1): #remove the detected event from the detection score
                        det_score[i, event_id] = 0
            event_list.extend(parallel_events)

        return event_list
    
    def get_det_result(self, det_filename, confidence_threshold = 0.5,
                    min_event_lensec = 1,):
        """
        det_filename: npy file containing the detection score
        output: a list of ordered detected audio events labels
        """
        det_score = np.load(det_filename) #[20, 25]
        det_score = det_score >= confidence_threshold
        det_score = det_score.astype(np.int32)
        det_label_list = []
        min_event_len = int(min_event_lensec/0.5)
        event_list = list() # list of detected events

        for time_step in range(det_score.shape[0]):
            potential_event_ids = np.where(det_score[time_step, :]==1)[0]
            if len(potential_event_ids) == 0:
                continue
            parallel_events = []
            for event_id in potential_event_ids:
                start_time = time_step
                #get the end time
                end_time = time_step + 1
                while end_time < det_score.shape[0] and det_score[end_time, event_id] == 1:
                    end_time += 1
                if end_time - start_time >= min_event_len:
                    parallel_events.append(event_id + 1)
                    for i in range(start_time, end_time+1): #remove the detected event from the detection score
                        det_score[i, event_id] = 0
            event_list.append(parallel_events)

        #get the final event list, get all posible combinations
        det_label_list = list()

        all_combinations = list(itertools.product(*event_list))
        for combination in all_combinations:
            det_label_list.append(list(combination))

        return det_label_list

    def get_tagging_result(self, tagging_filename, confidence_threshold = 0.5,):
        """
        det_filename: npy file containing the tagging score
        output: a list of ordered detected audio events labels
        """
        tagging_score = np.load(tagging_filename).squeeze() #[25]
        tagging_result = np.where(tagging_score >= confidence_threshold)[0]
        if len(tagging_result) == 0:
            return []
        tagging_result = tagging_result + 1

        return tagging_result.tolist()

    def get_loudness(self, audio):
        """get the loudness of the audio
        """
        loudness = np.linalg.norm(audio)
        return loudness
    
    def check_blending(self, audio1, audio2):
        """check if the two audios are blended
        """
        loudness1 = self.get_loudness(audio1)
        loudness2 = self.get_loudness(audio2)
        if abs(loudness1 - loudness2) < max(loudness1, loudness2) * self.config['REL_EVAL_CONFIG']['Perceptuality']['blending_loudness_thred']:
            return True
        else:
            return False
        
    def check_closeness(self, audio1, audio2):
        """check if audio1 is closer than audio2
        """
        loudness1 = self.get_loudness(audio1)
        loudness2 = self.get_loudness(audio2)
        if loudness1 - loudness2 > loudness1 * self.config['REL_EVAL_CONFIG']['Spatiality']['closeness_loudness_thred']:
            return True
        else:
            return False

    def check_farness(self, audio1, audio2):
        """check if audio1 is farther than audio2
        """
        loudness1 = self.get_loudness(audio1)
        loudness2 = self.get_loudness(audio2)
        if loudness2 - loudness1 > loudness2 * self.config['REL_EVAL_CONFIG']['Spatiality']['farness_loudness_thred']:
            return True
        else:
            return False
    
    def check_proximity(self, audio1, audio2):
        """check if audio1 is at almost the same distance of audio2
        """
        loudness1 = self.get_loudness(audio1)
        loudness2 = self.get_loudness(audio2)
        if abs(loudness1 - loudness2) < max(loudness1, loudness2) * self.config['REL_EVAL_CONFIG']['Spatiality']['proximity_loudness_thred']:
            return True
        else:
            return False

    def check_balancing(self, domin_audio1, backg_audio):
        """check if audio1 dominates audio2
        """
        domain_audio_loudness = self.get_loudness(domin_audio1)
        backg_audio_loudness = self.get_loudness(backg_audio)
        if domain_audio_loudness - backg_audio_loudness > domain_audio_loudness * self.config['REL_EVAL_CONFIG']['Perceptuality']['balancing_loudness_thred']:
            return True
        else:
            return False