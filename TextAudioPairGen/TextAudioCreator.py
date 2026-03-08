import os
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import random
import yaml
from datetime import datetime
import json
import AudioEffectGen
import NestCombAudioGen
import sys
sys.path.append('../')
import ontology_manager

class TextAudioCreator(object):
    def __init__(self, config):
        self.config = config
        random.seed(self.config['DATA_CREATION_CONFIG']['random_seed'])
        np.random.seed(self.config['DATA_CREATION_CONFIG']['random_seed'])
        self.seed_audio_path = self.config['DATA_CREATION_CONFIG']['seed_audio_path']
        self.audio_len = self.config['DATA_CREATION_CONFIG']['audio_length']
        self.audioeffect_generator = AudioEffectGen.AudioEffect_Generater(config)
        audio_event_corpus = self.config['AudioEvent_Corpus']
        audio_relation_corpus = self.config['AudioRelation_Corpus']
        self.relation_manager = ontology_manager.OntologyManager(audio_relation_corpus)
        self.event_manager = ontology_manager.OntologyManager(audio_event_corpus)
        self.event_id_list = list()
        for event_ids in self.event_manager.get_tree_one_depth(input_depth=1, return_id=True):
            if not event_ids[0] in self.config['DATA_CREATION_CONFIG']['exclude_event_ids']:
                self.event_id_list.append(event_ids[0])
        self.get_relations_ids()
        self.nestedcomb_generator = NestCombAudioGen.NestCombAudioGen(config)

    def get_relations_ids(self):
        self.nestedcomb_relids = self.relation_manager.retrieve_leafnodes('Nested_Combination', return_id=True)
        self.temporality_relids = self.relation_manager.retrieve_leafnodes('Temporality', return_id=True)
        self.spatiality_relids = self.relation_manager.retrieve_leafnodes('Spatiality', return_id=True)
        self.perceptuality_relids = self.relation_manager.retrieve_leafnodes('Perceptuality', return_id=True)
        self.compositionality_relids = self.relation_manager.retrieve_leafnodes('Compositionality', return_id=True)
        self.count_relids = self.relation_manager.retrieve_leafnodes('Count', return_id=True)

    def normalize_and_convert_audio(self, audio):
        audio = audio / (np.max(np.abs(audio))+0.00001)
        audio = audio * 0.95 #avoid being too loud, 
        audio = audio.astype(np.float16) * 32767

        return audio
    
    def generate_incremental_list(self, min_value, max_value, min_diff):
        if max_value - min_value < min_diff:
            raise ValueError("The range between min_value and max_value is too small for the given minimum difference.")
        
        # Start with the minimum value
        current_value = min_value
        result = [current_value]
        
        # Continue adding values while ensuring the minimum difference
        while current_value + min_diff <= max_value:
            next_value = current_value + random.randint(min_diff, max_value - current_value)
            result.append(next_value)
            current_value = next_value
            
        return result
    
    def get_Temporality_refaudio(self, audio_data_list, relation):
        audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
        audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
        composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)
        if relation in ['Precedence', 'Succession', 'Simultaneity']:
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio_data_list[1].shape[0] // audio_sr

            if relation == 'Precedence':
                audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
                audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            elif relation == 'Succession':
                audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
                audio1_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio1_sec + 1))
            elif relation == 'Simultaneity':
                audio_start_sec = random.randint(0, max(0, audio_len2gen - max(audio1_sec, audio2_sec)))
                audio1_start_sec = audio_start_sec
                audio2_start_sec = audio_start_sec

            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0]
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[1]
            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        
        elif relation in ['Periodicity']:
            audio_sec = audio_data_list[0].shape[0] // audio_sr
            audio_start_sec = random.randint(0, 3)

            audio_start_sec_list = self.generate_incremental_list(audio_start_sec, audio_len2gen - audio_sec, 1)
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)
            for audio_start_sec in audio_start_sec_list:
                composite_audio[audio_start_sec*audio_sr: (audio_start_sec + audio_sec)*audio_sr] += audio_data_list[0]
            composite_audio = composite_audio / np.max(np.abs(composite_audio))

            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        else:
            raise ValueError('Unknown relation')
        
    def get_Spatiality_refaudio(self, audio_data_list, relation):
        audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
        audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
        composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)
        if relation in ['Closeness', 'Farness']:
            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio1_sec
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            reduced_audio = audio_data_list[0] * loudness_reduction_ratio
            if relation == 'Closeness':
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0]
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += reduced_audio
            elif relation == 'Farness':
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[0]
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += reduced_audio

            composite_audio = self.normalize_and_convert_audio(composite_audio)

            return composite_audio
        
        elif relation in ['Proximity']:
            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio1_sec
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            if random.uniform(0, 1) > 0.5:
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0]
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[0] * proximity_ratio
            else:
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[0]
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0] * proximity_ratio
            composite_audio = self.normalize_and_convert_audio(composite_audio)

            return composite_audio

        elif relation in ['Approaching', 'Departuring']:
            audio_effected = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[0], 
                                                                                  relation,
                                                                                  self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
            audio_sec = audio_effected.shape[0] // audio_sr
            audio_start_sec = random.randint(0, max(0, audio_len2gen - audio_sec))
            composite_audio[audio_start_sec*audio_sr: (audio_start_sec + audio_sec)*audio_sr] += audio_effected
            composite_audio = self.normalize_and_convert_audio(composite_audio)

            return composite_audio
        
        else:
            raise ValueError('Unknown relation')
        
    def get_Perceptuality_refaudio(self, audio_data_list, relation):
        audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
        audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
        composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)

        if relation in ['Balancing', 'Blending']:
            if relation == 'Balancing':
                audio1, audio2 = self.audioeffect_generator.simulate_balancing_effect(audio_data_list)
            if relation == 'Blending':
                audio1, audio2 = self.audioeffect_generator.simulate_blending_effect(audio_data_list)
            audio1_sec = audio1.shape[0] // audio_sr
            audio2_sec = audio2.shape[0] // audio_sr
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2

            composite_audio = self.normalize_and_convert_audio(composite_audio)

            return composite_audio
        
        elif relation in ['Reverberation', 'TimeStretching', 'Amplification', 'Attenuation']:
            if relation == 'Reverberation':
                audio = self.audioeffect_generator.simulate_reverberation_effect(audio_data_list[0])
            if relation == 'TimeStretching':
                audio = self.audioeffect_generator.simulate_timestretching_effect(audio_data_list[0])
            if relation == 'Amplification':
                audio = self.audioeffect_generator.simulate_amplification_effect(audio_data_list[0])
            if relation == 'Attenuation':
                audio = self.audioeffect_generator.simulate_attenuation_effect(audio_data_list[0])
            audio_sec = audio.shape[0] // audio_sr
            audio_start_sec = random.randint(0, max(0, audio_len2gen - audio_sec))
            composite_audio[audio_start_sec*audio_sr: (audio_start_sec + audio_sec)*audio_sr] += audio

            if relation != 'Attenuation':
                composite_audio = composite_audio / (np.max(np.abs(composite_audio)) + 0.00001)
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        else:
            raise ValueError('Unknown relation')
        
    def get_Compositionality_refaudio(self, audio_data_list, relation):
        audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
        audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
        if relation in ['Conjunction', 'Disjunction', 'ExclusiveOr']:
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio_data_list[1].shape[0] // audio_sr
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32)
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0]
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[1]

            composite_audio2[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0]
            composite_audio3[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[1]

            composite_audio1 = self.normalize_and_convert_audio(composite_audio1)
            composite_audio2 = self.normalize_and_convert_audio(composite_audio2)
            composite_audio3 = self.normalize_and_convert_audio(composite_audio3)

            if relation == 'Disjunction':
                return np.stack([composite_audio1, composite_audio2, composite_audio3], axis=0)
            if relation == 'ExclusiveOr':
                return np.stack([composite_audio2, composite_audio3], axis=0)
            if relation == 'Conjunction':
                return composite_audio1
        elif relation == 'Implication':
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio_data_list[1].shape[0] // audio_sr
            audio3_sec = audio_data_list[2].shape[0] // audio_sr
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([composite_audio.shape[0]], np.float32)
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_sec = audio_data_list[2].shape[0] // audio_sr
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] = audio_data_list[0]
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] = audio_data_list[1]
            composite_audio = self.normalize_and_convert_audio(composite_audio)

            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] = audio_data_list[2]
            composite_audio2 = self.normalize_and_convert_audio(composite_audio2)

            return np.stack([composite_audio, composite_audio2], axis=0)
        elif relation == 'Negation':
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio = composite_audio.astype(np.float16) * 32767
            return composite_audio
        else:
            raise ValueError('Unknown relation')

    def get_reference_audio(self, audio_data_list, relation = 'count'):
        audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
        audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
        composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)
        if relation in self.count_relids:
            #if Count, each audio has independent start time, and the composite audio is the sum of all audios
            #use tagging model to evaluate
            for audio in audio_data_list:
                audio_sec = audio.shape[0] // audio_sr
                audio_start_sec = random.randint(0, max(0, audio_len2gen - audio_sec))
                composite_audio[audio_start_sec*audio_sr: (audio_start_sec + audio_sec)*audio_sr] += audio
            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        
        if relation in ['Precedence', 'Succession', 'Simultaneity', 'Periodicity']:
            return self.get_Temporality_refaudio(audio_data_list, relation)
        elif relation in ['Balancing', 'Blending', 'Reverberation', 'TimeStretching', 'Amplification', 'Attenuation']:
            return self.get_Perceptuality_refaudio(audio_data_list, relation)
        elif relation in ['Closeness', 'Farness', 'Proximity', 'Approaching', 'Departuring']:
            return self.get_Spatiality_refaudio(audio_data_list, relation)
        elif relation in ['Conjunction', 'Disjunction', 'Implication', 'Negation', 'ExclusiveOr']:
            return self.get_Compositionality_refaudio(audio_data_list, relation)
        elif relation in self.nestedcomb_relids:
            return self.nestedcomb_generator.generate(audio_data_list, relation)
        else:
            raise ValueError('Unknown relation')

    def get_text_prompt(self, audio_name_list, relation_id):
        text_prompt_templates = self.relation_manager.get_relation_info(relation_id)['text_prompt_template']
        assert len(text_prompt_templates) > 0
        text_template = random.choice(text_prompt_templates)

        if relation_id in self.count_relids:
            text_prompt = text_template.format(A=len(audio_name_list), B = ', '.join(audio_name_list) )
        else:
            # Dynamically create the dictionary of placeholders based on the length of the list
            placeholders = {chr(65 + i): name for i, name in enumerate(audio_name_list)}
            # Format the text template with the generated placeholders
            text_prompt = text_template.format(**placeholders)
        # replace 'audio' with 'sound' with 30% probability
        if random.uniform(0, 1) > 0.7:
            text_prompt = text_prompt.replace('audio', 'sound')

        return text_prompt
    
    def get_one_audio(self, exclude_event_ids, max_time_len = None):
        iter_times = 0
        while True:
            event_id = random.choice(self.event_id_list)
            if event_id not in exclude_event_ids:
                break
            iter_times += 1
            if iter_times > 100:
                break
        allowed_time_lens = list()
        if max_time_len is not None:
            for time_len in [1,2,3,4,5]:
                try:
                    allowed_time_lens.append(time_len) if time_len <= max_time_len else None
                except:
                    return None, None, None
        else:
            allowed_time_lens = [1,2,3,4,5]

        assert len(allowed_time_lens) > 0
        time_len = random.choice(allowed_time_lens)

        audio_label = self.event_manager.get_id_label(event_id)
        seed_audio_dir = self.event_manager.get_seed_audio_dir(event_id)


        audio_filenames = os.listdir(os.path.join(self.config['DATA_CREATION_CONFIG']['seed_audio_path'],
                                                  seed_audio_dir))
        allowed_audio_filenames = list()
        for audio_filebasename in audio_filenames:
            audio_timelen_sec = int(float(audio_filebasename.split('_')[-1].replace('.wav', '').replace('len', '')))
            if audio_timelen_sec in allowed_time_lens:
                allowed_audio_filenames.append(audio_filebasename)

        if len(allowed_audio_filenames) == 0:
            target_time_len = random.choice(allowed_time_lens)
            audio_filename = os.path.join(self.config['DATA_CREATION_CONFIG']['seed_audio_path'],
                                          seed_audio_dir,
                                          random.choice(audio_filenames))
            audio, sr = librosa.load(audio_filename,
                                     sr=self.config['DATA_CREATION_CONFIG']['audio_sr'],
                                     mono=True)
            audio_time_sec = audio.shape[0] // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_start_sec = random.randint(0, audio_time_sec - target_time_len)
            audio = audio[audio_start_sec*self.config['DATA_CREATION_CONFIG']['audio_sr']: (audio_start_sec + target_time_len)*self.config['DATA_CREATION_CONFIG']['audio_sr']]
        else:
            audio_filebasename = random.choice(allowed_audio_filenames)

            audio_filename = os.path.join(self.config['DATA_CREATION_CONFIG']['seed_audio_path'],
                                          seed_audio_dir,
                                          audio_filebasename)
            assert os.path.exists(audio_filename)
            audio, sr = librosa.load(audio_filename,
                                     sr=self.config['DATA_CREATION_CONFIG']['audio_sr'],
                                     mono=True)
        #necessarily pad or clip the audio to the target length
        audio_len_ori = audio.shape[0]
        audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
        audio_len_target = int(float(audio_len_ori)/audio_sr + 0.5)*audio_sr
        if audio_len_target > audio_len_ori:
            audio = np.pad(audio, (0, audio_len_target - audio_len_ori), 'constant')
        
        #augment event_id with synomyns
        event_info = self.event_manager.get_event_info(event_id)
        event_names = [event_info['name']] + event_info['synonyms']
        assert len(event_names) > 0
        event_name = random.choice(event_names)

        #finally normalize the audio
        audio = audio/np.max(np.abs(audio))
        audio = audio/2.0 #halve the amplitude so that the audio is not too loud, the amplification purpose
        
        return audio, audio_label, event_name

    def get_N_audios(self,
                     event_rule_code = 'ABC',
                     event_max_timelen_code = '555'):
        """
        inter_source_category: whether the two audios are from different source categories, the source means how the source generated.
        inter_audio_category: whether the two audios are from different audio categories, the audio category means the audio content,
                              the audio content must be in the same source category
        max audio length: 5 seconds
        """
        audio_names = list()
        audio_data = list()
        audio_labels = list()

        event_rule_code = list(event_rule_code)
        event_max_timelen_code = [int(time_len_tmp) for time_len_tmp in list(event_max_timelen_code)]

        for event_num_idx, (event_rule, event_max_timelen) in enumerate(zip(event_rule_code, event_max_timelen_code)):
            exclude_event_ids = list()
            noneed_to_create = False
            for event_num in range(0, event_num_idx):
                if event_rule == event_rule_code[event_num]: # strong similarity, just take the same data
                    audio, audio_label, audio_id = audio_data[event_num], audio_labels[event_num], audio_names[event_num]
                    noneed_to_create = True
                    break
                elif event_rule != event_rule_code[event_num] and event_rule.lower() == event_rule_code[event_num].lower():
                    for event_id_tmp in self.event_id_list:
                        if event_id_tmp != audio_names[event_num]:
                            exclude_event_ids.append(event_id_tmp) if event_id_tmp not in exclude_event_ids else None
                else:
                    exclude_event_ids.append(audio_names[event_num])

            if not noneed_to_create:
                audio, audio_label, audio_id = self.get_one_audio(exclude_event_ids=exclude_event_ids, max_time_len=event_max_timelen)
            audio_data.append(audio)
            audio_labels.append(audio_label)
            audio_names.append(audio_id)

        return audio_data, audio_names, audio_labels
    
    def get_promptaudio_pairs(self):
        output_dir = self.config['DATA_CREATION_CONFIG']['save_dir']
        os.makedirs(output_dir, exist_ok=True) if not os.path.exists(output_dir) else None
        mainrel2create = self.config['DATA_CREATION_CONFIG']['mainrel2create']
        assert len(mainrel2create) > 0
        data_dict = dict()
        data_dict['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data_dict['author'] = 'Aurelius'
        for main_relation in mainrel2create:
            print('generating for main relation: {}'.format(main_relation))
            data_dict[main_relation] = dict()
            num2gen = self.config['DATA_CREATION_CONFIG']['each_relation_num2gen']
            if main_relation == 'Temporality':
                sub_relations = self.temporality_relids
            elif main_relation == 'Spatiality':
                sub_relations = self.spatiality_relids
            elif main_relation == 'Perceptuality':
                sub_relations = self.perceptuality_relids
            elif main_relation == 'Compositionality':
                sub_relations = self.compositionality_relids
            elif main_relation == 'Nested_Combination':
                sub_relations = self.nestedcomb_relids
            elif main_relation == 'Count':
                sub_relations = self.count_relids
            else:
                raise ValueError('Unknown main relation')

            for sub_relation_id in sub_relations:
                print('generating for sub category: {}'.format(sub_relation_id))
                data_dict[main_relation][sub_relation_id] = list()
                text_prompt_list = list()
                relation_info  = self.relation_manager.get_relation_info(sub_relation_id)
                for num_id in range(num2gen):
                    # get audios, audio names, audio labels
                    if main_relation == 'Count':
                        event_num = random.randint(self.config['DATA_CREATION_CONFIG']['Count_Config']['min_count'],
                                                   self.config['DATA_CREATION_CONFIG']['Count_Config']['max_count'])
                        event_rule_code = [chr(ord('A') + i) for i in range(event_num)]
                        event_rule_code = ''.join(event_rule_code)
                        event_max_timelen_code = [5]*event_num
                    else:
                        event_num = relation_info['arity']
                        event_rule_code = relation_info['event_rule_code']
                        event_max_timelen_code = relation_info['event_max_timelen_code']
                    audio_data_list, audio_name_list, audio_label_list = self.get_N_audios(event_rule_code=event_rule_code,
                                                                                           event_max_timelen_code=event_max_timelen_code)
                    
                    reference_audio = self.get_reference_audio(audio_data_list, relation=sub_relation_id)
                    text_prompt = self.get_text_prompt(audio_name_list, sub_relation_id)

                    print(text_prompt)

                    reference_audio_savename = os.path.join(output_dir, '{}_refaudio_{}.wav'.format(sub_relation_id, num_id))
                    
                    #further save refaudio as .wav data
                    more_ref_audios = False
                    if len(reference_audio.shape) == 1:
                        wavfile.write(reference_audio_savename, 
                                    self.config['DATA_CREATION_CONFIG']['audio_sr'], 
                                    reference_audio.astype(np.int16))
                    else:
                        more_ref_audios = True
                        more_ref_basenames = list()
                        for audio_id in range(reference_audio.shape[0]):
                            ref_basename_tmp = os.path.basename(reference_audio_savename.replace('.wav', '_{}.wav'.format(audio_id)))
                            more_ref_basenames.append(ref_basename_tmp)
                            wavfile.write(reference_audio_savename.replace('.wav', '_{}.wav'.format(audio_id)), 
                                        self.config['DATA_CREATION_CONFIG']['audio_sr'], 
                                        reference_audio[audio_id,:].astype(np.int16))

                    one_data_dict = dict()
                    one_data_dict['text_prompt'] = text_prompt
                    if more_ref_audios:
                        one_data_dict['reference_audio'] = more_ref_basenames
                    else:
                        one_data_dict['reference_audio'] = [os.path.basename(reference_audio_savename)]

                    one_data_dict['audio_name_list'] = audio_name_list
                    one_data_dict['audio_label_list'] = audio_label_list
                    one_data_dict['sub_relation'] = sub_relation_id
                    one_data_dict['main_relation'] = main_relation
                    data_dict[main_relation][sub_relation_id].append(one_data_dict)

                    text_prompt_list.append(text_prompt)

                with open(os.path.join(output_dir, '{}_{}_textprompt.txt'.format(main_relation, sub_relation_id)), 'w') as f:
                    for text_prompt in text_prompt_list:
                        f.writelines(text_prompt + '\n')

        with open(self.config['DATA_CREATION_CONFIG']['save_name'], 'w') as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=True)

        print('Done!')