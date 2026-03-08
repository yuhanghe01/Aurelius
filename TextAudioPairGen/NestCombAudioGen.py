import os
import numpy as np
import random
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ontology_manager
import AudioEffectGen

class NestCombAudioGen(object):
    def __init__(self, config):
        self.config = config
        self.audio_event_corpus = self.config['AudioEvent_Corpus']
        self.audio_relation_corpus = self.config['AudioRelation_Corpus']
        self.relation_manager = ontology_manager.OntologyManager(self.audio_relation_corpus)
        self.event_manager = ontology_manager.OntologyManager(self.audio_event_corpus)
        self.get_nestedcomb_all_relation_ids()
        self.audioeffect_generator = AudioEffectGen.AudioEffect_Generater(config)
    
    def get_nestedcomb_all_relation_ids(self):
        self.nested_comb_binary_ids = self.relation_manager.retrieve_leafnodes('Binary_Arity', return_id=True)
        self.nested_comb_ternary_ids = self.relation_manager.retrieve_leafnodes('Ternary_Arity', return_id=True)
        self.nested_comb_quaternary_ids = self.relation_manager.retrieve_leafnodes('Quaternary_Arity', return_id=True)
        self.nested_comb_quinary_ids = self.relation_manager.retrieve_leafnodes('Quinary_Arity', return_id=True)

        assert len(self.nested_comb_binary_ids) + len(self.nested_comb_ternary_ids) + \
            len(self.nested_comb_quaternary_ids) + len(self.nested_comb_quinary_ids) == 79
    
    def get_nestcomb_binary_refaudio(self, audio_data_list, relation):
        if  relation in ['Approaching_Conjunction_Both_Binary', 'Departuring_Conjunction_Both_Binary', \
                'Approaching_Departuring_Conjunction_Binary', 'Succession_Approaching_Binary',\
                'Precedence_Approaching_Binary', 'Precedence_Departuring_Binary', \
                'Succession_Departuring_Binary', 'Precedence_Reverberation_Binary', \
                'Precedence_TimeStretching_Binary', 'Precedence_Amplification_Binary', \
                'Precedence_Attenuation_Binary', 'Succession_Reverberation_Binary', \
                'Succession_Reverberation_Binary', 'Succession_TimeStretching_Binary', \
                'Succession_Amplification_Binary', 'Succession_Attenuation_Binary']:
            if relation == 'Approaching_Conjunction_Both_Binary':
                audio1 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[0], 'Approaching',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
                audio2 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[1], 'Approaching',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
            elif relation == 'Departuring_Conjunction_Both_Binary':
                audio1 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[0], 'Departuring',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
                audio2 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[1], 'Departuring',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
            elif relation == 'Approaching_Departuring_Conjunction_Binary':
                audio1 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[0], 'Departuring',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
                audio2 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[1], 'Approaching',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
                if random.random() < 0.5:
                    audio1, audio2 = audio2, audio1
            elif relation in ['Precedence_Approaching_Binary', 'Succession_Approaching_Binary']:
                audio1 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[0], 'Approaching',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
                audio2 = audio_data_list[1]
            elif relation in ['Precedence_Departuring_Binary', 'Succession_Departuring_Binary']:
                audio1 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[0], 'Departuring',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
                audio2 = audio_data_list[1]
            elif relation in ['Precedence_Reverberation_Binary', 'Succession_Reverberation_Binary']:
                audio1 = self.audioeffect_generator.simulate_reverberation_effect(audio_data_list[0])
                audio2 = audio_data_list[1]
            elif relation in ['Precedence_TimeStretching_Binary', 'Succession_TimeStretching_Binary']:
                audio1 = self.audioeffect_generator.simulate_timestretching_effect(audio_data_list[0])
                audio2 = audio_data_list[1]
            elif relation in ['Precedence_Amplification_Binary', 'Succession_Amplification_Binary']:
                audio1 = self.audioeffect_generator.simulate_amplification_effect(audio_data_list[0])
                audio2 = audio_data_list[1]/np.max(np.abs(audio_data_list[1]))
                audio2 = audio2/5.0 # add contrast to amplify the audio1 loudness
            elif relation in ['Precedence_Attenuation_Binary', 'Succession_Attenuation_Binary']:
                audio1 = self.audioeffect_generator.simulate_attenuation_effect(audio_data_list[0])
                audio2 = audio_data_list[1]/np.max(np.abs(audio_data_list[1]))

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            
            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))

            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        elif relation in ['Approaching_Departuring_ExclusiveOr_Binary']:
            audio1 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[0], 'Approaching',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
            audio2 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[1], 'Departuring',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            
            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Reverberation_Conjunction_Binary', 'TimeStretching_Conjunction_Binary', \
                'Amplification_Conjunction_Binary', 'Attenuation_Conjunction_Binary']:
            if relation == 'Reverberation_Conjunction_Binary':
                audio1 = self.audioeffect_generator.simulate_reverberation_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_reverberation_effect(audio_data_list[1])
            if relation == 'TimeStretching_Conjunction_Binary':
                audio1 = self.audioeffect_generator.simulate_timestretching_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_timestretching_effect(audio_data_list[1])
            if relation == 'Amplification_Conjunction_Binary':
                audio1 = self.audioeffect_generator.simulate_amplification_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_amplification_effect(audio_data_list[1])
            if relation == 'Attenuation_Conjunction_Binary':
                audio1 = self.audioeffect_generator.simulate_attenuation_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_attenuation_effect(audio_data_list[1])
            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))

            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2

            if relation != 'Attenuation_Conjunction_Binary':
                composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        elif relation in ['Reverberation_Disjunction_Binary', 'TimeStretching_Disjunction_Binary', \
                'Amplification_Disjunction_Binary', 'Attenuation_Disjunction_Binary']:
            if relation == 'Reverberation_Disjunction_Binary':
                audio1 = self.audioeffect_generator.simulate_reverberation_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_reverberation_effect(audio_data_list[1])
            if relation == 'TimeStretching_Disjunction_Binary':
                audio1 = self.audioeffect_generator.simulate_timestretching_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_timestretching_effect(audio_data_list[1])
            if relation == 'Amplification_Disjunction_Binary':
                audio1 = self.audioeffect_generator.simulate_amplification_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_amplification_effect(audio_data_list[1])
            if relation == 'Attenuation_Disjunction_Binary':
                audio1 = self.audioeffect_generator.simulate_attenuation_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_attenuation_effect(audio_data_list[1])
            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
                
            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2

            if relation != 'Attenuation_Disjunction_Binary':
                composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            if relation != 'Attenuation_Disjunction_Binary':
                composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            #ensure order
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.randint(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec)
            composite_audio3[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio3[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2

            if relation != 'Attenuation_Disjunction_Binary':
                composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3], axis=0)
        elif relation in ['Reverberation_ExclusiveOr_Binary', 'TimeStretching_ExclusiveOr_Binary', \
                'Amplification_ExclusiveOr_Binary', 'Attenuation_ExclusiveOr_Binary']:
            if relation == 'Reverberation_ExclusiveOr_Binary':
                audio1 = self.audioeffect_generator.simulate_reverberation_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_reverberation_effect(audio_data_list[1])
            if relation == 'TimeStretching_ExclusiveOr_Binary':
                audio1 = self.audioeffect_generator.simulate_timestretching_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_timestretching_effect(audio_data_list[1])
            if relation == 'Amplification_ExclusiveOr_Binary':
                audio1 = self.audioeffect_generator.simulate_amplification_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_amplification_effect(audio_data_list[1])
            if relation == 'Attenuation_ExclusiveOr_Binary':
                audio1 = self.audioeffect_generator.simulate_attenuation_effect(audio_data_list[0])
                audio2 = self.audioeffect_generator.simulate_attenuation_effect(audio_data_list[1])
            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
                
            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            if relation != 'Attenuation_ExclusiveOr_Binary':
                composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            if relation != 'Attenuation_ExclusiveOr_Binary':
                composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)

        raise ValueError('Invalid relation for binary arity nested combination')
    
    def get_nestcomb_ternary_refaudio(self, audio_data_list, relation):
        if relation in ['Proximity_Conjunction_Approaching_Ternary']:
            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)

            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            if random.uniform(0, 1) > 0.5:
                audio1 = audio1 * proximity_ratio
            else:
                audio2 = audio2 * proximity_ratio

            audio3 = self.audioeffect_generator.simulate_spatial_movement(audio3, 'Approaching',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))

            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3

            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        elif relation in ['Proximity_ExclusiveOr_Departuring_Ternary']:
            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)

            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            if random.uniform(0, 1) > 0.5:
                audio1 = audio1 * proximity_ratio
            else:
                audio2 = audio2 * proximity_ratio

            audio3 = self.audioeffect_generator.simulate_spatial_movement(audio3, 'Departuring',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.randint(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec)
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2

            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3

            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Closeness_Conjunction_Ternary']:
            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']
            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio_data_list[1].shape[0] // audio_sr
            audio3_sec = audio_data_list[2].shape[0] // audio_sr
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            audio3 = audio_data_list[2]
            audio3_start_sec = random.randint(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec)
            reduced_audio = audio_data_list[0] * loudness_reduction_ratio
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0]
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += reduced_audio
            composite_audio[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3

            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        elif relation in ['Closeness_ExclusiveOr_Ternary']:
            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']
            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio_data_list[1].shape[0] // audio_sr
            audio3_sec = audio_data_list[2].shape[0] // audio_sr
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            audio3 = audio_data_list[2]
            audio3_start_sec = random.randint(0, audio_len2gen - audio3_sec)
            reduced_audio = audio_data_list[0] * loudness_reduction_ratio
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0]
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += reduced_audio
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3

            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Proximity_Conjunction_Ternary']:
            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)
            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio_data_list[1].shape[0] // audio_sr
            audio3_sec = audio_data_list[1].shape[0] // audio_sr
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)
            if random.uniform(0, 1) > 0.5:
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0]
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[0] * proximity_ratio
            else:
                composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[0]
                composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0] * proximity_ratio
            composite_audio[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio_data_list[1]

            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        elif relation in ['Approaching_Departuring_Conjunction_Ternary']:
            audio1 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[0], 'Approaching',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
            audio2 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[1], 'Departuring',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])
            audio3 = self.audioeffect_generator.simulate_spatial_movement(audio_data_list[2], 'Approaching',
                                self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['motion_volume_type'])

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))

            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3

            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        elif relation in ['Proximity_ExclusiveOr_Ternary']:
            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)
            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            audio1_sec = audio_data_list[0].shape[0] // audio_sr
            audio2_sec = audio1_sec
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            audio3_sec = audio_data_list[1].shape[0] // audio_sr
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)
            if random.uniform(0, 1) > 0.5:
                composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0]
                composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[0] * proximity_ratio
            else:
                composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio_data_list[0]
                composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio_data_list[0] * proximity_ratio
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio_data_list[1]

            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Precedence_Conjunction_Ternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))

            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3

            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        
        elif relation in ['Succession_Conjunction_Ternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - max(audio1_sec, audio2_sec) - audio3_sec))
            audio1_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio1_sec + 1))
            audio2_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio2_sec + 1))

            # audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            # audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            # audio3_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))

            composite_audio[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3

            composite_audio = composite_audio / np.max(np.abs(composite_audio))
            composite_audio = composite_audio.astype(np.float16) * 32767

            return composite_audio
        elif relation in ['Precedence_Disjunction_Ternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #A->C
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #B->C
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #A and B -> C

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec - audio2_sec))
            audio3_start_sec = random.randint(max(audio1_sec+audio1_start_sec, audio2_start_sec+audio2_sec), 
                                              max(0, audio_len2gen - audio3_sec))
            
            composite_audio3[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio3[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio3[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3], axis=0)
        
        elif relation in ['Succession_Disjunction_Ternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec - audio1_sec))
            audio1_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio1_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio2_sec + 1))
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec - max(audio1_sec, audio2_sec)))
            audio1_start_sec = random.randint(audio3_start_sec + audio3_sec, audio_len2gen - audio1_sec)
            audio2_start_sec = random.randint(audio3_start_sec + audio3_sec, audio_len2gen - audio2_sec)
            composite_audio3[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio3[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio3[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3], axis=0)
        elif relation in ['Precedence_ExclusiveOr_Ternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Succession_ExclusiveOr_Ternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec - audio1_sec))
            audio1_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio1_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio2_sec + 1))
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Blending_ExclusiveOr_Ternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Conjunction_ExclusiveOr_Ternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        else:
            raise ValueError('Invalid relation for ternary arity nested combination')
        
    def get_nestcomb_quaternary_refaudio(self, audio_data_list, relation):
        if relation in ['Precedence_ExclusiveOr_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #A->C
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #A->D
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #B->C
            composite_audio4 = np.zeros([audio_len2gen*audio_sr], np.float32) #B->D

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio4_sec))
            audio4_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio4_sec + 1))
            composite_audio2[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16)

            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio3[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio3[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec - audio4_sec))
            audio4_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio4_sec + 1))
            composite_audio4[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio4[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio4 = composite_audio4 / np.max(np.abs(composite_audio4))
            composite_audio4 = composite_audio4.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3, composite_audio4], axis=0)
        elif relation in ['Precedence_Implication_If_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']
            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Precedence_Implication_Then_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.randint(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec)
            audio3_start_sec = random.randint(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec)

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Precedence_Implication_Else_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec - audio4_sec))
            audio4_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio4_sec + 1))
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Conjunction_ExclusiveOr_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #A->C
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #A->D
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #B->C
            composite_audio4 = np.zeros([audio_len2gen*audio_sr], np.float32) #B->D

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio4_sec))
            audio4_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio4_sec + 1))
            composite_audio2[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio3[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio3[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec - audio4_sec))
            audio4_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio4_sec + 1))
            composite_audio4[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio4[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio4 = composite_audio4 / np.max(np.abs(composite_audio4))
            composite_audio4 = composite_audio4.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3, composite_audio4], axis=0)
        elif relation in ['Conjunction_Implication_If_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - max(audio1_sec, audio2_sec) - audio3_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - max(audio1_sec, audio2_sec) - audio3_sec))
            audio3_start_sec = random.choice(range(max(audio1_sec, audio2_sec), audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Conjunction_Implication_Then_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - max(audio3_sec, audio2_sec) - audio1_sec))
            audio2_start_sec = random.randint(audio1_sec + audio1_start_sec, audio_len2gen - max(audio2_sec, audio3_sec))
            audio3_start_sec = random.randint(audio1_sec + audio1_start_sec, audio_len2gen - max(audio2_sec, audio3_sec))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Conjunction_Implication_Else_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['ExclusiveOr_Implication_If_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            composite_audio3[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3], axis=0)
        
        elif relation in ['ExclusiveOr_Implication_Then_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #A->B
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #A->C
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #D

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio3_sec + 1))
            composite_audio2[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            composite_audio3[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3], axis=0)
        elif relation in ['ExclusiveOr_Implication_Else_Quaternary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #A->B
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #C
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #D

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            composite_audio3[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3], axis=0)
        else:
            raise ValueError('Invalid relation for quaternary arity nested combination')
    
    def get_nestcomb_quinary_refaudio(self, audio_data_list, relation):
        if relation in ['Count_Implication_If_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen -  audio1_sec - audio4_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen -  audio2_sec - audio4_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen -  audio3_sec - audio4_sec))
            audio4_start_sec = random.choice(range(max(audio1_start_sec + audio1_sec,\
                                                       audio2_start_sec + audio2_sec, \
                                                       audio3_start_sec + audio3_sec), audio_len2gen - audio4_sec + 1))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Count_Implication_Then_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - max(audio2_sec, audio3_sec, audio4_sec)))
            audio2_start_sec = random.randint(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec)
            audio3_start_sec = random.randint(audio1_start_sec + audio1_sec, audio_len2gen - audio3_sec)
            audio4_start_sec = random.randint(audio1_start_sec + audio1_sec, audio_len2gen - audio4_sec)
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Count_Implication_Else_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Count_ExclusiveOr_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Precedence_Implication_IfThen_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec - audio4_sec))
            audio4_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio4_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Precedence_Implication_IfElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec + 1))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec - audio5_sec))
            audio5_start_sec = random.choice(range(audio4_start_sec + audio4_sec, audio_len2gen - audio5_sec + 1))
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Precedence_Implication_ThenElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec - audio5_sec))
            audio5_start_sec = random.choice(range(audio4_start_sec + audio4_sec, audio_len2gen - audio5_sec + 1))
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Closeness_Implication_IfThen_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec - audio4_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec - audio4_sec + 1))
            reduced_audio = audio2 * loudness_reduction_ratio
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += reduced_audio

            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec - audio4_sec + 1))
            audio4_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio4_sec + 1))
            reduced_audio = audio4 * loudness_reduction_ratio
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += reduced_audio
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Closeness_Implication_IfElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            audio3_start_sec = random.randint(audio2_start_sec + audio2_sec, max(0, audio_len2gen - audio3_sec))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2*loudness_reduction_ratio
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec - audio5_sec))
            audio5_start_sec = random.randint(audio4_start_sec+audio4_sec, max(0, audio_len2gen - audio5_sec))
            reduced_audio = audio5*loudness_reduction_ratio
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += reduced_audio
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Closeness_Implication_ThenElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']
            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.randint(audio1_start_sec + audio1_sec, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3*loudness_reduction_ratio
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec - audio5_sec))
            audio5_start_sec = random.choice(range(audio4_start_sec + audio4_sec, audio_len2gen - audio5_sec + 1))
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5*loudness_reduction_ratio
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Proximity_Implication_IfThen_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec - audio4_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec - audio4_sec + 1))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec - audio4_sec + 1))
            audio4_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio4_sec + 1))
            if random.uniform(0, 1) > 0.5:
                composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
                composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2 * proximity_ratio
            else:
                composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1 * proximity_ratio
                composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2

            if random.uniform(0, 1) > 0.5:
                composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
                composite_audio1[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4 * proximity_ratio
            else:
                composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3 * proximity_ratio
                composite_audio1[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Proximity_Implication_IfElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            if random.uniform(0, 1) > 0.5:
                composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
                composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2 * proximity_ratio
            else:
                composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1 * proximity_ratio
                composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec - audio5_sec))
            audio5_start_sec = random.choice(range(audio4_start_sec + audio4_sec, audio_len2gen - audio5_sec + 1))
            if random.uniform(0, 1) > 0.5:
                composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
                composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5 * proximity_ratio
            else:
                composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4 * proximity_ratio
                composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Proximity_Implication_ThenElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.randint(audio1_start_sec + audio1_sec, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            if random.uniform(0, 1) > 0.5:
                composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
                composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3 * proximity_ratio
            else:
                composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2 * proximity_ratio
                composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec - audio5_sec))
            audio5_start_sec = random.choice(range(audio4_start_sec + audio4_sec, audio_len2gen - audio5_sec + 1))
            if random.uniform(0, 1) > 0.5:
                composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
                composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5 * proximity_ratio
            else:
                composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4 * proximity_ratio
                composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Closeness_Proximity_Implication_IfThen_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']
            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec - audio4_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec - audio4_sec + 1))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec - audio4_sec + 1))
            audio4_start_sec = random.choice(range(audio3_start_sec + audio3_sec, audio_len2gen - audio4_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2 * loudness_reduction_ratio
            if random.uniform(0, 1) > 0.5:
                composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
                composite_audio1[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4 * proximity_ratio
            else:
                composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3 * proximity_ratio
                composite_audio1[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Closeness_Proximity_Implication_IfElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']
            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.choice(range(audio1_start_sec + audio1_sec, audio_len2gen - audio2_sec - audio3_sec + 1))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2 * loudness_reduction_ratio
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec - audio5_sec))
            audio5_start_sec = random.choice(range(audio4_start_sec + audio4_sec, audio_len2gen - audio5_sec + 1))
            if random.uniform(0, 1) > 0.5:
                composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
                composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5 * proximity_ratio
            else:
                composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4 * proximity_ratio
                composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Closeness_Proximity_Implication_ThenElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']
            proximity_minratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['proximity_reduction_min_ratio']
            proximity_ratio = random.uniform(proximity_minratio, 1.0)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec - audio2_sec - audio3_sec))
            audio2_start_sec = random.randint(audio1_start_sec + audio1_sec, max(0, audio_len2gen - audio2_sec - audio3_sec))
            audio3_start_sec = random.choice(range(audio2_start_sec + audio2_sec, audio_len2gen - audio3_sec + 1))
            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3 * loudness_reduction_ratio
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec - audio5_sec))
            audio5_start_sec = random.choice(range(audio4_start_sec + audio4_sec, audio_len2gen - audio5_sec + 1))
            if random.uniform(0, 1) > 0.5:
                composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
                composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5 * proximity_ratio
            else:
                composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4 * proximity_ratio
                composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Conjunction_Implication_IfThen_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            loudness_reduction_ratio = self.config['DATA_CREATION_CONFIG']['Spatiality_Config']['loudness_reduction_ratio']

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Conjunction_Implication_IfElse_Quinary', 'Conjunction_Implication_ThenElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32)
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32)

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['ExclusiveOr_Implication_IfThen_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #A,C
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #A,D
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #B,C
            composite_audio4 = np.zeros([audio_len2gen*audio_sr], np.float32) #B,D
            composite_audio5 = np.zeros([audio_len2gen*audio_sr], np.float32) #E

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            composite_audio3[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio3[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            composite_audio4[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio4[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio4 = composite_audio4 / np.max(np.abs(composite_audio4))
            composite_audio4 = composite_audio4.astype(np.float16) * 32767

            composite_audio5[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio5 = composite_audio5 / np.max(np.abs(composite_audio5))
            composite_audio5 = composite_audio5.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3, composite_audio4, composite_audio5], axis=0)
        elif relation in ['ExclusiveOr_Implication_IfElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #A,C
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #B,C
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #D
            composite_audio4 = np.zeros([audio_len2gen*audio_sr], np.float32) #E

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            composite_audio3[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            composite_audio4[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio4 = composite_audio4 / np.max(np.abs(composite_audio4))
            composite_audio4 = composite_audio4.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3, composite_audio4], axis=0)
        elif relation in ['ExclusiveOr_Implication_ThenElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #A,B
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #A,C
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #D
            composite_audio4 = np.zeros([audio_len2gen*audio_sr], np.float32) #E

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio2[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            composite_audio3[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))
            composite_audio3 = composite_audio3.astype(np.float16) * 32767

            composite_audio4[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio4 = composite_audio4 / np.max(np.abs(composite_audio4))
            composite_audio4 = composite_audio4.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2, composite_audio3, composite_audio4], axis=0)
        elif relation in ['Conjunction_Implication_IfThen_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #AB,CD
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #E

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1[audio4_start_sec * audio_sr: (audio4_start_sec + audio4_sec) * audio_sr] += audio4
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            return np.stack([composite_audio1, composite_audio2], axis=0)
        elif relation in ['Conjunction_ExclusiveOr_Implication_IfElse_Quinary', 'Conjunction_ExclusiveOr_Implication_ThenElse_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #AB,C
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #D
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #E

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            composite_audio3[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))

            return np.stack([composite_audio1, composite_audio2, composite_audio3], axis=0)

        elif relation in ['Conjunction_ExclusiveOr_Implication_IfThen_Quinary']:
            # , 'Conjunction_ExclusiveOr_Implication_ThenElse_Quinary', \
            #               'Conjunction_ExclusiveOr_Implication_IfThen_Quinary']:
            audio1 = audio_data_list[0]
            audio2 = audio_data_list[1]
            audio3 = audio_data_list[2]
            audio4 = audio_data_list[3]
            audio5 = audio_data_list[4]

            audio1_sec = len(audio1) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio2_sec = len(audio2) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio3_sec = len(audio3) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio4_sec = len(audio4) // self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio5_sec = len(audio5) // self.config['DATA_CREATION_CONFIG']['audio_sr']

            audio_sr = self.config['DATA_CREATION_CONFIG']['audio_sr']
            audio_len2gen = self.config['DATA_CREATION_CONFIG']['audio_length']

            composite_audio1 = np.zeros([audio_len2gen*audio_sr], np.float32) #AB,C
            composite_audio2 = np.zeros([audio_len2gen*audio_sr], np.float32) #AB,D
            composite_audio3 = np.zeros([audio_len2gen*audio_sr], np.float32) #E

            audio1_start_sec = random.randint(0, max(0, audio_len2gen - audio1_sec))
            audio2_start_sec = random.randint(0, max(0, audio_len2gen - audio2_sec))
            audio3_start_sec = random.randint(0, max(0, audio_len2gen - audio3_sec))
            audio4_start_sec = random.randint(0, max(0, audio_len2gen - audio4_sec))
            audio5_start_sec = random.randint(0, max(0, audio_len2gen - audio5_sec))

            composite_audio1[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio1[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio1[audio3_start_sec*audio_sr: (audio3_start_sec + audio3_sec)*audio_sr] += audio3
            composite_audio1 = composite_audio1 / np.max(np.abs(composite_audio1))
            composite_audio1 = composite_audio1.astype(np.float16) * 32767

            composite_audio2[audio1_start_sec*audio_sr: (audio1_start_sec + audio1_sec)*audio_sr] += audio1
            composite_audio2[audio2_start_sec*audio_sr: (audio2_start_sec + audio2_sec)*audio_sr] += audio2
            composite_audio2[audio4_start_sec*audio_sr: (audio4_start_sec + audio4_sec)*audio_sr] += audio4
            composite_audio2 = composite_audio2 / np.max(np.abs(composite_audio2))
            composite_audio2 = composite_audio2.astype(np.float16) * 32767

            composite_audio3[audio5_start_sec*audio_sr: (audio5_start_sec + audio5_sec)*audio_sr] += audio5
            composite_audio3 = composite_audio3 / np.max(np.abs(composite_audio3))

            return np.stack([composite_audio1, composite_audio2, composite_audio3], axis=0)
        else:
            raise ValueError('Invalid relation type: {}'.format(relation))
    
    def generate(self, audio_data_list, relation):
        if relation in self.nested_comb_binary_ids:
            return self.get_nestcomb_binary_refaudio(audio_data_list, relation)
        elif relation in self.nested_comb_ternary_ids:
            return self.get_nestcomb_ternary_refaudio(audio_data_list, relation)
        elif relation in self.nested_comb_quaternary_ids:
            return self.get_nestcomb_quaternary_refaudio(audio_data_list, relation)
        elif relation in self.nested_comb_quinary_ids:
            return self.get_nestcomb_quinary_refaudio(audio_data_list, relation)
        else:
            raise ValueError('Invalid relation type: {}'.format(relation))
