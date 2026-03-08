import numpy as np
import os
import AudioEventAnalyzer

class BinaryArityEvaluator(object):
    def __init__(self, config = None):
        self.config = config
        self.half_sample_rate = int(config['sample_rate'] //2 )
        self.audioevent_analyzer = AudioEventAnalyzer.AudioEventAnalyzer(config)

    def eval_Temp_Spat_Binary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in ['Precedence_Approaching_Binary','Precedence_Departuring_Binary','Succession_Approaching_Binary','Succession_Departuring_Binary']

        if sub_relation in ['Precedence_Approaching_Binary']:
            #[A approaching -> B]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    audio_A = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(audio_A, 'Approaching'):
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Precedence_Departuring_Binary']:
            #[A departing -> B]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    audio_A = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(audio_A, 'Departuring'):
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Succession_Approaching_Binary']:
            #[B -> A approaching]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    audio_A = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(audio_A, 'Approaching'):
                        all_before_events = self.audioevent_analyzer.get_all_before_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Succession_Departuring_Binary']:
            #[B -> A departing]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    audio_A = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(audio_A, 'Departuring'):
                        all_before_events = self.audioevent_analyzer.get_all_before_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[1]]):
                            return 1.
                        
            return 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))
    
    def eval_Percep_Comp_Binary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in [
            "Reverberation_Conjunction_Binary",
            "Reverberation_Disjunction_Binary",
            "Reverberation_ExclusiveOr_Binary",
            "TimeStretching_Conjunction_Binary",
            "TimeStretching_Disjunction_Binary",
            "TimeStretching_ExclusiveOr_Binary",
            "Amplification_Conjunction_Binary",
            "Amplification_Disjunction_Binary",
            "Amplification_ExclusiveOr_Binary",
            "Attenuation_Conjunction_Binary",
            "Attenuation_Disjunction_Binary",
            "Attenuation_ExclusiveOr_Binary"
        ]
        if sub_relation in ['Reverberation_Conjunction_Binary']:
            # [A reverberation, B reverberation]
            for eveint_id1, pred_event1 in enumerate(pred_audioevent_list):
                if pred_event1[0] == gt_label_list[0]:
                    current_audio1 = pred_audio[pred_event1[1]*self.half_sample_rate:pred_event1[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio1, 'Reverberation'):
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id2 == eveint_id1:
                                continue
                            if pred_event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.run_effect_classifier(current_audio2, 'Reverberation'):
                                    return 1.
            return 0.
        
        if sub_relation in ['Reverberation_Disjunction_Binary']:
            # [A reverberation] or [B reverberation] or [A reverberation, B reverberation]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Reverberation'):
                        return 1.
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[1]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Reverberation'):
                        return 1.
            return 0.
        
        if sub_relation in ['Reverberation_ExclusiveOr_Binary']:
            # [A reverberation] or [B reverberation]
            event1_exist, event2_exist = False, False
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Reverberation'):
                        event1_exist = True
                        break
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[1]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Reverberation'):
                        event2_exist = True
                        break

            return 1. if sum([event1_exist, event2_exist]) == 1 else 0.
        
        if sub_relation in ['TimeStretching_Conjunction_Binary']:
            # [A time stretching, B time stretching]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'TimeStretching'):
                        for pred_event2 in pred_audioevent_list:
                            if pred_event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.run_effect_classifier(current_audio2, 'TimeStretching'):
                                    return 1.
            return 0.
        
        if sub_relation in ['TimeStretching_Disjunction_Binary']:
            # [A time stretching] or [B time stretching] or [A time stretching, B time stretching]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'TimeStretching'):
                        return 1.
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[1]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'TimeStretching'):
                        return 1.
            return 0.
        
        if sub_relation in ['TimeStretching_ExclusiveOr_Binary']:
            # [A time stretching] or [B time stretching]
            event1_exist, event2_exist = False, False
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'TimeStretching'):
                        event1_exist = True
                        break
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[1]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'TimeStretching'):
                        event2_exist = True
                        break

            return 1. if sum([event1_exist, event2_exist]) == 1 else 0.
        
        if sub_relation in ['Amplification_Conjunction_Binary']:
            # [A amplification, B amplification]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Amplification'):
                        for pred_event2 in pred_audioevent_list:
                            if pred_event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.run_effect_classifier(current_audio2, 'Amplification'):
                                    return 1.
            return 0.
        
        if sub_relation in ['Amplification_Disjunction_Binary']:
            # [A amplification] or [B amplification] or [A amplification, B amplification]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Amplification'):
                        return 1.
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[1]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Amplification'):
                        return 1.
            return 0.
        
        if sub_relation in ['Amplification_ExclusiveOr_Binary']:
            # [A amplification] or [B amplification]
            event1_exist, event2_exist = False, False
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Amplification'):
                        event1_exist = True
                        break
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[1]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Amplification'):
                        event2_exist = True
                        break

            return 1. if sum([event1_exist, event2_exist]) == 1 else 0.
        
        if sub_relation in ['Attenuation_Conjunction_Binary']:
            # [A attenuation, B attenuation]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Attenuation'):
                        for pred_event2 in pred_audioevent_list:
                            if pred_event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.run_effect_classifier(current_audio2, 'Attenuation'):
                                    return 1.
            return 0.
        
        if sub_relation in ['Attenuation_Disjunction_Binary']:
            # [A attenuation] or [B attenuation] or [A attenuation, B attenuation]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Attenuation'):
                        return 1.
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[1]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Attenuation'):
                        return 1.
            return 0.
        
        if sub_relation in ['Attenuation_ExclusiveOr_Binary']:
            # [A attenuation] or [B attenuation]
            event1_exist, event2_exist = False, False
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Attenuation'):
                        event1_exist = True
                        break
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[1]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Attenuation'):
                        event2_exist = True
                        break

            return 1. if sum([event1_exist, event2_exist]) == 1 else 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))

        
    def eval_Temp_Percep_Binary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in [
            "Precedence_Reverberation_Binary",
            "Precedence_TimeStretching_Binary",
            "Precedence_Amplification_Binary",
            "Precedence_Attenuation_Binary",
            "Succession_Reverberation_Binary",
            "Succession_TimeStretching_Binary",
            "Succession_Amplification_Binary",
            "Succession_Attenuation_Binary"]
        
        if sub_relation in ['Precedence_Reverberation_Binary']:
            # [A reverberation -> B]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Reverberation'):
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Precedence_TimeStretching_Binary']:
            # [A time stretching -> B]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'TimeStretching'):
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Precedence_Amplification_Binary']:
            # [A amplification -> B]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Amplification'):
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Precedence_Attenuation_Binary']:
            # [A attenuation -> B]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Attenuation'):
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Succession_Reverberation_Binary']:
            # [B -> A reverberation]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Reverberation'):
                        all_before_events = self.audioevent_analyzer.get_all_before_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Succession_TimeStretching_Binary']:
            # [B -> A time stretching]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'TimeStretching'):
                        all_before_events = self.audioevent_analyzer.get_all_before_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Succession_Amplification_Binary']:
            # [B -> A amplification]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Amplification'):
                        all_before_events = self.audioevent_analyzer.get_all_before_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[1]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Succession_Attenuation_Binary']:
            # [B -> A attenuation]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Attenuation'):
                        all_before_events = self.audioevent_analyzer.get_all_before_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[1]]):
                            return 1.
                        
            return 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))

    def eval_Spat_Comp_Binary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in [
            "Approaching_Conjunction_Both_Binary",
            "Departuring_Conjunction_Both_Binary",
            "Approaching_Departuring_Conjunction_Binary",
            "Approaching_Departuring_ExclusiveOr_Binary"
        ]
        if sub_relation in ['Approaching_Conjunction_Both_Binary']:
            # [A approaching, B approaching]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Approaching'):
                        for pred_event2 in pred_audioevent_list:
                            if pred_event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.run_effect_classifier(current_audio2, 'Approaching'):
                                    return 1.
            return 0.
        
        if sub_relation in ['Departuring_Conjunction_Both_Binary']:
            # [A departing, B departing]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Departuring'):
                        for pred_event2 in pred_audioevent_list:
                            if pred_event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.run_effect_classifier(current_audio2, 'Departuring'):
                                    return 1.
            return 0.
        
        if sub_relation in ['Approaching_Departuring_Conjunction_Binary']:
            # [A departuring, B approaching]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Departuring'):
                        for pred_event2 in pred_audioevent_list:
                            if pred_event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.run_effect_classifier(current_audio2, 'Approaching'):
                                    return 1.
            return 0.
        
        if sub_relation in ['Approaching_Departuring_ExclusiveOr_Binary']:
            # [A departurng] or [B approaching]
            event1_exist, event2_exist = False, False
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Departuring'):
                        event1_exist = True
                        break
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[1]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Approaching'):
                        event2_exist = True
                        break

            return 1. if sum([event1_exist, event2_exist]) == 1 else 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))
    
class TernaryArityEvaluator(object):
    def __init__(self, config = None):
        self.config = config
        self.half_sample_rate = int(config['sample_rate'] //2 )
        self.audioevent_analyzer = AudioEventAnalyzer.AudioEventAnalyzer(config)

    def eval_Temp_Comp_Ternary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in [
            "Precedence_ExclusiveOr_Ternary",
            "Precedence_Conjunction_Ternary",
            "Precedence_Disjunction_Ternary",
            "Succession_ExclusiveOr_Ternary",
            "Succession_Conjunction_Ternary",
            "Succession_Disjunction_Ternary"
        ]
        if sub_relation in ['Precedence_ExclusiveOr_Ternary']:
            # [A -> C] or [B -> C]
            #step1: check not all the three gt events exist in the pred
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[2]:
                    all_before_events = self.audioevent_analyzer.get_all_before_audioevents(pred_audioevent_list, pred_event)
                    A_include = self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[0]])
                    B_include = self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[1]])

                    if sum([A_include, B_include]) == 1:
                        return 1.
            return 0.
        
        if sub_relation in ['Precedence_Conjunction_Ternary']:
            # [(A, B) -> C]
            for event_id1, pred_event1 in enumerate(pred_audioevent_list):
                if pred_event1[0] == gt_label_list[0]:
                    for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                        if event_id1 == event_id2:
                            continue
                        if pred_event2[0] == gt_label_list[1]:
                            relative_later_event = pred_event1 if pred_event1[2] > pred_event2[2] else pred_event2
                            all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, relative_later_event)
                            if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[2]]):
                                return 1.
            return 0.
        
        if sub_relation in ['Precedence_Disjunction_Ternary']:
            # [A -> C], [B -> C], [(A, B) -> C]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[2]:
                    all_before_events = self.audioevent_analyzer.get_all_before_audioevents(pred_audioevent_list, pred_event)
                    A_include = self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[0]])
                    B_include = self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[1]])
                    if sum([A_include, B_include]) >= 1:
                        return 1.
            return 0.

        if sub_relation in ['Succession_ExclusiveOr_Ternary']:
            # [C -> A] or [C -> B]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[2]:
                    all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                    A_include = self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[0]])
                    B_include = self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]])
                    if sum([A_include, B_include]) == 1:
                        return 1.
            return 0.
        
        if sub_relation in ['Succession_Conjunction_Ternary']:
            #[C -> (A + B)]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[2]:
                    all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                    if self.audioevent_analyzer.check_all_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[0], gt_label_list[1]]):
                        return 1.
            return 0.
        
        if sub_relation in ['Succession_Disjunction_Ternary']:
            # [C -> (A,B)] or [C-> A] or [C -> B]
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[2]:
                    all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                    if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[0], gt_label_list[1]]):
                        return 1.                    
            return 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))

    def eval_Percep_Comp_Ternary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in ['Blending_ExclusiveOr_Ternary']
        if sub_relation in ['Blending_ExclusiveOr_Ternary']:
            # [A blends B] or [C]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if gt_label_list[2] in pred_label_list and gt_label_list[0] not in pred_label_list and gt_label_list[1] not in pred_label_list:
                return 1.
            if gt_label_list[0] in pred_label_list and gt_label_list[1] in pred_label_list and gt_label_list[2] not in pred_label_list:
                for event_id1, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        audioevent_A = pred_event
                        audioA = pred_audio[audioevent_A[1]*self.half_sample_rate:audioevent_A[2]*self.half_sample_rate+1]
                        for event_id2, pred_event in enumerate(pred_audioevent_list):
                            if event_id1 == event_id2:
                                continue
                            if pred_event[0] == gt_label_list[1]:
                                audioevent_B = pred_event
                                audioB = pred_audio[audioevent_B[1]*self.half_sample_rate:audioevent_B[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_blending(audioA, audioB):
                                    return 1.
            return 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))
    
    def eval_Spat_Comp_Ternary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in [
            "Closeness_Conjunction_Ternary",
            "Closeness_ExclusiveOr_Ternary",
            "Proximity_Conjunction_Ternary",
            "Proximity_ExclusiveOr_Ternary",
            "Approaching_Departuring_Conjunction_Ternary"
        ]
        if sub_relation in ['Closeness_Conjunction_Ternary']:
            # [A closer B, C]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[event2[1]*self.half_sample_rate:event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(current_audio, current_audio2):
                                    return 1.
            return 0.
        
        if sub_relation in ['Closeness_ExclusiveOr_Ternary']:
            # [A closer B] or [C]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label not in pred_label_list for gt_label in gt_label_list) and gt_label_list[2] in pred_label_list:
                return 1.
            for event_id, pred_event in enumerate(pred_audioevent_list):
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    for event_id2, event2 in enumerate(pred_audioevent_list):
                        if event_id == event_id2:
                            continue
                        if event2[0] == gt_label_list[1]:
                            current_audio2 = pred_audio[event2[1]*self.half_sample_rate:event2[2]*self.half_sample_rate+1]
                            if self.audioevent_analyzer.check_closeness(current_audio, current_audio2):
                                return 1.
            return 0.
        
        if sub_relation in ['Proximity_Conjunction_Ternary']:
            # [A prox B, C]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if not all(gt_label in pred_label_list for gt_label in gt_label_list):
                return 0.
            for event_id, pred_event in enumerate(pred_audioevent_list):
                if pred_event[0] == gt_label_list[0]:
                    current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    for event_id2, event2 in enumerate(pred_audioevent_list):
                        if event_id == event_id2:
                            continue
                        if event2[0] == gt_label_list[1]:
                            current_audio2 = pred_audio[event2[1]*self.half_sample_rate:event2[2]*self.half_sample_rate+1]
                            if self.audioevent_analyzer.check_proximity(current_audio, current_audio2):
                                return 1.
            return 0.
        
        if sub_relation in ['Proximity_ExclusiveOr_Ternary']:
            # [A prox B] or [C]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:2]) and gt_label_list[2] in pred_label_list:
                return 1.
            if gt_label_list[2] not in pred_label_list:
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[event2[1]*self.half_sample_rate:event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(current_audio, current_audio2):
                                    return 1.                            
            return 0.
        
        if sub_relation in ['Approaching_Departuring_Conjunction_Ternary']:
            # [A approaching, B departuring, C]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[event2[1]*self.half_sample_rate:event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(current_audio, current_audio2):
                                    return 1.
            return 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))
    
    def eval_Spat_Comp_Percep_Ternary(self, gt_label_list, pred_audioevent_list, pred_audio, sub_relation):
        assert sub_relation in [
            "Proximity_Conjunction_Approaching_Ternary",
            "Proximity_ExclusiveOr_Departuring_Ternary"
        ]
        if sub_relation in ['Proximity_Conjunction_Approaching_Ternary']:
            # [A prox B, C approaching]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        audio_A = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if event2[0] == gt_label_list[1]:
                                audio_B = pred_audio[event2[1]*self.half_sample_rate:event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audio_A, audio_B):
                                    for event_id3, event3 in enumerate(pred_audioevent_list):
                                        if event_id3 == event_id or event_id3 == event_id2:
                                            continue
                                        if event3[0] == gt_label_list[2]:
                                            audio_C = pred_audio[event3[1]*self.half_sample_rate:event3[2]*self.half_sample_rate+1]
                                            if self.audioevent_analyzer.run_effect_classifier(audio_C, 'Approaching'):
                                                return 1.
            return 0.
        
        if sub_relation in ['Proximity_ExclusiveOr_Departuring_Ternary']:
            # [A prox B], or [C departuring]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:2]) and gt_label_list[2] in pred_label_list:
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[2]:
                        current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        if self.audioevent_analyzer.run_effect_classifier(current_audio, 'Departuring'):
                            return 1.
            
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:2]) and gt_label_list[2] not in pred_label_list:
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if event2[0] == gt_label_list[1]:
                                current_audio2 = pred_audio[event2[1]*self.half_sample_rate:event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(current_audio, current_audio2):
                                    return 1.
            return 0.

        raise ValueError("Invalid sub_relation: {}".format(sub_relation))

    def eval_Comp_Comp_Ternary(self, gt_label_list, pred_audioevent_list, sub_relation = None):
        assert sub_relation in ["Conjunction_ExclusiveOr_Ternary"]
        if sub_relation in ['Conjunction_ExclusiveOr_Ternary']:
            # [A, C] or [B, C]
            ac_label = [gt_label_list[0], gt_label_list[2]]
            non_ac_label = [gt_label_list[1]]
            if all(gt_label in pred_audioevent_list for gt_label in ac_label) and all(gt_label not in pred_audioevent_list for gt_label in non_ac_label):
                return 1.
            bc_label = [gt_label_list[1], gt_label_list[2]]
            non_bc_label = [gt_label_list[0]]
            if all(gt_label in pred_audioevent_list for gt_label in bc_label) and all(gt_label not in pred_audioevent_list for gt_label in non_bc_label):
                return 1.
            
            return 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))
        
class QuaternaryArityEvaluator(object):
    def __init__(self, config = None):
        self.config = config
        self.audioevent_analyzer = AudioEventAnalyzer.AudioEventAnalyzer(config)
        self.sample_rate = config['sample_rate']
        self.half_sample_rate = int(self.sample_rate//2)
    
    def eval_Temp_Comp_Quaternary(self, gt_label_list, pred_audioevent_list, sub_relation = None):
        assert sub_relation in [
            "Precedence_ExclusiveOr_Quaternary",
            "Precedence_Implication_If_Quaternary",
            "Precedence_Implication_Then_Quaternary",
            "Precedence_Implication_Else_Quaternary"
        ]
        if sub_relation in ['Precedence_ExclusiveOr_Quaternary']:
            # [A -> C] or [B -> C], or [A -> D] or [B -> D]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if sum(gt_label in pred_label_list for gt_label in gt_label_list) != 2:
                return 0.
            if gt_label_list[0] in pred_label_list and gt_label_list[2] in pred_label_list:
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[2]]):
                            return 1.
            if gt_label_list[0] in pred_label_list and gt_label_list[3] in pred_label_list:
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[3]]):
                            return 1.
            if gt_label_list[1] in pred_label_list and gt_label_list[2] in pred_label_list:
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[1]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[2]]):
                            return 1.
            if gt_label_list[1] in pred_label_list and gt_label_list[3] in pred_label_list:
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[1]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[3]]):
                            return 1.
            return 0.
        
        # if sub_relation in ['Precedence_Conjunction_ExclusiveOr_Quaternary']:
        #     # A -> C or A -> D, or B -> C, or B -> D
        #     pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
        #     if all(gt_label in pred_label_list for gt_label in gt_label_list):
        #         return 0.
            
        #     if gt_label_list[0] in pred_label_list and gt_label_list[1] in pred_label_list:
        #         return 0.
        #     if gt_label_list[2] in pred_label_list and gt_label_list[3] in pred_label_list:
        #         return 0.

        #     if gt_label_list[0] in pred_label_list and gt_label_list[2] in pred_label_list:
        #         for pred_event in pred_audioevent_list:
        #             if pred_event[0] == gt_label_list[0]:
        #                 all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
        #                 if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[2]]):
        #                     return 1.
        #     if gt_label_list[0] in pred_label_list and gt_label_list[3] in pred_label_list:
        #         for pred_event in pred_audioevent_list:
        #             if pred_event[0] == gt_label_list[0]:
        #                 all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
        #                 if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[3]]):
        #                     return 1.
        #     if gt_label_list[1] in pred_label_list and gt_label_list[2] in pred_label_list:
        #         for pred_event in pred_audioevent_list:
        #             if pred_event[0] == gt_label_list[1]:
        #                 all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
        #                 if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[2]]):
        #                     return 1.
        #     if gt_label_list[1] in pred_label_list and gt_label_list[3] in pred_label_list:
        #         for pred_event in pred_audioevent_list:
        #             if pred_event[0] == gt_label_list[1]:
        #                 all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
        #                 if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[3]]):
        #                     return 1.
        #     return 0.
        
        if sub_relation in ['Precedence_Implication_If_Quaternary']:
            # [A -> B, C], or D
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] not in pred_label_list:
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                            return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] in pred_label_list:
                return 1.
            return 0.
        
        if sub_relation in ['Precedence_Implication_Then_Quaternary']:
            #[A, B -> C] or D
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] not in pred_label_list:
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[1]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[2]]):
                            return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] in pred_label_list:
                return 1.
            return 0.
        
        if sub_relation in ['Precedence_Implication_Else_Quaternary']:
            # [A, B] or [C -> D]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:2]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[2:4]):
                return 1.
        
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:2]) and all(gt_label in pred_label_list for gt_label in gt_label_list[2:4]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[2]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[3]]):
                            return 1.   
            return 0.      
       
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))

    def eval_Comp_Comp_Quaternary(self, gt_label_list, pred_audioevent_list, sub_relation = None):
        assert sub_relation in [
            "Conjunction_ExclusiveOr_Quaternary",
            "Conjunction_Implication_If_Quaternary",
            "Conjunction_Implication_Then_Quaternary",
            "Conjunction_Implication_Else_Quaternary",
            "ExclusiveOr_Implication_If_Quaternary",
            "ExclusiveOr_Implication_Then_Quaternary",
            "ExclusiveOr_Implication_Else_Quaternary"
        ]
        if sub_relation in ['Conjunction_ExclusiveOr_Quaternary']:
            # [A, C] or [B, C] or [A, D] or [B, D]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if sum(gt_label in pred_label_list for gt_label in gt_label_list) == 2:
                if gt_label_list[0] in pred_label_list and gt_label_list[2] in pred_label_list:
                    return 1.
                if gt_label_list[1] in pred_label_list and gt_label_list[2] in pred_label_list:
                    return 1.
                if gt_label_list[0] in pred_label_list and gt_label_list[3] in pred_label_list:
                    return 1.
                if gt_label_list[1] in pred_label_list and gt_label_list[3] in pred_label_list:
                    return 1.
            return 0.
        
        if sub_relation in ['Conjunction_Implication_If_Quaternary', 'Conjunction_Implication_Then_Quaternary']:
            # [A, B, C] or D
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] not in pred_label_list:
                return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] in pred_label_list:
                return 1.
            return 0.
        
        if sub_relation in ['Conjunction_Implication_Else_Quaternary']:
            # [A, B] or [C, D]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:2]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[2:4]):
                return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:2]) and all(gt_label in pred_label_list for gt_label in gt_label_list[2:4]):
                return 1.
            return 0.
        
        if sub_relation in ['ExclusiveOr_Implication_If_Quaternary']:
            #[A, C] or [B, C] or D
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if sum(gt_label in pred_label_list for gt_label in gt_label_list) == 2:
                if gt_label_list[0] in pred_label_list and gt_label_list[2] in pred_label_list:
                    return 1.
                if gt_label_list[1] in pred_label_list and gt_label_list[2] in pred_label_list:
                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] in pred_label_list:
                return 1.
            return 0.
        
        if sub_relation in ['ExclusiveOr_Implication_Then_Quaternary']:
            # [A, B] or [A, C] or D
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if sum(gt_label in pred_label_list for gt_label in gt_label_list) == 2:
                if gt_label_list[0] in pred_label_list and gt_label_list[1] in pred_label_list:
                    return 1.
                if gt_label_list[0] in pred_label_list and gt_label_list[2] in pred_label_list:
                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] in pred_label_list:
                return 1.
            return 0.
        
        if sub_relation in ['ExclusiveOr_Implication_Else_Quaternary']:
            # [A, B] or [C] or [D]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:2]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[2:4]):
                return 1.
            if sum(gt_label in pred_label_list for gt_label in gt_label_list) == 1:
                if gt_label_list[2]:
                    return 1.
                if gt_label_list[3]:
                    return 1.
                
            return 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))

class QuinaryArityEvaluator(object):
    def __init__(self, config = None):
        self.config = config
        self.audioevent_analyzer = AudioEventAnalyzer.AudioEventAnalyzer(config)
        self.half_sample_rate = int(config['sample_rate']//2)
        """
        "child_ids": [
            "Count_Comp_Quinary",
            "Temporality_Comp_Quinary",
            "Spatiality_Comp_Quinary",
            "Comp_Comp_Quinary"
        ]
        """
    
    def eval_Count_Comp_Quinary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in [
            "Count_Implication_If_Quinary",
            "Count_Implication_Then_Quinary",
            "Count_Implication_Else_Quinary",
            "Count_ExclusiveOr_Quinary"
        ]
        if sub_relation in ['Count_Implication_If_Quinary', 'Count_Implication_Then_Quinary']:
            # [A,B,C,D] or E
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] not in pred_label_list:
                return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.
            return 0.
        
        if sub_relation in ['Count_Implication_Else_Quinary']:
            # [A,B] or [C,D,E] 
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:2]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[2:5]):
                return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:2]) and all(gt_label in pred_label_list for gt_label in gt_label_list[2:5]):
                return 1.
            return 0.
        if sub_relation in ['Count_ExclusiveOr_Quinary']:
            # [A,B,C] or [D, E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] not in pred_label_list and gt_label_list[4] not in pred_label_list:
                return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] in pred_label_list and gt_label_list[4] in pred_label_list:
                return 1.
            return 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))
    
    def eval_Temporality_Comp_Quinary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in [
            "Precedence_Implication_IfThen_Quinary",
            "Precedence_Implication_IfElse_Quinary",
            "Precedence_Implication_ThenElse_Quinary"
        ]
        if sub_relation in ['Precedence_Implication_IfThen_Quinary']:
            #[A->B, C->D] or E
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] not in pred_label_list:
                A_before_B, C_before_D = False, False
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                            A_before_B = True
                            break
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[2]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[3]]):
                            C_before_D = True
                            break
                if A_before_B and C_before_D:
                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.
            return 0.
        if sub_relation in ['Precedence_Implication_IfElse_Quinary']:
            # [A->B, C] or [D -> E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                            return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[3]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[4]]):
                            return 1.
            return 0.
        
        if sub_relation in ['Precedence_Implication_ThenElse_Quinary']:
            # [A, B->C] or [D -> E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[4:5]):
                for pred_event in pred_audioevent_list:
                    if pred_event[0] == gt_label_list[1]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[2]]):
                            return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[3]:
                        all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                        if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[4]]):
                            return 1.
            return 0.

        raise ValueError("Invalid sub_relation: {}".format(sub_relation))

    def eval_Spatiality_Comp_Quinary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in [
            "Closeness_Implication_IfThen_Quinary",
            "Closeness_Implication_IfElse_Quinary",
            "Closeness_Implication_ThenElse_Quinary",
            "Proximity_Implication_IfThen_Quinary",
            "Proximity_Implication_IfElse_Quinary",
            "Proximity_Implication_ThenElse_Quinary",
            "Closeness_Proximity_Implication_IfThen_Quinary",
            "Closeness_Proximity_Implication_IfElse_Quinary",
            "Closeness_Proximity_Implication_ThenElse_Quinary"
        ]
        if sub_relation in ['Closeness_Implication_IfThen_Quinary']:
            # [A closer B, C closer D] or E
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] not in pred_label_list:
                A_closer_B, C_closer_D = False, False
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        audioA = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if pred_event2[0] == gt_label_list[1]:
                                audioB = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(audioA, audioB):
                                    A_closer_B = True
                                    break
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[2]:
                        audioC = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id3, pred_event3 in enumerate(pred_audioevent_list):
                            if event_id3 == event_id:
                                continue
                            if pred_event3[0] == gt_label_list[3]:
                                audioD = pred_audio[pred_event3[1]*self.half_sample_rate:pred_event3[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(audioC, audioD):
                                    C_closer_D = True
                                    break
                if A_closer_B and C_closer_D:
                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.
            return 0.
        
        if sub_relation in ['Closeness_Implication_IfElse_Quinary']:
            #[A closer B, C] or [D closer E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        audioA = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if pred_event2[0] == gt_label_list[1]:
                                audioB = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(audioA, audioB):
                                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[3]:
                        audioC = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id3, pred_event3 in enumerate(pred_audioevent_list):
                            if event_id3 == event_id:
                                continue
                            if pred_event3[0] == gt_label_list[4]:
                                audioD = pred_audio[pred_event3[1]*self.half_sample_rate:pred_event3[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(audioC, audioD):
                                    return 1.
            return 0.
        if sub_relation in ['Closeness_Implication_ThenElse_Quinary']:
            # [A, B closer C] or [D closer E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[1]:
                        audioB = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if pred_event2[0] == gt_label_list[2]:
                                audioC = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(audioB, audioC):
                                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[3]:
                        audioD = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id3, pred_event3 in enumerate(pred_audioevent_list):
                            if event_id3 == event_id:
                                continue
                            if pred_event3[0] == gt_label_list[4]:
                                audioE = pred_audio[pred_event3[1]*self.half_sample_rate:pred_event3[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(audioD, audioE):
                                    return 1.
            return 0.
        
        if sub_relation in ['Proximity_Implication_IfThen_Quinary']:
            # [A proximity B, C proximity D] or E
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] not in pred_label_list:
                A_proximity_B, C_proximity_D = False, False
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        audioA = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if pred_event2[0] == gt_label_list[1]:
                                audioB = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audioA, audioB):
                                    A_proximity_B = True
                                    break
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[2]:
                        audioC = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id3, pred_event3 in enumerate(pred_audioevent_list):
                            if event_id3 == event_id:
                                continue
                            if pred_event3[0] == gt_label_list[3]:
                                audioD = pred_audio[pred_event3[1]*self.half_sample_rate:pred_event3[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audioC, audioD):
                                    C_proximity_D = True
                                    break
                if A_proximity_B and C_proximity_D:
                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.
            return 0.
        
        if sub_relation in ['Proximity_Implication_IfElse_Quinary']:
            # [A proximity B, C] or [D proximity E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        audioA = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if pred_event2[0] == gt_label_list[1]:
                                audioB = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audioA, audioB):
                                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[3]:
                        audioC = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id3, pred_event3 in enumerate(pred_audioevent_list):
                            if event_id3 == event_id:
                                continue
                            if pred_event3[0] == gt_label_list[4]:
                                audioD = pred_audio[pred_event3[1]*self.half_sample_rate:pred_event3[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audioC, audioD):
                                    return 1.
            return 0.
        
        if sub_relation in ['Proximity_Implication_ThenElse_Quinary']:
            #[A, B proximity C] or [D proximity E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[1]:
                        audioB = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if pred_event2[0] == gt_label_list[2]:
                                audioC = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audioB, audioC):
                                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[3]:
                        audioD = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id3, pred_event3 in enumerate(pred_audioevent_list):
                            if event_id3 == event_id:
                                continue
                            if pred_event3[0] == gt_label_list[4]:
                                audioE = pred_audio[pred_event3[1]*self.half_sample_rate:pred_event3[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audioD, audioE):
                                    return 1.
            return 0.
        
        if sub_relation in ['Closeness_Proximity_Implication_IfThen_Quinary']:
            # [A closer B, C proximity D] or E
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] not in pred_label_list:
                A_closer_B, C_proximity_D = False, False
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        audioA = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if pred_event2[0] == gt_label_list[1]:
                                audioB = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(audioA, audioB):
                                    A_closer_B = True
                                    break
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[2]:
                        audioC = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id3, pred_event3 in enumerate(pred_audioevent_list):
                            if event_id3 == event_id:
                                continue
                            if pred_event3[0] == gt_label_list[3]:
                                audioD = pred_audio[pred_event3[1]*self.half_sample_rate:pred_event3[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audioC, audioD):
                                    C_proximity_D = True
                                    break
                if A_closer_B and C_proximity_D:
                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.
            return 0.
        if sub_relation in ['Closeness_Proximity_Implication_IfElse_Quinary']:
            # [A closer B, C] or [D proximity E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[0]:
                        audioA = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if pred_event2[0] == gt_label_list[1]:
                                audioB = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(audioA, audioB):
                                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[3]:
                        audioC = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id3, pred_event3 in enumerate(pred_audioevent_list):
                            if event_id3 == event_id:
                                continue
                            if pred_event3[0] == gt_label_list[4]:
                                audioD = pred_audio[pred_event3[1]*self.half_sample_rate:pred_event3[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audioC, audioD):
                                    return 1.
            return 0.
        
        if sub_relation in ['Closeness_Proximity_Implication_ThenElse_Quinary']:
            # [A, B closer C] or [D proximity E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[1]:
                        audioB = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id2, pred_event2 in enumerate(pred_audioevent_list):
                            if event_id == event_id2:
                                continue
                            if pred_event2[0] == gt_label_list[2]:
                                audioC = pred_audio[pred_event2[1]*self.half_sample_rate:pred_event2[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_closeness(audioB, audioC):
                                    return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label in pred_label_list for gt_label in gt_label_list[3:5]):
                for event_id, pred_event in enumerate(pred_audioevent_list):
                    if pred_event[0] == gt_label_list[3]:
                        audioD = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                        for event_id3, pred_event3 in enumerate(pred_audioevent_list):
                            if event_id3 == event_id:
                                continue
                            if pred_event3[0] == gt_label_list[4]:
                                audioE = pred_audio[pred_event3[1]*self.half_sample_rate:pred_event3[2]*self.half_sample_rate+1]
                                if self.audioevent_analyzer.check_proximity(audioD, audioE):
                                    return 1.
            return 0.
        
        raise ValueError("Invalid sub_relation: {}".format(sub_relation))
    
    def eval_Comp_Comp_Quinary(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in [
            "Conjunction_Implication_IfThen_Quinary",
            "Conjunction_Implication_IfElse_Quinary",
            "Conjunction_Implication_ThenElse_Quinary",
            "ExclusiveOr_Implication_IfThen_Quinary",
            "ExclusiveOr_Implication_IfElse_Quinary",
            "ExclusiveOr_Implication_ThenElse_Quinary",
            "Conjunction_ExclusiveOr_Implication_IfThen_Quinary",
            "Conjunction_ExclusiveOr_Implication_IfElse_Quinary",
            "Conjunction_ExclusiveOr_Implication_ThenElse_Quinary"
        ]
        if sub_relation in ['Conjunction_Implication_IfThen_Quinary']:
            # [A, B, C, D] or E
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] not in pred_label_list:
                return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.
            return 0.
        
        if sub_relation in ['Conjunction_Implication_IfElse_Quinary', 'Conjunction_Implication_ThenElse_Quinary']:
            # [A, B, C] or [D, E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[3:5]):
                return 1.
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label in pred_label_list for gt_label in gt_label_list[3:5]):
                return 1.
            return 0.
        
        if sub_relation in ['ExclusiveOr_Implication_IfThen_Quinary']:
            # [A, C] or [A, D] or [B, C] or [B, D] or [E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            
            AC_gt_labels = [gt_label_list[0], gt_label_list[2]]
            nonAC_gt_labels = [gt_label_list[1], gt_label_list[3], gt_label_list[4]]
            if all(gt_label in pred_label_list for gt_label in AC_gt_labels) and all(gt_label not in pred_label_list for gt_label in nonAC_gt_labels):
                return 1.
            
            AD_gt_labels = [gt_label_list[0], gt_label_list[3]]
            nonAD_gt_labels = [gt_label_list[1], gt_label_list[2], gt_label_list[4]]
            if all(gt_label in pred_label_list for gt_label in AD_gt_labels) and all(gt_label not in pred_label_list for gt_label in nonAD_gt_labels):
                return 1.
            
            BC_gt_labels = [gt_label_list[1], gt_label_list[2]]
            nonBC_gt_labels = [gt_label_list[0], gt_label_list[3], gt_label_list[4]]
            if all(gt_label in pred_label_list for gt_label in BC_gt_labels) and all(gt_label not in pred_label_list for gt_label in nonBC_gt_labels):
                return 1.
            
            BD_gt_labels = [gt_label_list[1], gt_label_list[3]]
            nonBD_gt_labels = [gt_label_list[0], gt_label_list[2], gt_label_list[4]]
            if all(gt_label in pred_label_list for gt_label in BD_gt_labels) and all(gt_label not in pred_label_list for gt_label in nonBD_gt_labels):
                return 1.
            
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.
            
            return 0.
        
        if sub_relation in ['ExclusiveOr_Implication_IfElse_Quinary']:
            # [A, C] or [B, C] or [D] or [E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]

            AC_gt_labels = [gt_label_list[0], gt_label_list[2]]
            nonAC_gt_labels = [gt_label_list[1], gt_label_list[3], gt_label_list[4]]
            if all(gt_label in pred_label_list for gt_label in AC_gt_labels) and all(gt_label not in pred_label_list for gt_label in nonAC_gt_labels):
                return 1.
            
            BC_gt_labels = [gt_label_list[1], gt_label_list[2]]
            nonBC_gt_labels = [gt_label_list[0], gt_label_list[3], gt_label_list[4]]
            if all(gt_label in pred_label_list for gt_label in BC_gt_labels) and all(gt_label not in pred_label_list for gt_label in nonBC_gt_labels):
                return 1.
            
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] in pred_label_list and gt_label_list[4] not in pred_label_list:
                return 1.
        
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.

            return 0.
        
        if sub_relation in ['ExclusiveOr_Implication_ThenElse_Quinary']:
            # [A, B] or [A, C] or [D] or [E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]
            AB_gt_labels = [gt_label_list[0], gt_label_list[1]]
            nonAB_gt_labels = [gt_label_list[2], gt_label_list[3], gt_label_list[4]]
            if all(gt_label in pred_label_list for gt_label in AB_gt_labels) and all(gt_label not in pred_label_list for gt_label in nonAB_gt_labels):
                return 1.
            
            AC_gt_labels = [gt_label_list[0], gt_label_list[2]]
            nonAC_gt_labels = [gt_label_list[1], gt_label_list[3], gt_label_list[4]]
            if all(gt_label in pred_label_list for gt_label in AC_gt_labels) and all(gt_label not in pred_label_list for gt_label in nonAC_gt_labels):
                return 1.
            
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] in pred_label_list and gt_label_list[4] not in pred_label_list:
                return 1.
            
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.
            
            return 0.
        
        if sub_relation in ['Conjunction_ExclusiveOr_Implication_IfThen_Quinary']:
            # [A, B, C] or [A, B, D] or [E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]

            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and all(gt_label not in pred_label_list for gt_label in gt_label_list[3:5]):
                return 1.
            
            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:2]) and gt_label_list[2] not in pred_label_list and gt_label_list[3] in pred_label_list and gt_label_list[4] not in pred_label_list:
                return 1.
            
            if all(gt_label not in pred_label_list for gt_label in gt_label_list[0:4]) and gt_label_list[4] in pred_label_list:
                return 1.
            
            return 0.
        
        if sub_relation in ['Conjunction_ExclusiveOr_Implication_IfElse_Quinary', 'Conjunction_ExclusiveOr_Implication_ThenElse_Quinary']:
            # [A, B, C] or [D] or [E]
            pred_label_list = [pred_event[0] for pred_event in pred_audioevent_list]

            if all(gt_label in pred_label_list for gt_label in gt_label_list[0:3]) and gt_label_list[3] not in pred_label_list and gt_label_list[4] not in pred_label_list:
                return 1.
            
            if sum(gt_label in pred_label_list for gt_label in gt_label_list) == 1 and gt_label_list[3] in pred_label_list:
                return 1.
           
            if sum(gt_label in pred_label_list for gt_label in gt_label_list) == 1 and gt_label_list[4] in pred_label_list:
                return 1.
            
            return 0.

        raise ValueError("Invalid sub_relation: {}".format(sub_relation))

class NestedCombinationRelEvaluator(object):
    def __init__(self, config = None):
        self.config = config
        self.half_sample_rate = int(config['sample_rate']/2)
        self.audioevent_analyzer = AudioEventAnalyzer.AudioEventAnalyzer(config)

        self.binary_arity_evaluator = BinaryArityEvaluator(config)
        self.ternary_arity_evaluator = TernaryArityEvaluator(config)
        self.quaternary_arity_evaluator = QuaternaryArityEvaluator(config)
        self.quinary_arity_evaluator = QuinaryArityEvaluator(config)
    
    def eval_Binary_Arity(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_main_relation = None, sub_relation = None):
        assert sub_main_relation in ['Temp_Spat_Binary','Temp_Percep_Binary','Percep_Comp_Binary','Spat_Comp_Binary']
        if sub_main_relation == 'Temp_Spat_Binary':
            return self.binary_arity_evaluator.eval_Temp_Spat_Binary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if sub_main_relation == 'Temp_Percep_Binary':
            return self.binary_arity_evaluator.eval_Temp_Percep_Binary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if sub_main_relation == 'Percep_Comp_Binary':
            return self.binary_arity_evaluator.eval_Percep_Comp_Binary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if sub_main_relation == 'Spat_Comp_Binary':
            return self.binary_arity_evaluator.eval_Spat_Comp_Binary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        raise ValueError("Invalid sub_main_relation: {}".format(sub_main_relation))
    
    def eval_Ternary_Arity(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_main_relation = None, sub_relation = None):
        assert sub_main_relation in [
            "Temp_Comp_Ternary",
            "Percep_Comp_Ternary",
            "Comp_Comp_Ternary",
            "Spat_Comp_Ternary",
            "Spat_Comp_Percep_Ternary"
        ]
        if sub_main_relation in ['Temp_Comp_Ternary']:
            return self.ternary_arity_evaluator.eval_Temp_Comp_Ternary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if sub_main_relation in ['Percep_Comp_Ternary']:
            return self.ternary_arity_evaluator.eval_Percep_Comp_Ternary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if sub_main_relation in ['Comp_Comp_Ternary']:
            return self.ternary_arity_evaluator.eval_Comp_Comp_Ternary(gt_label_list, pred_audioevent_list, sub_relation)
        if sub_main_relation in ['Spat_Comp_Ternary']:
            return self.ternary_arity_evaluator.eval_Spat_Comp_Ternary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if sub_main_relation in ['Spat_Comp_Percep_Ternary']:
            return self.ternary_arity_evaluator.eval_Spat_Comp_Percep_Ternary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        raise ValueError("Invalid sub_main_relation: {}".format(sub_main_relation))
    
    def eval_Quaternary_Arity(self, gt_label_list, pred_audioevent_list, sub_main_relation = None, sub_relation = None):
        assert sub_main_relation in [
            "Temp_Comp_Quaternary",
            "Comp_Comp_Quaternary",
        ]
        if sub_main_relation in ['Temp_Spat_Comp_Quaternary']:
            return self.quaternary_arity_evaluator.eval_Temp_Comp_Quaternary(gt_label_list, pred_audioevent_list, sub_relation)
        if sub_main_relation in ['Comp_Comp_Quaternary']:
            return self.quaternary_arity_evaluator.eval_Comp_Comp_Quaternary(gt_label_list, pred_audioevent_list, sub_relation)
        if sub_main_relation in ['Temp_Comp_Quaternary']:
            return self.quaternary_arity_evaluator.eval_Temp_Comp_Quaternary(gt_label_list, pred_audioevent_list, sub_relation)
        raise ValueError("Invalid sub_main_relation: {}".format(sub_main_relation))
    
    def eval_Quinary_Arity(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_main_relation = None, sub_relation = None):
        assert sub_main_relation in [
            "Count_Comp_Quinary",
            "Temporality_Comp_Quinary",
            "Spatiality_Comp_Quinary",
            "Comp_Comp_Quinary"
        ]
        if sub_main_relation in ['Count_Comp_Quinary']:
            return self.quinary_arity_evaluator.eval_Count_Comp_Quinary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if sub_main_relation in ['Temporality_Comp_Quinary']:
            return self.quinary_arity_evaluator.eval_Temporality_Comp_Quinary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if sub_main_relation in ['Spatiality_Comp_Quinary']:
            return self.quinary_arity_evaluator.eval_Spatiality_Comp_Quinary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if sub_main_relation in ['Comp_Comp_Quinary']:
            return self.quinary_arity_evaluator.eval_Comp_Comp_Quinary(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        raise ValueError("Invalid sub_main_relation: {}".format(sub_main_relation))

    def eval(self, gt_label_list, pred_audioevent_list, pred_audio = None, main_relation = None, sub_main_relation = None, sub_relation = None):
        if main_relation == 'Binary_Arity':
            return self.eval_Binary_Arity(gt_label_list, pred_audioevent_list, pred_audio, sub_main_relation, sub_relation)
        if main_relation == 'Ternary_Arity':
            return self.eval_Ternary_Arity(gt_label_list, pred_audioevent_list, pred_audio, sub_main_relation, sub_relation)
        if main_relation == 'Quaternary_Arity':
            return self.eval_Quaternary_Arity(gt_label_list, pred_audioevent_list, sub_main_relation, sub_relation)
        if main_relation == 'Quinary_Arity':
            return self.eval_Quinary_Arity(gt_label_list, pred_audioevent_list, pred_audio, sub_main_relation, sub_relation)
        
        raise ValueError("Invalid sub_main_relation: {}".format(sub_main_relation))