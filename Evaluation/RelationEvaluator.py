import numpy as np
import os
import pickle
import scipy.io.wavfile as wavfile
import AudioEventAnalyzer
import NestedCombRelEval

class RelationEvaluator(object):
    def __init__(self, 
                 config = None,
                 rel_ontology_manager = None):
        self.config = config
        self.sample_rate = config['sample_rate']
        self.half_sample_rate = int(self.sample_rate/2)
        
        self.audioevent_analyzer = AudioEventAnalyzer.AudioEventAnalyzer(config)
        assert rel_ontology_manager is not None
        self.rel_ontology_manager = rel_ontology_manager

        self.nested_combination_evaluator = NestedCombRelEval.NestedCombinationRelEvaluator(config)
        
        self.conf_score_thred_list = np.arange(self.config['conf_score_thred_list'][0], 
                                               self.config['conf_score_thred_list'][2]+0.01,
                                               self.config['conf_score_thred_list'][1]).tolist()

        self.parsimony_weight = config['parsimony_weight']
        
    def get_MSR_score(self, gt_label_list, pred_audioevent_list, pred_audio, main_relation = None, sub_relation = None):
        if len(pred_audioevent_list) == 0:
            return 0., 0., 0.
        
        # get presence score
        pred_label_list = [audioevent[0] for audioevent in pred_audioevent_list]
        presence_score = self.get_presence_score(pred_label_list, gt_label_list)
        if presence_score == 0.:
            relation_score = 0.
            parsimony_score = 0.

            return presence_score, relation_score, parsimony_score

        # get relation correctness score
        if main_relation == 'Temporality':
            relation_score = self.eval_Temporality(gt_label_list, pred_audioevent_list, sub_relation)
        if main_relation == 'Spatiality':
            relation_score = self.eval_Spatiality(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if main_relation == 'Count':
            relation_score = presence_score
        if main_relation == 'Perceptuality':
            relation_score = self.eval_Perceptuality(gt_label_list, pred_audioevent_list, pred_audio, sub_relation)
        if main_relation == 'Compositionality':
            relation_score = self.eval_Compositionality(gt_label_list, pred_audioevent_list, sub_relation)
        if main_relation == 'Nested_Combination':
            sub_main_relation = self.rel_ontology_manager.retrieve_parent(sub_relation)
            main_relation = self.rel_ontology_manager.retrieve_parent(sub_main_relation)
            relation_score = self.nested_combination_evaluator.eval(gt_label_list, 
                                                                    pred_audioevent_list, 
                                                                    pred_audio,
                                                                    main_relation,
                                                                    sub_main_relation, 
                                                                    sub_relation)

        # get parsimony score
        parsimony_score = self.get_parsimony_score(gt_label_list, pred_label_list)
            
        return presence_score, relation_score, parsimony_score

    def eval_Temporality(self, gt_label_list, pred_audioevent_list, sub_relation = None ):
        assert sub_relation in ['Precedence', 'Succession', 'Simultaneity', 'Periodicity']
        if sub_relation in ['Precedence']:
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    first_audioevent = pred_event
                    all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, first_audioevent)
                    if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_after_events], [gt_label_list[1]]):
                        return 1.
            return 0.
        
        if sub_relation in ['Succession']:
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    second_audioevent = pred_event
                    all_before_events = self.audioevent_analyzer.get_all_before_audioevents(pred_audioevent_list, second_audioevent)
                    if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_before_events], [gt_label_list[1]]):
                        return 1.
            return 0.

        if sub_relation in ['Simultaneity']:
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    current_audioevent = pred_event
                    all_together_events = self.audioevent_analyzer.get_all_together_audioevents(pred_audioevent_list, current_audioevent)
                    if self.audioevent_analyzer.check_any_include([audioevent[0] for audioevent in all_together_events], [gt_label_list[1]]):
                        return 1.
            return 0.

        if sub_relation in ['Periodicity']:
            #to ensure periodicity, we ask the specified event exists at least three times
            exist_times = 0
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    exist_times += 1
            if exist_times >= 3:
                return 1.
            return 0.
    
    def eval_Perceptuality(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = None):
        assert sub_relation in ['Balancing','Blending','Reverberation','TimeStretching','Amplification','Attenuation']
        if sub_relation in ['TimeStretching', 'Amplification', 'Attenuation', 'Reverberation']:
            for pred_event in pred_audioevent_list:
                current_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                if self.audioevent_analyzer.run_effect_classifier(current_audio, sub_relation):
                    return 1.
            return 0.
        if sub_relation in ['Balancing']:
            # check if the audio events are balanced
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    dominate_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                    for after_event in all_after_events:
                        if after_event[0] == gt_label_list[1]:
                            after_audio = pred_audio[after_event[1]*self.half_sample_rate:after_event[2]*self.half_sample_rate+1]
                            if self.audioevent_analyzer.check_balancing(dominate_audio, after_audio):
                                return 1.
            return 0.
        
        if sub_relation in ['Blending']:
            # check if the audio events are balanced
            for pred_event in pred_audioevent_list:
                if pred_event[0] == gt_label_list[0]:
                    first_audio = pred_audio[pred_event[1]*self.half_sample_rate:pred_event[2]*self.half_sample_rate+1]
                    all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, pred_event)
                    for after_event in all_after_events:
                        if after_event[0] == gt_label_list[1]:
                            after_audio = pred_audio[after_event[1]*self.half_sample_rate:after_event[2]*self.half_sample_rate+1]
                            if self.audioevent_analyzer.check_blending(first_audio, after_audio):
                                return 1.
            return 0.
        
        raise ValueError('The sub relation {} is not supported'.format(sub_relation))

    def eval_Spatiality(self, gt_label_list, pred_audioevent_list, pred_audio = None, sub_relation = 'Closeness'):
        assert sub_relation in ['Closeness', 'Farness', 'Proximity', 'Approaching', 'Departuring']

        # the relation arity is 2
        if sub_relation in ['Closeness', 'Farness', 'Proximity']:
            for pred_event in pred_audioevent_list:
                first_audioevent = pred_event
                first_audio = pred_audio[first_audioevent[1]*self.half_sample_rate:first_audioevent[2]*self.half_sample_rate+1]
                all_after_events = self.audioevent_analyzer.get_all_after_audioevents(pred_audioevent_list, first_audioevent)
                for after_event in all_after_events:
                    if after_event[0] == gt_label_list[0]:
                        after_audio = pred_audio[after_event[1]*self.half_sample_rate:after_event[2]*self.half_sample_rate+1]
                        if sub_relation == 'Closeness':
                            if self.audioevent_analyzer.check_closeness(first_audio, after_audio):
                                return 1.
                        if sub_relation == 'Farness':
                            if self.audioevent_analyzer.check_farness(first_audio, after_audio):
                                return 1.
                        if sub_relation == 'Proximity':
                            if self.audioevent_analyzer.check_proximity(first_audio, after_audio):
                                return 1.
            return 0.

        # arity = 1
        if sub_relation in ['Approaching', 'Departuring']:
            for pred_event in pred_audioevent_list:
                current_audioevent = pred_event
                current_audio = pred_audio[current_audioevent[1]*self.half_sample_rate:current_audioevent[2]*self.half_sample_rate+1]
                if self.audioevent_analyzer.run_effect_classifier(current_audio, sub_relation):
                    return 1.
                
            return 0.
        
    def eval_Compositionality(self, gt_label_list, pred_audioevent_list, sub_relation = None):
        assert sub_relation in ['Conjunction','Disjunction','Negation','ExclusiveOr','Implication']
        pred_label_list = [audioevent[0] for audioevent in pred_audioevent_list]
        if sub_relation in ['Conjunction']:
            relation_score = 1. if self.audioevent_analyzer.check_all_include(ref_label_list=pred_label_list, label2check_list=gt_label_list) else 0.

            return relation_score
        if sub_relation in ['Disjunction']:
            any_include = self.audioevent_analyzer.check_any_include(ref_label_list=pred_label_list, label2check_list=gt_label_list)
            all_include = self.audioevent_analyzer.check_all_include(ref_label_list=pred_label_list, label2check_list=gt_label_list)
            if (not all_include) and any_include:
                relation_score = 1.
            else:
                relation_score = 0.

            return relation_score
        
        if sub_relation in ['ExclusiveOr']:
            any_include = self.audioevent_analyzer.check_any_include(ref_label_list=pred_label_list, label2check_list=gt_label_list)
            all_include = self.audioevent_analyzer.check_all_include(ref_label_list=pred_label_list, label2check_list=gt_label_list)
            if (not all_include) and any_include:
                relation_score = 1.
            else:
                relation_score = 0.

            return relation_score
        
        if sub_relation in ['Negation']:
            relation_score = 1. if self.audioevent_analyzer.check_not_include(ref_label_list=pred_label_list, label2check_list=gt_label_list) else 0.

            return relation_score
        
        if sub_relation in ['Implication']:
            if self.audioevent_analyzer.check_all_include(ref_label_list=pred_label_list, label2check_list=gt_label_list[:2]) and\
                  not self.audioevent_analyzer.check_all_include(ref_label_list=pred_label_list, label2check_list=[gt_label_list[2]]):
                relation_score = 1.
            elif self.audioevent_analyzer.check_not_include(ref_label_list=pred_label_list, label2check_list=gt_label_list[:2]) and \
                  self.audioevent_analyzer.check_all_include(ref_label_list=pred_label_list, label2check_list=[gt_label_list[2]]):
                relation_score = 1.
            else:
                relation_score = 0.

            return relation_score

        raise ValueError('The sub relation {} is not supported'.format(sub_relation))
    
    def get_presence_score(self, ref_label_list, pred_label_list):
        '''if all labels in ref_label_list are in pred_label_list, return 1, else return 0'''
        if self.audioevent_analyzer.check_all_include(ref_label_list, pred_label_list):
            return 1.
        else:
            return 0.

    def target_audio_presence(self, gt_label_list, pred_label_list):
        '''if all labels in gt_label_list are in pred_label_list, return 1, else return 0'''
        gt_label_set = set(gt_label_list)
        pred_label_set = set(pred_label_list)

        intersection = gt_label_set.intersection(pred_label_set)

        return  float(intersection == gt_label_set)

    def get_parsimony_score(self, gt_label_list, pred_label_list):
        '''The parsimony score is the number of labels in the predicted label list that are not in the ground truth label list.
        '''
        redundant_label_num = abs(len(pred_label_list) - len(gt_label_list))
        return np.exp(-1.0*redundant_label_num*self.parsimony_weight)
    
    def eval_relation(self, data_dict, pred_dir, pred_audio_key, relarity_dict = None):
        """Evaluate the relation between the audio events in the predicted label list and the ground truth label list.
        """
        eval_result = dict()
        # all_cate_result = list()
        for main_cate in data_dict.keys():
            if main_cate in ['time', 'author']:
                continue
            eval_result[main_cate] = dict()
            for sub_cate in data_dict[main_cate].keys():
                print('sub_cate: {}'.format(sub_cate))
                # eval_result[main_cate][sub_cate] = dict()
                sub_cate_result = list()
                for data_id, data_tmp in enumerate(data_dict[main_cate][sub_cate]):
                    # print('eval {}-th for sub_cate {} in main_cate {}'.format(data_id, sub_cate, main_cate))
                    result_tmp = list()
                    pred_audio_basename = data_tmp['reference_audio'][0].replace('.wav', '_{}.wav'.format(pred_audio_key))
                    pred_audio_filename = os.path.join(pred_dir, pred_audio_basename)
                    assert os.path.exists(pred_audio_filename)
                    pred_audio = wavfile.read(pred_audio_filename)[1].astype(np.float32) / 32768.0
                    gt_label_list = data_tmp['audio_label_list']
                    det_score_filename = os.path.join(pred_dir, pred_audio_basename.replace('.wav', '_det.pkl'))
                    assert os.path.exists(det_score_filename)
                    with open(det_score_filename, 'rb') as f:
                        det_data = pickle.load(f)

                    pred_det_score = det_data['det_score']

                    # cls_score_filename = os.path.join(pred_dir, pred_audio_basename.replace('.wav', '_cls.pkl'))
                    # assert os.path.exists(cls_score_filename)
                    # with open(cls_score_filename, 'rb') as f:
                    #     cls_data = pickle.load(f)

                    # cls_score = cls_data['cls_score']

                    # pred_det_score = det_data['det_score']

                    # eval_result_tmp = {'Pre': list(), 'Rel': list(), 'Par': list(), 'MSR': list()}
                    for conf_score_thred in self.conf_score_thred_list:
                        pred_audioevent_list = self.audioevent_analyzer.get_all_det_audioevents(pred_det_score, conf_thrd=conf_score_thred)
                        presence_score, relation_score, parsimony_score = self.get_MSR_score(gt_label_list, 
                                                                                            pred_audioevent_list, 
                                                                                            pred_audio,
                                                                                            main_relation=main_cate,
                                                                                            sub_relation=sub_cate)
                        
                        # eval_result_tmp['Pre'].append(float(presence_score))
                        # eval_result_tmp['Rel'].append(float(relation_score))
                        # eval_result_tmp['Par'].append(float(parsimony_score))
                        # eval_result_tmp['MSR'].append(float(presence_score * relation_score * parsimony_score))
                        # print('conf_score_thred: {}, presence_score: {}, relation_score: {}, parsimony_score: {}'.format(conf_score_thred, presence_score, relation_score, parsimony_score))
                        result_tmp.append([float(presence_score), float(relation_score), float(parsimony_score), float(presence_score * relation_score * parsimony_score)])
                    # hello = np.array(result_tmp, np.float32)
                    # print('result_tmp_shape: {}'.format(hello.shape))
                    # if np.sum(hello) > 0.:
                    #     breakpoint()
                    sub_cate_result.append(result_tmp)
                sub_cate_result = np.array(sub_cate_result, np.float32)
                sub_cate_result = np.mean(sub_cate_result, axis=0).tolist() #average across samples
                assert len(sub_cate_result) == 5
                eval_result[main_cate][sub_cate] = sub_cate_result
                
            # # print('sub_cate_result_shape: {}'.format(np.array(sub_cate_result).shape))
            # eval_result_new.append(np.array(sub_cate_result, np.float32))
            # one_main_cate_result = np.array(eval_result_new, np.float32).squeeze()
            # if len(one_main_cate_result.shape) != 3:
            #     one_main_cate_result = np.expand_dims(one_main_cate_result, axis=0)
            # one_main_cate_result = np.mean(one_main_cate_result, axis=0) #across test samples
            # print('one_main_cate_result_shape: {}'.format(one_main_cate_result.shape))
            # # one_main_cate_result = np.mean(one_main_cate_result, axis=0)
            # # print('one_main_cate_result_shape: {}'.format(one_main_cate_result.shape))
            # all_cate_result.append(one_main_cate_result)
        
        # get the result for all main categories
        result_report = dict()

        result_report['main_cate_result'] = dict()

        overal_results = list()
        arity1_results = []
        arity2_results = []
        arity3_results = []
        arity4_results = []
        arity5_results = []

        for main_cate in eval_result.keys():
            result_list = list()
            for sub_cate in eval_result[main_cate].keys():
                result_list.append(eval_result[main_cate][sub_cate])

                if sub_cate in relarity_dict[1]:
                    arity1_results.extend(result_list)
                if sub_cate in relarity_dict[2]:
                    arity2_results.extend(result_list)
                if sub_cate in relarity_dict[3]:
                    arity3_results.extend(result_list)
                if sub_cate in relarity_dict[4]:
                    arity4_results.extend(result_list)
                if sub_cate in relarity_dict[5]:
                    arity5_results.extend(result_list)

            result_list = np.array(result_list, np.float32)
            result_list = np.mean(result_list, axis=0)
            result_list = np.mean(result_list, axis=0).tolist() #across confidence score threshold
            result_report['main_cate_result'][main_cate] = result_list
            overal_results.append(result_list)

        # get the result overall
        overal_results = np.array(overal_results, np.float32)
        overal_results = np.mean(overal_results, axis=0).tolist() #across main categories
        # overal_results = np.mean(overal_results, axis=0).tolist() #across confidence score threshold
        result_report['overall_result'] = overal_results

        # get the result for arity
        result_report['arity_result'] = dict()
        result_report['arity_result']['1'] = np.mean(np.mean(np.array(arity1_results, np.float32), axis=0), axis=0).tolist()
        result_report['arity_result']['2'] = np.mean(np.mean(np.array(arity2_results, np.float32), axis=0), axis=0).tolist()
        result_report['arity_result']['3'] = np.mean(np.mean(np.array(arity3_results, np.float32), axis=0), axis=0).tolist()
        result_report['arity_result']['4'] = np.mean(np.mean(np.array(arity4_results, np.float32), axis=0), axis=0).tolist()
        result_report['arity_result']['5'] = np.mean(np.mean(np.array(arity5_results, np.float32), axis=0), axis=0).tolist()

        result_report['metadata'] = ['presence_score', 'relation_score', 'parsimony_score', 'MSR_score']

        return result_report