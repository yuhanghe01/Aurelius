import yaml
import sys
import os
import json
sys.path.append('../')
import ontology_manager
import FeatExtractor as FeatExtractor
import GeneralEvaluator
import RelationEvaluator
import torch

if __name__ == '__main__':
    config_filename = 'eval_config.yaml'
    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device('cuda:0')
    pred_audio_dir = config['predaudio_dir']
    ref_audio_dir = config['refaudio_dir']

    #step 1: get general evaluation result
    extractor = FeatExtractor.EmbedExtractor(config)
    embed_type = ['vggish', 'panns']
    extractor.get_embedding(pred_audio_dir, embed_type)
    extractor.get_embedding(ref_audio_dir, embed_type)

    #step 2: extract SED and effect classification result
    sed_feat_extractor = FeatExtractor.SEDFeatExtractor(config, device=device)
    sed_feat_extractor.get_det_score(pred_audio_dir)

    #step 3: extract effect classification result
    effect_feat_extractor = FeatExtractor.EffectFeatExtractor(config, device=device)
    effect_feat_extractor.get_effect_classify_score(pred_audio_dir)

    
    #step 4: get general evaluation result
    audiorelset_filename = config['audiorelset_filename']
    relset_ontology_manager = ontology_manager.OntologyManager(audiorelset_filename)
    main_cate_list = relset_ontology_manager.retrieve_children('Relation_Root', return_id=True)
    relarity_dict = dict()
    for arity in [1,2,3,4,5]:
        relarity_dict[arity] = relset_ontology_manager.retrieve_leafnodes('Relation_Root', return_id=True, arity=arity)
    
    #step 5: get general evaluation result
    general_eval = GeneralEvaluator.GeneralEvaluator(
        ref_dir = config['refaudio_dir'],
        pred_dir = config['predaudio_dir'],
        ref_data_filename = config['refaudio_data_filename'],
        feat_embed_map = config['feat_embed_map'],
        predaudio_key=config['predaudio_key'])
    
    general_eval_report = general_eval.get_score_report(main_cate_list, relarity_dict)
    
    #step 6: get the relation evaluation result
    data_filename = config['refaudio_data_filename']
    with open(data_filename, 'r') as f:
        data_info = json.load(f)

    relation_eval = RelationEvaluator.RelationEvaluator(config,
                                                        rel_ontology_manager=relset_ontology_manager)

    relation_eval_report = relation_eval.eval_relation(data_dict=data_info,
                                                       pred_dir=config['predaudio_dir'],
                                                       pred_audio_key=config['predaudio_key'],
                                                       relarity_dict=relarity_dict)
    
    #step 7: dump the relation evaluation result
    save_dir = config['result_save_dir']
    with open(os.path.join(save_dir, 'relation_eval_rst.json'), 'w') as f:
        json.dump(relation_eval_report, f, indent=4)
    with open(os.path.join(save_dir, 'general_eval_rst.json'), 'w') as f:
        json.dump(general_eval_report, f, indent=4)
    print('Finished!')