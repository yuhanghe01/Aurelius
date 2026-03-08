import numpy as np
import os
import json


class OntologyManager(object):
    def __init__(self, ontology_filename=None):
        assert os.path.exists(ontology_filename), 'Ontology file does not exist: {}'.format(ontology_filename)
        ontology_dict = dict()
        with open(ontology_filename, 'rb') as f:
            ontology_list = json.load(f)
            for ontology_tmp in ontology_list:
                ontology_id = ontology_tmp['id']
                assert ontology_id not in ontology_dict.keys()
                ontology_dict[ontology_id] = ontology_tmp

        self.ontology_dict = ontology_dict

    def get_all_ids(self):
        all_ids = [id_tmp for id_tmp in self.ontology_dict.keys()]

        return sorted(all_ids)
    
    def get_id_label(self, id_tmp):
        return self.ontology_dict[id_tmp]['label_id']
    
    def get_all_names(self):
        all_names = [self.ontology_dict[id_tmp]['name'] for id_tmp in self.ontology_dict.keys()]

        return sorted(all_names)
    
    def get_parent_one_level(self, level = 0, return_id = True):
        """level = 0: root parent
           level = 1: parent just below root
           level = 2: parent just below level=1, and so on
        """
        level_vec = [-1]*len(self.ontology_dict.keys())
        all_ids = self.get_all_ids()
        #interate to fill in level_vec until reach to the target level
        while -1 in level_vec:
            if len(set(level_vec)) == 1 and next(iter(level_vec)) == -1:
                for id_idx, id_tmp in enumerate(all_ids):
                    if self.retrieve_parent(id_tmp, return_id=True) == 'No Parent':
                        level_vec[id_idx] = 0
            current_level = max(level_vec)
            if current_level == level:
                break
            for id_idx, id_tmp in enumerate(all_ids):
                if level_vec[id_idx] == current_level:
                    child_ids = self.retrieve_children(id_tmp, return_id=True)
                    for child_id in child_ids:
                        for id_idx_new in range(len(all_ids)):
                            if all_ids[id_idx_new] == child_id:
                                level_vec[id_idx_new] = current_level+1
        
        if return_id:
            target_ids = [all_ids[idx_tmp] for idx_tmp in np.where(level_vec==level)[0]]

            return target_ids
        else:
            target_names = list()
            for level_tmp, id_tmp in zip(level_vec, all_ids):
                if level_tmp == level:
                    target_names.append(self.ontology_dict[id_tmp]['name'])

            return target_names

    def DFS(self, id_key, max_depth, names):
        if len(self.ontology_dict[id_key]['child_ids']) == 0:
            return max_depth, names
        
        for id_tmp in self.ontology_dict[id_key]['child_ids']:
            return self.DFS(id_tmp, max_depth+1, names + [self.ontology_dict[id_tmp]['name']])

    def get_tree_max_depth(self):
        max_depth = 0
        for id_tmp in self.ontology_dict.keys():
            max_depth_tmp = 0
            names = list()
            max_depth_tmp, names = self.DFS(id_tmp, max_depth_tmp + 1, names + [self.ontology_dict[id_tmp]['name']])

            max_depth = max(max_depth, max_depth_tmp)

        return max_depth
    
    def get_tree_one_depth(self, input_depth, return_id = True ):
        depth_info = list()
        for id_tmp in self.ontology_dict.keys():
            max_depth_tmp = 0
            names = list()
            max_depth_tmp, names = self.DFS(id_tmp, max_depth_tmp + 1, names + [self.ontology_dict[id_tmp]['name']])

            if max_depth_tmp == input_depth:
                if not return_id:
                    depth_info.append(names)
                else:
                    id_list = list()
                    for onto_id in self.ontology_dict.keys():
                        if self.ontology_dict[onto_id]['name'] in names:
                            id_list.append(onto_id)
                    depth_info.append(id_list)
                    
        return depth_info
    
    def change_name_id(self, id_or_name):
        if id_or_name in self.ontology_dict.keys():
            return self.ontology_dict[id_or_name]['name']
        else:
            for id_tmp in self.ontology_dict.keys():
                if self.ontology_dict[id_tmp]['name'] == id_or_name:
                    return id_tmp
                
        raise ValueError('Unknown input: {}'.format(id_or_name))
    
    def check_twoparents_instance(self):
        '''check if one audio event catebory has at least two parents'''
        """Check result:
        Clapping: Human group actions, Hands
        Growling: Dog, Cat
        Cowbell: Percussion, Bell
        Bicyle bell: Bell, Bicyle, Alarm
        Vehicle horn, car horn, honking: Car, Alarm
        Doorbell: Door, Alarm
        """
        id_dict = dict()
        for id_tmp in self.ontology_dict.keys():
            id_dict[id_tmp] = list()
        
        for id_tmp in id_dict.keys():
            for child_id in self.ontology_dict[id_tmp]['child_ids']:
                id_dict[child_id].append(id_tmp)

        for id_tmp in id_dict.keys():
            if len(id_dict[id_tmp]) > 1:
                print('leaf name: {}, parents are:'.format(self.ontology_dict[id_tmp]['name']))
                for parent_id in id_dict[id_tmp]:
                    print(self.ontology_dict[parent_id]['name'])

    # def retrieve_leafnodes(self, parent_id, return_id = True):
    #     return self.retrieve_children(parent_id, return_id)

    def retrieve_children(self, parent_id, return_id = True):
        """parent_id can be either name or id, just one depth level
        """
        #step1: locate the parent id
        retrieved = False
        if parent_id not in self.ontology_dict.keys():            
            for parent_id_tmp in self.ontology_dict.keys():
                if parent_id == self.ontology_dict[parent_id_tmp]['name']:
                    parent_id = parent_id_tmp
                    retrieved = True
                    break
        
        if not retrieved and parent_id not in self.ontology_dict.keys():
            raise ValueError('Unknown parent id: {}'.format(parent_id))
        
        #step2: retrive children
        child_ids = self.ontology_dict[parent_id]['child_ids']
        if return_id or len(child_ids) == 0:
            return child_ids
        leafnode_names = list()
        for instance_id in child_ids:
            leafnode_names.append(self.ontology_dict[instance_id]['name'])

        return leafnode_names

    def retrieve_leafnodes(self, parent_id, return_id = True, arity = None):
        """parent_id can be either name or id
        If arity is specified, only return leaf nodes matching that arity.
        """
        #step1: locate the parent id
        retrieved = False
        if parent_id not in self.ontology_dict.keys():            
            for parent_id_tmp in self.ontology_dict.keys():
                if parent_id == self.ontology_dict[parent_id_tmp]['name']:
                    parent_id = parent_id_tmp
                    retrieved = True
                    break
        
        if not retrieved and parent_id not in self.ontology_dict.keys():
            raise ValueError('Unknown parent id: {}'.format(parent_id))
        
        #step2: retrive children
        leafnode_ids = list()
        def local_DFS(id_tmp):
            is_leaf = len(self.ontology_dict[id_tmp]['child_ids']) == 0
            if is_leaf:
                if arity is None or self.ontology_dict[id_tmp].get('arity') == arity:
                    leafnode_ids.append(id_tmp)
                return 
            for child_id in self.ontology_dict[id_tmp]['child_ids']:
                local_DFS(child_id)

        local_DFS(parent_id)

        if return_id:
            return leafnode_ids
        
        leafnode_names = list()
        for instance_id in leafnode_ids:
            leafnode_names.append(self.ontology_dict[instance_id]['name'])

        return leafnode_names

    def get_relation_info(self, relation_id):
        #step1: locate the relation id
        retrieved = False
        if relation_id not in self.ontology_dict.keys():
            for relation_id_tmp in self.ontology_dict.keys():
                if relation_id == self.ontology_dict[relation_id_tmp]['name']:
                    relation_id = relation_id_tmp
                    retrieved = True
                    break

        if not retrieved and relation_id not in self.ontology_dict.keys():
            raise ValueError('Unknown relation id: {}'.format(relation_id))

        return self.ontology_dict[relation_id]

    def get_event_info(self, event_id):
        return self.ontology_dict[event_id]
    
    def get_seed_audio_dir(self, event_id):
        seed_audio_dir = self.ontology_dict[event_id].get('seed_audio_dir', None)

        assert seed_audio_dir is not None

        return seed_audio_dir
    
    def retrieve_parent(self, leaf_id, return_id = True):
        #step1: locate the leaf id
        retrieved = False
        if leaf_id not in self.ontology_dict.keys():
            for leaf_id_tmp in self.ontology_dict.keys():
                if leaf_id == self.ontology_dict[leaf_id_tmp]['name']:
                    leaf_id = leaf_id_tmp
                    retrieved = True
                    break

        if not retrieved and leaf_id not in self.ontology_dict.keys():
            raise ValueError('Unknown leaf id: {}'.format(leaf_id))

        #step2: retrieve parent
        for instance_id in self.ontology_dict.keys():
            if leaf_id in self.ontology_dict[instance_id]['child_ids']:
                if return_id:
                    return instance_id
                else:
                    return self.ontology_dict[instance_id]['name']
                
        return 'No Parent'