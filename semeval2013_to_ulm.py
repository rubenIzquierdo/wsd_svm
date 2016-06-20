#!/usr/bin/env python

import sys
import pickle

from my_data_classes import *
from xml.etree import cElementTree
from nltk.corpus import WordNetCorpusReader


def convert_to_lexelt(xml_file, key_file, my_wn_reader):
    valid_ids = set()
    #Load teh valid ids
    fd = open(key_file)
    for line in fd:
        #d013 d013.s022.t005 freddie_mac%1:14:00::
        tokens = line.strip().split()
        valid_ids.add(tokens[1])
    fd.close()
        
    this_data = {}
    
    my_tree = cElementTree.parse(xml_file)
    total_instances = 0
    for text in my_tree.findall('text'):
        text_id = text.get('id')
        tokens_per_sent = {}
        list_sent_id_in_order = []
        target_token_ids_per_sent = {}
        for sentence in text.findall('sentence'):
            
            sent_id = sentence.get('id')
            tokens_per_sent[sent_id] = []
            list_sent_id_in_order.append(sent_id)
            target_token_ids_per_sent[sent_id] = []
            for num_token, element in enumerate(sentence):
                # 1931 instance
                # 6211 wf
                token_id = '%s_%s_%d' % (text_id, sent_id, num_token)
                new_token = Ctoken(token_id)
                new_token.set_text(element.text)
                new_token.set_lemma(element.get('lemma'))
                new_token.set_pos(element.get('pos'))
                tokens_per_sent[sent_id].append(new_token)
                
                if element.tag == 'instance':
                    semeval_id = element.get('id')
                    if semeval_id in valid_ids:
                        target_token_ids_per_sent[sent_id].append((num_token, token_id,semeval_id))
                
        
        SENTENCE_CONTEXT = 3
        for index_sentence, sentence_id in enumerate(list_sent_id_in_order):
            for num_token_id, target_token_id, semeval_id in target_token_ids_per_sent[sentence_id]:
                this_target_token = tokens_per_sent[sentence_id][num_token_id]
                lemma = this_target_token.get_lemma() 
                pos  = this_target_token.get_pos().lower()[0]
                
                   
                if (lemma,pos) not in this_data:
                    this_data[(lemma,pos)] = Clexelt(lemma,pos)
                    this_data[(lemma,pos)].set_wn_possible_skeys(my_wn_reader)

                start_at_sentence = max(index_sentence-SENTENCE_CONTEXT,0)
                end_at_sentence = min(index_sentence+SENTENCE_CONTEXT, len(list_sent_id_in_order)-1)
                
                rel_pos_and_sent_id = []
                for this_idx in range(start_at_sentence,end_at_sentence+1):
                    relative_position = this_idx - index_sentence
                    rel_pos_and_sent_id.append((relative_position, list_sent_id_in_order[this_idx]))
                    
                current_token_idx = 0
                list_tokens = []
                target_indexes = []
                relative_sentence_for_token_id = {}
                for relative_position, current_sentence_id in rel_pos_and_sent_id:
                    for this_token in tokens_per_sent[current_sentence_id]:
                        relative_sentence_for_token_id[this_token.get_id()] = relative_position
                        list_tokens.append(this_token)  
                        if this_token.get_id() == target_token_id:
                            target_indexes.append(current_token_idx)
                        current_token_idx += 1
                                
                new_instance = Cinstance()
                new_instance.set_id(semeval_id)
                new_instance.set_lemma(lemma)
                new_instance.set_pos(pos)
                new_instance.set_tokens(list_tokens)
                new_instance.set_index_head_list(target_indexes)
                new_instance.set_relative_sentence_position_for_token_id(relative_sentence_for_token_id)
                this_data[(lemma,pos)].add_instance(new_instance)
                total_instances += 1

  
    return this_data, total_instances
                    
            

if __name__ == '__main__':
    path_to_semevalfile = 'semeval-2013-task12-test-data/data/multilingual-all-words.en.xml'
    path_to_key = 'semeval-2013-task12-test-data/keys/gold/wordnet/wordnet.en.key'
    path_to_wn_dict = '/home/rbevia/wsd_lfs/resources/WordNet-3.0/dict'
    
    my_wn_reader = WordNetCorpusReader(path_to_wn_dict,None)
    
    data_lexelt, total_instances = convert_to_lexelt(path_to_semevalfile,path_to_key,my_wn_reader)
    
    print('Total of test instances: %d' % total_instances, '(It should be 1644)')
    
    out_file = sys.argv[1]
    fd_out = open(out_file,'wb')
    pickle.dump(data_lexelt, fd_out, protocol=-1)
    fd_out.close()
    print('Output in file %s' % out_file)
    