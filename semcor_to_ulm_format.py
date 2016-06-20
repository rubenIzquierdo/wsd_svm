#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import os
import glob
import pickle
from lxml import html, etree
from xml.sax.saxutils import escape

from nltk.corpus import WordNetCorpusReader
from my_data_classes import Ctoken, Cinstance, Clexelt
from sensekey_utils import add_sense_info_to_clexelt
from copy import deepcopy


__here__ = os.path.realpath(os.path.dirname(__file__))

def get_pos_from_skey(this_skey):
    m = {'1':'NN', '2': 'VB', '3':'JJ', '4':'R','5':'J'}
    this_pos = None
    if this_skey is not None:
        p = this_skey.find('%')
        if p!= -1:  
            this_pos = m.get(this_skey[p+1], None)
    return this_pos

def add_file(filename, my_wn_reader, data_lexelt):
    my_html_tree = html.parse(filename)
           
    context_node = my_html_tree.find('body/contextfile/context')
    file_id = context_node.get('filename')
    tokens_per_sent = {}            # for every sentence identifier a list of Ctoken 
    sentence_id_for_token = {}      # sentence identifier for token identifier
    sents_in_order = []             # list of sentence identifiers in order
    num_token = 0
    lex_key_for_token_id = {}
    for p_node in context_node.findall('p'):
        for s_node in p_node.findall('s'):
            sentence_id = int(s_node.get('snum'))
            tokens_per_sent[sentence_id] = []
            sents_in_order.append(sentence_id)
            for word_node in s_node:
                token_id = '%s#w%d' % (file_id, num_token)
                new_token = Ctoken(token_id)
                if word_node.tag == 'wf':
                    pos = word_node.get('pos')
                    text = word_node.text
                    cmd = word_node.get('cmd')
                    wnsn = None
                    lemma = None
                    if cmd == 'ignore':
                        lemma = text.lower()
                    elif cmd == 'done':
                        lemma = word_node.get('lemma')
                        wnsn = word_node.get('lexsn')
                        if wnsn is not None:
                            list_wnsn = wnsn.split(';')
                            lex_key_for_token_id[token_id] = []
                            for wnsn in list_wnsn:
                                lex_key_for_token_id[token_id].append(lemma+'%'+wnsn)
                    new_token.set_pos(pos)
                    new_token.set_text(text)
                    new_token.set_lemma(lemma)
                    
                elif word_node.tag == 'punc':
                    pos = 'PUNC'
                    text = word_node.text
                    lemma = text.lower()
                    new_token.set_pos(pos)
                    new_token.set_text(text)
                    new_token.set_lemma(lemma)            
                
                tokens_per_sent[sentence_id].append(new_token)
                num_token += 1
                
                
            
                
    SENTENCE_CONTEXT = 3
    for index_sentence, sentence_id in enumerate(sents_in_order):
        for token in tokens_per_sent[sentence_id]:
            token_id = token.get_id()
            if token_id in lex_key_for_token_id: #Is target
                lemma = token.get_lemma() 
                pos  = token.get_pos()
                
                
                this_lemma_key = '%s.%s' % (lemma,pos[0].lower())
                if this_lemma_key not in data_lexelt:
                    data_lexelt[this_lemma_key] = Clexelt(lemma,pos)
                    data_lexelt[this_lemma_key].set_wn_possible_skeys(my_wn_reader)

                gold_lexkeys = lex_key_for_token_id[token_id]
                
                   
                
                #For this line we need to have called first to set_wn_possible_skeys
                for nsense, gold_key in enumerate(gold_lexkeys):
                    if data_lexelt[this_lemma_key].is_valid_lexkey(gold_key):
                        
                        new_instance = Cinstance()
                        new_instance.set_lemma(lemma)
                        new_instance.set_pos(pos)
                        new_instance.set_id(token_id+'_%s' % nsense)
                        new_instance.set_docsrc(file_id)
                        new_instance.set_lexkeys([gold_key])
                        new_instance.set_confidence_for_senses({gold_key: 1.0})
                        new_instance.set_annotation_type('manual')
        
                        
                        list_tokens = []
                        target_indexes = []
                        
                        start_at_sentence = max(index_sentence-SENTENCE_CONTEXT,0)
                        end_at_sentence = min(index_sentence+SENTENCE_CONTEXT, len(sents_in_order)-1)
                        
                        rel_pos_and_sent_id = []
                        for this_idx in range(start_at_sentence,end_at_sentence+1):
                            relative_position = this_idx - index_sentence
                            rel_pos_and_sent_id.append((relative_position, sents_in_order[this_idx]))
                            
                       
                        current_token_idx = 0
                        
                        relative_sentence_for_token_id = {}
                        for relative_position, current_sentence_id in rel_pos_and_sent_id:
                            for this_token in tokens_per_sent[current_sentence_id]:
                                relative_sentence_for_token_id[this_token.get_id()] = relative_position
                                list_tokens.append(this_token)  # this_token is a Ctoken object
                                if this_token.get_id() == token.get_id():
                                    target_indexes.append(current_token_idx)
                                current_token_idx += 1
                                
                        new_instance.set_tokens(list_tokens)
                        new_instance.set_index_head_list(target_indexes)
                        new_instance.set_relative_sentence_position_for_token_id(relative_sentence_for_token_id)
                        data_lexelt[this_lemma_key].add_instance(new_instance)
                    else:
                        print('Token %s in file %s not valid with lexkeys %s' % (token_id, file_id, str(gold_lexkeys)), file=sys.stderr)    
                    
                            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts semcor to our intermediate ULM format')
    parser.add_argument('-v', action='version', version = '1.0')
    parser.add_argument('-i', dest='input_folder', required=True, help='Path to original semcor main folder')
    parser.add_argument('-o', dest='output', required=True, help='Output folder')
    parser.add_argument('-wn',dest='path_to_wn', required = True, help='Path to the wordnet root folder')
    args = parser.parse_args()
    
    data_lexelt = {}
    
    #Load the NLTK wordnet reader
    if 'dict' not in args.path_to_wn:
        args.path_to_wn = os.path.join(args.path_to_wn,'dict')
    my_wn_reader = WordNetCorpusReader(args.path_to_wn,None)
    
        
    for folder in ['brown1', 'brown2', 'brownv']:
        print('Reading folder %s' % folder, file=sys.stderr)
        path_to_files = os.path.join(args.input_folder,folder,'tagfiles','br-*')
        for filename in glob.glob(path_to_files):
            print('\tFilename: %s' % filename, file=sys.stderr)
            add_file(filename,my_wn_reader, data_lexelt)
    
    os.mkdir(args.output)
    print('Creating training data...', file=sys.stderr)
    total_instances = 0
    total_lemmas = 0
    for item_key, lexelt in data_lexelt.items():
        

        if len(lexelt) > 0:
            # First we call to add_sense_info_to_clexelt to create all the senskey related info
            # We do not need to assign the return value to a variable, as the object is passed by reference
            # and it's modified already 
            print('\tLexical item: %s' % item_key, file=sys.stderr)

            add_sense_info_to_clexelt(lexelt, my_wn_reader, debug=False)
            
            total_lemmas += 1
            total_instances += len(lexelt)
                    
            #Save the lexelt object 
            item_key = item_key.replace('/','_').lower()
            output_bin = os.path.join(args.output,item_key+'.bin')
            fd_bin = open(output_bin,'wb')
            pickle.dump(lexelt, fd_bin, protocol=-1)
            fd_bin.close()
    print('Total number of lemma.pos (unique): %d' % total_lemmas)
    print('Total number of instances: %d' % total_instances)
        
            