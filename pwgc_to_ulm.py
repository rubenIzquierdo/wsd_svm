#!/usr/bin/env python

#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import os
import pickle
from lxml import etree
from nltk.corpus import WordNetCorpusReader

from my_data_classes import Ctoken, Cinstance, Clexelt
from sensekey_utils import add_sense_info_to_clexelt


__here__ = os.path.realpath(os.path.dirname(__file__))


def process_wf(child_node):
    #returns text, lemma, pos, list_skeys
    text = lemma = pos = skeys = None
    if child_node.get('tag') == 'un': #unnanotated
        text = child_node.text
        long_lemma = child_node.get('lemma')
        p = long_lemma.find('|')
        if p != -1:
            long_lemma = long_lemma[p+1:]
        p = long_lemma.find('%')
        lemma = long_lemma[:p]
        pos = child_node.get('pos')
        ##SAVE DATA
        #print(text,lemma,pos)
    elif child_node.get('tag') == 'ignore': #ignore
        text = child_node.text
        lemma = child_node.get('lemma')
        pos = child_node.get('pos')
        ##SAVE DATA
        #print(text,lemma,pos)
    elif child_node.get('tag') == 'man' or child_node.get('tag') == 'auto': #manual or auto annotation
        #Look for the <id> taht contains a valid lemma (non empty)
        pos = child_node.get('pos')
        skeys = []
        for id_node in child_node:
            skeys.append(id_node.get('sk'))
        
        #The text and lemma are taken from the last 'id'
        text = id_node.tail
        lemma = id_node.get('lemma')
        if pos is None:
            pos = get_pos_from_skey(skeys[0])
        ##SAVE DATA
        #print(text, lemma, pos, skeys)
    return text, lemma, pos, skeys
       
    
def get_pos_from_skey(this_skey):
    m = {'1':'NN', '2': 'VB', '3':'JJ', '4':'R','5':'J'}
    this_pos = None
    if this_skey is not None:
        p = this_skey.find('%')
        if p!= -1:  
            this_pos = m.get(this_skey[p+1], None)
    return this_pos
            
def process_node(this_root_node, this_id):
    is_first_cf = True
    info_for_cf = None
    list_of_tokens = []
    num_token = 0
    type_tag_for_token_id = {}
    sense_keys_for_token_id = {}
    
    children = []
    for child_node in this_root_node:
        if child_node.tag == 'qf':
            children.extend(child_node.getchildren())
        else:
            children.append(child_node)
        
    
    #for child_node in this_root_node:
    for child_node in children:
        if child_node.tag == 'wf':    
            if info_for_cf is not None:  #there is info from a previous cf
                text, lemma, pos, type_tag, sensekeys = info_for_cf
                is_first_cf = True
                info_for_cf = None

                token_id = '%s_%d' % (this_id,num_token)
                num_token += 1
                new_token = Ctoken(token_id)
                new_token.set_pos(pos)
                new_token.set_text(text)
                if lemma is None: lemma = text.lower()
                new_token.set_lemma(lemma)
                list_of_tokens.append(new_token)
                type_tag_for_token_id[token_id] = type_tag
                sense_keys_for_token_id[token_id] = sensekeys
                
            text, lemma, pos, sensekeys = process_wf(child_node)
            token_id = '%s_%d' % (this_id,num_token)
            num_token += 1
            new_token = Ctoken(token_id)
            new_token.set_pos(pos)
            new_token.set_text(text)
            if lemma is None: lemma = text.lower()
            new_token.set_lemma(lemma)
            list_of_tokens.append(new_token)
            type_tag_for_token_id[token_id] = child_node.get('tag')
            sense_keys_for_token_id[token_id] = sensekeys
            
        elif child_node.tag == 'cf':
            if is_first_cf:
                glob_node = child_node.find('glob')
                if glob_node is None:
                    #case of putting ...... together
                    continue
                type_tag = glob_node.get('tag')
                pos = child_node.get('pos')
                sensekeys = []
                if type_tag == 'un':
                    lemma = glob_node.get('lemma')
                    p = lemma.find('|')
                    if p!=-1:
                        lemma = lemma[p+1:]
                    p = lemma.rfind('%')
                    lemma = lemma[:p]
                else:  
                    id_node = glob_node.find('id')
                    lemma = id_node.get('lemma').replace(' ','_')
                    sensekeys.append(id_node.get('sk'))
                    fixed_pos = get_pos_from_skey(id_node.get('sk'))
                    if fixed_pos is not None:
                        pos = fixed_pos
                text = glob_node.tail
                info_for_cf = [text, lemma, pos, type_tag, sensekeys]
                is_first_cf = False
            else:
                second_text = child_node.text
                text, lemma, pos, type_tag, sensekeys = info_for_cf
                text = text+'_'+second_text
                info_for_cf = [text, lemma, pos, type_tag, sensekeys]
        elif child_node.tag == 'mwf':
            #multiword
            for this_wf_node in child_node:
                text, lemma, pos, sensekeys = process_wf(this_wf_node)
                token_id = '%s_%d' % (this_id,num_token)
                num_token += 1
                new_token = Ctoken(token_id)
                new_token.set_pos(pos)
                new_token.set_text(text)
                if lemma is None: lemma = text.lower()
                new_token.set_lemma(lemma)
                list_of_tokens.append(new_token)
                type_tag_for_token_id[token_id] = this_wf_node.get('tag')
                sense_keys_for_token_id[token_id] = sensekeys
        elif child_node.tag == 'aux':
            #Auxiliar information, do nothing
            pass
    return list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id
        
def generate_instances(this_id, list_tokens, sense_keys_for_token_id, type_tag_for_token_id, data_lexelt, my_wn_reader):
    
    for index_token, token in enumerate(list_tokens):
        token_id = token.get_id()
        gold_lexkeys = sense_keys_for_token_id.get(token_id)
        if gold_lexkeys is not None and len(gold_lexkeys) != 0:
            lemma = token.get_lemma() 
            # The pos is fixed according to the lexkeys
            pos = get_pos_from_skey(gold_lexkeys[0])
            if pos is None:
                #<id id="n00037200_id.5" lemma="purposefully ignored" sk="purposefully_ignored%0:00:00::"/>credit<
                continue
            
            #pos  = token.get_pos()
            
            this_lemma_key = '%s.%s' % (lemma,pos[0].lower())
            if this_lemma_key not in data_lexelt:
                data_lexelt[this_lemma_key] = Clexelt(lemma,pos)
                data_lexelt[this_lemma_key].set_wn_possible_skeys(my_wn_reader)
            
            #For this line we need to have called first to set_wn_possible_skeys
            if True or data_lexelt[this_lemma_key].contains_valid_lexkey(gold_lexkeys):
                new_instance = Cinstance()
                new_instance.set_lemma(lemma)
                new_instance.set_pos(pos)
                new_instance.set_id(token_id)
                new_instance.set_docsrc(this_id)
                new_instance.set_lexkeys(gold_lexkeys)
                new_instance.set_confidence_for_senses({skey: 1.0 for skey in gold_lexkeys})
                new_instance.set_annotation_type(type_tag_for_token_id[token_id])
                new_instance.set_tokens(list_tokens)
                new_instance.set_index_head_list([index_token])
                data_lexelt[this_lemma_key].add_instance(new_instance) 
            else:
                print('Token %s in file %s not valid with lexkeys %s' % (token_id, file_id, str(gold_lexkeys)), file=sys.stderr)     
            
            
            
def process_file(this_file, my_wn_reader, data_lexelt):
    my_tree = etree.parse(this_file)
    n=0
    for synset_node in my_tree.findall('synset'):
        synset_id = synset_node.get('id')
        # Find the gloss node
        gloss_node = None
        for gloss_node in synset_node.findall('gloss'):
            if gloss_node.get('desc') == 'wsd':
                break
            
        ####################################################
        # Process the definition and generate an example   #
        ####################################################
        def_node = gloss_node.find('def')
        
        list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id = process_node(def_node, def_node.get('id'))
        generate_instances(def_node.get('id'), list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id, data_lexelt, my_wn_reader)
        
        
        #The examples
        for example_node in gloss_node.findall('ex'):
            list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id = process_node(example_node, example_node.get('id'))
            generate_instances(example_node.get('id'), list_of_tokens, sense_keys_for_token_id, type_tag_for_token_id, data_lexelt, my_wn_reader)
  
  
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts WordNet gloss corpus to our intermediate ULM format')
    parser.add_argument('-v', action='version', version = '1.0')
    parser.add_argument('-i', dest='input_folder', required=True, help='Path to original wordnet gloss corpus main folder')
    parser.add_argument('-o', dest='output', required=True, help='Output folder')
    parser.add_argument('-wn',dest='path_to_wn', required = True, help='Path to the wordnet root folder')
    args = parser.parse_args()
    
    data_lexelt = {}
    
    #Load the NLTK wordnet reader
    if 'dict' not in args.path_to_wn:
        args.path_to_wn = os.path.join(args.path_to_wn,'dict')
    my_wn_reader = WordNetCorpusReader(args.path_to_wn,None)
    
    files = [('adj.xml','a'), ('adv.xml','r'), ('noun.xml','n'), ('verb.xml','v')]
    #files = [('noun.xml','n')]
    
    for this_file, short_pos in files:
        path_to_file = os.path.join(args.input_folder,'merged', this_file)
        print('Processing %s' % path_to_file)
        process_file(path_to_file, my_wn_reader, data_lexelt)
        #brea
    
    os.mkdir(args.output)
    print('Creating training data...', file=sys.stderr)
    total_instances = 0
    total_lemmas = 0
    for item_key, lexelt in data_lexelt.items():
        
        # First we call to add_sense_info_to_clexelt to create all the senskey related info
        
        if len(lexelt) > 0:
            # First we call to add_sense_info_to_clexelt to create all the senskey related info
            # We do not need to assign the return value to a variable, as the object is passed by reference
            # and it's modified already 
            print('\tLexical item: %s' % item_key, file=sys.stderr)

            add_sense_info_to_clexelt(lexelt, my_wn_reader, debug=False)
            
            
            total_lemmas += 1
            item_key = item_key.replace('/','_').lower()
            
            total_instances += len(lexelt)
            
            #Save the lexelt object 
            output_bin = os.path.join(args.output,item_key+'.bin')
            fd_bin = open(output_bin,'wb')
            pickle.dump(lexelt, fd_bin, protocol=3)
            fd_bin.close()
        

    print('Total number of lemma.pos (unique): %d' % total_lemmas)
    print('Total number of instances: %d' % total_instances)
    
    