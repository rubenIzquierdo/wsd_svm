#!/usr/bin/env python

import os
import sys
import pickle

from python_mods import SVMClassifier
from nltk.corpus import WordNetCorpusReader

def get_mfs(lemma,pos,wn_reader):
    mfs_key = None
    lemmas = wn_reader.lemmas(lemma,pos)
    if lemmas != None and len(lemmas) != 0:
        mfs_key = lemmas[0].key()
    return mfs_key


if __name__ == '__main__':
    lexelt_filename = sys.argv[1]
    model_folder = sys.argv[2]
    
    path_to_wn_dict = '/home/rbevia/wsd_lfs/resources/WordNet-3.0/dict'
    
    my_wn_reader = WordNetCorpusReader(path_to_wn_dict,None)
    
    fd = open(lexelt_filename,'rb')
    lexelt_data = pickle.load(fd)
    fd.close()

    total_no_class = total_yes_class = 0
    no_classifiers = []
    keys_for_instance_id = {}
    for (lemma,pos), lexelt in lexelt_data.items():
        lemma = lemma.lower()
        short_pos = pos.lower()[0]
        print('Running classification for %s.%s' % (lemma,pos), file=sys.stderr)
        my_classifier = SVMClassifier()
        exists_classifier = my_classifier.start_classifier(lemma,short_pos,model_folder) 
        if exists_classifier:
            total_yes_class += 1
            values_for_instance_id = my_classifier.disambiguate_lexelt(lexelt)
            for this_id, these_values in values_for_instance_id.items():
                keys_for_instance_id[this_id] = these_values
        else:
            total_no_class += 1
            no_classifiers.append((lemma,short_pos))
            #Get the MFS for (lemma,pos) and assign it to all the instance for this lexical item
            print('\tNo classifier for %s %s' % (lemma,short_pos), file = sys.stderr)
            mfs_key = get_mfs(lemma, short_pos, my_wn_reader)
            for instance in lexelt:
                keys_for_instance_id[instance.get_id()] = [(mfs_key,1.0)]                
    
    total_instances = 0
    for this_id, list_key_confidence in sorted(list(keys_for_instance_id.items()), key=lambda t: t[0]):
        file_id = this_id[:this_id.find('.')]
        if len(list_key_confidence) != 0:
            total_instances += 1
            best_key, best_confidence = list_key_confidence[0] 
            print('%s %s %s' % (file_id, this_id, best_key))
    print('Output generated for %d test instances' % total_instances, file=sys.stderr)
    print('Total of lemmas for which there is a trained classifier: %d' % total_yes_class, file=sys.stderr)
    print('Total of lemmas for which there is NO a trained classifier: %d' % total_no_class, file=sys.stderr)
    print('Classifiers not found for:', file=sys.stderr)
    for lemma,pos in no_classifiers:
        print('\t',lemma,pos, file=sys.stderr)
        
    