#!/usr/bin/env python

import os
import shutil

from python_mods import SVMClassifier, FEATURE_FILENAME


def train_classifiers(path_to_bin_files, file_lemmas,model_folder, config_file):
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    os.mkdir(model_folder)
    
    fd = open(file_lemmas)
    for line in fd:
        lemma_pos = line.strip()
        print('Training classifier for %s' % lemma_pos)
        bin_file = '%s/%s.bin' % (path_to_bin_files,lemma_pos)
        print('\tBinary file: %s' % bin_file)
        if os.path.exists(bin_file):
            my_classifier = SVMClassifier()
            my_classifier.train(bin_file,config_file,model_folder)
        else:
            print('\tThere is no bin file for %s, nothing trained' % lemma_pos)
    fd.close()
    
    #Save the selected feature file
    this_feature_filename = os.path.join(model_folder,FEATURE_FILENAME)
    shutil.copy(config_file, this_feature_filename) 
    

if __name__ == '__main__':
    file_with_lemma_pos = 'sem2013.lemma_pos.list'
    config_file = 'feature_files/file_3.xml'
    path_to_bin_files='data/semcor30_ulm/'
    #path_to_bin_files='./data/semcor30_pwgc_ulm'
    #path_to_bin_files='/home/rbevia/wsd_lfs/data/experiments/Bps'
    
    model_folder=path_to_bin_files+'/models'
    
    
    train_classifiers(path_to_bin_files, file_with_lemma_pos,model_folder,config_file)
    print('Models created in %s' % model_folder)