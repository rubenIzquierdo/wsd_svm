#!/usr/bin/env python

import sys
import os
import pickle
import subprocess

from . import feature_extractor
from .my_names import *
from collections import defaultdict
from xml.etree import ElementTree
import tempfile


class SVMClassifier:
    def __init__(self):
        self.lemma = None
        self.pos = None
        self.list_feature_extractors = None
        self.index_features = {}
        self.svm_class_for_key = {}
        self.main_folder = ''
        
    def start_classifier(self,lemma,pos,folder):
        self.lemma = lemma.lower()
        self.__set_normalised_pos(pos)
        self.main_folder = folder
        model_file = self.__get_model_filename()
        if os.path.exists(model_file):
            self.__load_index_features()
            self.__load_index_classes()
            self.load_feature_extractors(self.__get_feature_config_filename())
            return True
        else:
            return False
        
        
        
    def __set_normalised_pos(self,this_pos):
        self.pos = this_pos.lower()[0]
        
    def __get_feature_config_filename(self):
        feature_config_filename = os.path.join(self.main_folder,FEATURE_FILENAME)
        return feature_config_filename
    
    def __get_training_filename__(self):
        training_filename = os.path.join(self.main_folder,'%s.%s.%s' % (self.lemma, self.pos, TRAINING_BASE_FILENAME))
        return training_filename
    
    def __get_index_feature_filename(self):
        index_feature_filename = os.path.join(self.main_folder,'%s.%s.%s' % (self.lemma, self.pos, INDEX__FILENAME))
        return index_feature_filename
    
    def __get_index_class_filename(self):
        index_class_filename = os.path.join(self.main_folder,'%s.%s.%s' % (self.lemma, self.pos, INDEX_CLASS_FILENAME))
        return index_class_filename
    
    def __get_model_filename(self):
        model_filename = os.path.join(self.main_folder,'%s.%s.%s' % (self.lemma, self.pos, MODEL_FILENAME))
        return model_filename
    
    
    def load_feature_extractors(self,_file):
        self.list_feature_extractors = []
        tree_s = ElementTree.parse(_file)
        for function in tree_s.findall('function'):
            options = {}    
            for arg in function.findall('arg'):
                options[arg.get('name')] = arg.text
                self.list_feature_extractors.append((function.get('name'), options))

                
    def extract_features(self, this_instance):
        list_string_features = []
        for this_function_name, these_options in self.list_feature_extractors:
            this_function = getattr(feature_extractor, this_function_name)
            this_function(this_instance, these_options, list_string_features)
        return list_string_features


    def encode(self, list_features, update_index):
        #List of s is a list of strings
        map_feat = defaultdict(int)
        for string_feat in list_features:
            num_feat = self.index_features.get(string_feat,None)
            if num_feat is None:
                if update_index:
                    num_feat = len(self.index_features)+1
                    self.index_features[string_feat] = num_feat
            
            if num_feat is not None:
                map_feat[num_feat] += 1
                
        if len(map_feat) != 0:
            vector_feat = sorted(map_feat.items(), key=lambda t: t[0])
        else:
            vector_feat = []
        return vector_feat
        
        
    
    def train(self,bin_file, features_file, model_folder):
        self.main_folder = model_folder
        if not os.path.exists(self.main_folder):
            os.mkdir(self.main_folder)
        
        ###########################
        #  Load the instances 
        ###########################
        self.load_feature_extractors(features_file)
        
        
        
        ###########################
        #  Load the instances 
        ###########################
        fd = open(bin_file,'rb')
        
        #This is a list of Cinstance objects
        lexelt = pickle.load(fd)
        fd.close()
        
        self.lemma = lexelt.get_lemma().lower()
        self.__set_normalised_pos(lexelt.get_pos())
        print('\tTraining classifier for %s.%s' %(self.lemma,self.pos))
        
        ###########################
        ###########################

        ###########################
        #  Load the instances 
        ###########################
        
        features_for_instance_id = {}
        keys_for_instance_id = {}
        feature_frequency = defaultdict(int)
        total_features = 0
        instance_ids_in_order = []
        for this_instance in lexelt:
            instance_ids_in_order.append(this_instance.get_id())
            features_for_instance_id[this_instance.get_id()] = self.extract_features(this_instance)
            ###print(this_instance.get_id(),features_for_instance_id[this_instance.get_id()])
            for f in features_for_instance_id[this_instance.get_id()]:
                feature_frequency[f] += 1
            total_features += len(features_for_instance_id[this_instance.get_id()])
            keys_for_instance_id[this_instance.get_id()] = this_instance.get_lexkeys()
        print('\tTotal instances: %d' % len(keys_for_instance_id))
        print('\tTotal features: %d' % total_features)

        #for f, v in sorted(list(_frequency.items()), key=lambda t: -t[1]):
        #    print('%s -> %d' % (f,v))
        ###########################
        ###########################
        
        ###########################
        #  Encode the instances
        #  Generates the index
        #  Generate the training file
        ###########################
        training_filename = self.__get_training_filename__()
        fd_training = open(training_filename,'w')
        
        for instance_id in instance_ids_in_order:
            list_string_features = features_for_instance_id[instance_id]
            keys = list(keys_for_instance_id[instance_id])
            this_class = keys[0]
            if this_class in self.svm_class_for_key:
                this_svm_class = self.svm_class_for_key[this_class]
            else:
                this_svm_class = len(self.svm_class_for_key) + 1  #First is 1
                self.svm_class_for_key[this_class] = this_svm_class
                
            #only one key
            fd_training.write('%s' % this_svm_class)
            int_features_and_frequency = self.encode(list_string_features, update_index=True)
            for int_feat, freq in int_features_and_frequency:
                fd_training.write(' %d:%d' % (int_feat, 1))
            fd_training.write('\n')
        fd_training.close()
        
        ###########################
        #  Train the model
        ###########################
        model_filename = self.__get_model_filename()

        training_cmd = []
        training_cmd.append(SVM_LEARN)
        training_cmd.append('-c')
        training_cmd.append('1')
        #training_cmd.append('-w')
        #training_cmd.append('4')        
        training_cmd.append(training_filename)
        training_cmd.append(model_filename)
        log_training = model_filename+".log"
        fd_log = open(log_training,'w')
        training_code = subprocess.check_call(training_cmd, stdout=fd_log)
        fd_log.close()
        print('\tTraining done with exit code: %d' % training_code)
        print('\tLog training file in %s' % log_training)
        
        
        ###########################
        #  Save the indexes
        ###########################
        
        index_feature_filename = self.__get_index_feature_filename()
        fd_index_feat = open(index_feature_filename,'wb')
        pickle.dump(self.index_features, fd_index_feat, protocol=-1)
        fd_index_feat.close()
        
        index_class_filename = self.__get_index_class_filename()
        fd_index_class = open(index_class_filename,'wb')
        pickle.dump(self.svm_class_for_key, fd_index_class, protocol=-1)
        fd_index_class.close()
        
    def disambiguate_lexelt(self,this_lexelt):
        values_for_instance_id = {}
        # Create the testing file
        fd_testing = tempfile.NamedTemporaryFile('w',delete=False)
        ids_in_order = []
        for this_instance in this_lexelt:
            ids_in_order.append(this_instance.get_id())
            fd_testing.write('1 ')
            string_features = self.extract_features(this_instance)
            int_features_and_frequency = self.encode(string_features, update_index=False)
            for int_feat, freq in int_features_and_frequency:
                fd_testing.write(' %d:%d' % (int_feat, 1))
            fd_testing.write('\n')
        fd_testing.close()
        
        # Run the classification
        ### usage: svm_struct_classify [options] example_file model_file output_file
        model_filename = self.__get_model_filename()
        output_filename = tempfile.mktemp()
        
        testing_cmd = []
        testing_cmd.append(SVM_CLASSIFY)
        testing_cmd.append(fd_testing.name)
        testing_cmd.append(model_filename)
        testing_cmd.append(output_filename)
        evaluation_code = subprocess.check_call(testing_cmd, stdout=subprocess.DEVNULL)
        print('\tClassification done with code %d' % evaluation_code, file=sys.stderr)
        
        # Reverse the dictionary
        wn_class_for_svm_class = {}
        for wn_class, svm_class in self.svm_class_for_key.items():
            wn_class_for_svm_class[svm_class] = wn_class
            
            
        fd_out = open(output_filename)
        for idx_instance, line in enumerate(fd_out):
            instance_id = ids_in_order[idx_instance]
            these_values = []
            
            '''
            #### This was for liblinear 
            wn_class = wn_class_for_svm_class[int(line.strip())]
            
            these_values.append((wn_class,1.0))
            values_for_instance_id[instance_id] = these_values
            continue
            ############
            '''
            
            tokens = line.strip().split(' ')
            
            for idx_class, value in enumerate(tokens[1:], 1): #Starts from 1 the first class
                wn_class = wn_class_for_svm_class[idx_class]
                these_values.append((wn_class,float(value)))
            these_values.sort(key=lambda t: -t[1])
            values_for_instance_id[instance_id] = these_values
        fd_out.close()
        
        os.remove(fd_testing.name)  
        os.remove(output_filename)      
        return values_for_instance_id
        
        
            
    def __load_index_features(self):
        index_feature_filename = self.__get_index_feature_filename()
        fd_index_feat = open(index_feature_filename,'rb')
        self.index_features = pickle.load(fd_index_feat)
        fd_index_feat.close()
    
    def __load_index_classes(self):
        index_class_filename = self.__get_index_class_filename()
        fd_index_class = open(index_class_filename,'rb')
        self.svm_class_for_key = pickle.load(fd_index_class)
        fd_index_class.close()
        
        
 
        
        
        
        
        
if __name__ == '__main__':
    my_classifier = SVMClassifier()
    my_classifier.train('/exp2/rbevia/wsd_lfs/data/semcor30_ulm_v2/house.n.bin','/home/rbevia/wsd_svm/_files/file_1.xml', 'folder_test')
