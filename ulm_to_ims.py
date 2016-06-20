#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import os
import glob
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts  our intermediate ULM format to IMS format (xml and key)')
    parser.add_argument('-v', action='version', version = '1.0')
    parser.add_argument('-i', dest='input_folder', required=True, help='Path to pickled object files')
    parser.add_argument('-o', dest='output', required=True, help='Output folder')
    
    args = parser.parse_args()
    os.mkdir(args.output)
    if args.output[-1] == '/':
        word_list_file = args.output[:-1]+'.word_list'
    else:
        word_list_file = args.output+'.word_list'
    fd_list = open(word_list_file,'w')
    total_instances = 0
    total_lemmas = 0

    for bin_file in glob.glob(os.path.join(args.input_folder,'*.bin')):
        total_lemmas += 1
        fd_in = open(bin_file,'rb')
        lexelt = pickle.load(fd_in)
        total_instances += len(lexelt)
        
        item_key = lexelt.get_item_key()
        item_key = item_key.replace('/','_')
        
        print('\tLexical item: %s' % item_key, file=sys.stderr)
        fd_list.write('%s\n' % item_key)
        
        output_xml = os.path.join(args.output,item_key+'.train.xml')
        lexelt.save_xml_to_file(output_xml)
        
        key_filename = os.path.join(args.output,item_key+'.train.key')
        lexelt.save_key_to_file(key_filename)
            
        fd_in.close()
    fd_list.close()
    print('List of words in %s' % fd_list.name)
    print('Total number of lemma.pos (unique): %d' % total_lemmas)
    print('Total number of instances: %d' % total_instances)
   
    