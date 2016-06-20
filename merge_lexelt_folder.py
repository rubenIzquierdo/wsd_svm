#!/usr/bin/env python

import sys
import os
import glob
import pickle
from copy import deepcopy

if __name__ == '__main__':
    out_folder = sys.argv[-1]
    
    data_lexelt = {}
    os.mkdir(out_folder)
    for folder in sys.argv[1:-1]:
        for bin_file in glob.glob(folder+'/*.bin'):
            print(bin_file)
            fd_in = open(bin_file,'rb')
            lexelt = pickle.load(fd_in)
            fd_in.close()
            
            item_key = lexelt.get_item_key()
            item_key = item_key.replace('/','_')
            
            if item_key not in data_lexelt:
                print('  First found')
                data_lexelt[item_key] = lexelt ###deepcopy(lexelt)
            else:
                print('  Already found')
                print('  Len before: %d' % len(data_lexelt[item_key]))
                data_lexelt[item_key].instances.extend(lexelt.instances)
                print('  Len after: %d' % len(data_lexelt[item_key]))
    
    for item_key, lexelt in data_lexelt.items():
        output_bin = os.path.join(out_folder,item_key+'.bin')
        fd_bin = open(output_bin,'wb')
        pickle.dump(lexelt, fd_bin, protocol=-1)
        fd_bin.close()
    print('Done all')