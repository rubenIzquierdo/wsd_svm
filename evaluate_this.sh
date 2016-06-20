#!/bin/bash

scorer=semeval-2013-task12-test-data/scorer/scorer2
path_to_key=semeval-2013-task12-test-data/keys/gold/wordnet/wordnet.en.key

out_file=$1

$scorer $out_file $path_to_key