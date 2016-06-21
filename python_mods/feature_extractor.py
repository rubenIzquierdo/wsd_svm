## Bag of words of tokens in a window

import string
english_stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])


def is_valid_as_lemma(lemma):
    is_valid = True
    if lemma in english_stop_words:
        is_valid = False
    elif len(lemma) == '1' and lemma in string.punctuation:
        is_valid = False
    return is_valid

def transform_lemma(lemma):
    transformed_lemma = lemma.lower()
    if transformed_lemma.isdigit():
        transformed_lemma = '#NUMBER#'

    return transformed_lemma

def extract_bow_lemmas(this_instance, options, features):
    sentence_window = options.get('sentence_window',3)
    if not isinstance(sentence_window, int):   sentence_window = int(sentence_window)
    
    for token in this_instance:
        if abs(this_instance.get_relative_sentence_position_for_token_id(token.get_id())) <= sentence_window:
            this_lemma = token.lemma
            if this_lemma is not None:
                text = transform_lemma(this_lemma)
                if is_valid_as_lemma(text):
                    features.append('BOW#%s' % text)
      

##PoS 
def extract_pos(this_instance, options, features):
    index_of_target = this_instance.get_position_target_token()
    size_context = options.get('window',3)
    if not isinstance(size_context, int):   size_context = int(size_context)
    start_index = max(0,index_of_target-size_context)
    end_index = min(this_instance.get_num_tokens()-1,index_of_target+size_context)
    
    for this_idx in range(start_index,end_index+1):
        relative_position = this_idx - index_of_target
        this_token = this_instance.get_token(this_idx)
        if this_token is not None:
            features.append('POS#%d#%s' % (relative_position,this_token.pos))
        

def extract_collocations(this_instance, options, features):
    #print('CALL TO EXTRACT COLLOCATIONS', this_instance.get_id(),file=sys.stderr)
    collocations = []
    string_collocations = options.get('collocations')
    list_collocations = string_collocations.split(';')
    for this_str_col in list_collocations:
        if len(this_str_col) != 0:
            tokens = this_str_col.split('#')
            collocations.append((int(tokens[1]), int(tokens[2])))
    
    index_of_target = this_instance.get_position_target_token()
    #print('\tToken target',this_instance.get_token(index_of_target),file=sys.stderr)
    for relative_start, relative_end in collocations:
        #print('\tStart: %d  End: %d' % (relative_start,relative_end),file=sys.stderr)
        abs_start = index_of_target + relative_start
        abs_end = index_of_target + relative_end
        tokens = []
        for this_idx in range(abs_start, abs_end+1):
            this_token = this_instance.get_token(this_idx)
            if this_token is not None:      
                tokens.append(this_token.text)
            else:
                tokens = None
                break
        if tokens is not None:
            feat = 'C#%d#%d#%s' % (relative_start,relative_end,'_'.join(tokens))
            #print('\t\tValue', feat,file=sys.stderr)
            features.append(feat)
    
    
            

