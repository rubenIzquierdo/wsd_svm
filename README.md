# WSD system based on SVM #

This is a WSD based on supervised machine learning: Support Vector Machines. The system is developed in Python and is trained on annotated instances using the [SVMlight library](http://svmlight.joachims.org/) and a set of widely used
features that can be easily extended:
+ Surrounding words
+ Local collocations
+ Local PoS tags

The input for training is a set of python pickle objects, which follow the class definitions in the file `my_data_classes.py`: Clexelt, Cinstance and Ctoken classes. We call this format the ULM format. There are different converters
to the ULM format for different formats.


##Training the system##

You need to provide the following parameters for the `train_lemmas.py` script:
+ List of lemma.pos for which you want to train a classifier
+ Folder to the training data in ULM format
+ XML feature definition file



##Feature definition and extending the feature set##

To define the features to be used during the training, you need to specify them on an XML file. An example of such a file:
```
<?xml version="1.0"?>
<feature_functions>
  <function name="extract_bow_lemmas">
    <arg name="sentence_window">1</arg>
  </function>
  <function name="extract_pos">
    <arg name="window">3</arg>
  </function>
  <function name="extract_collocations">
    <arg name="collocations">C#-2#-2;C#-1#-1;C#1#1;C#2#2;C#-2#-1;C#-1#1;C#1#2;C#-3#-1;C#-2#1;C#-1#2;C#1#3;</arg>
  </function>
</feature_functions>
```

A `<function>` element is defined for every function with the name and a set of parameters with a set of arguments (always with a name and a value). All the feature extractor functions can be found
in the file `python_mods/feature_extractor.py`. The interface of all the feature extractor function is always the same:
```
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
```

It takes always 3 parameters, the Cinstance object for which we want to extract the features, a map called options which contains the parametes created from the `<arg>` elements, and a list of feautures that
will be used to include the new features, which is just a list of string elements. You can add you new feature extractors by including them on this file, following the same schema and creating the XML feature
defitinion to include the new function.

##Classification of new text##

The classification works with a Clexelt object as input, which is basically a list of instances for a specific (lemma,pos). To call to the classifier, you just need to provide a Clexelt object and the path
to the trained models. The feature definition file is not required, as the same one used for training the models will be used.

##Contact##
- Ruben Izquierdo
- Vrije University of Amsterdam
- ruben.izquierdobevia@vu.nl rubensanvi@gmail.com
- http://rubenizquierdobevia.com/

##License##
This repositority and code is distributed under the GNU GENERAL PUBLIC License, the terms can be found in the file `LICENSE` contained in this repository.


