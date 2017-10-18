# spamnet

## Introduction

In our modern society, text intelligence has become a crucial tool used in a broad number of applications. Even if our language is a rather complex system of rules and facts, current algorithms are able to extract an excellent amount of useful information out of such kind of data. The key concept, allowing us to use those techniques in a wide range of tasks like security analysis, information extraction or the construction of complex classification system is commonly summarized under the term "big data": Not only our calculation power gives us the capabilities to find, process and interpret the gigantic amount of data. For the first time in the history of the humanity, we have these facts actually available in fractions of seconds;  our knowledge is not longer stored in libraries used by few privileged, but completely available.

But without a good algorithm, even this available data are just wasted bytes. In order to gain real results, the algorithms we may use are crucial. With the boom of Neural Networks in the last decades, new architectures were developed especially optimized for Text intelligence. One of those are Recurrent Neural Networks, which have the unique advantage of using not only the plain features as the foundation of there prediction but even their sequence. The aim of this paper is the critical evaluation of such models in comparison to other Python-based state-of-the-art technology. The benchmark of choice is the commonly used task of classifying comments according to their nature as real expressions of opinions or mostly unwanted messages like advertisements, often referred to as "Spam". Besides a focus on an optimal structure, this analysis will focus on the performance of the network in relation to the required preprocessing to conclude at the end, if the performance of recurrent neural networks is significantly better than comparable algorithms.

## Method

### Datasets

Every data mining algorithm is only that good as the data, which is feed into it. Therefore, in order to train that actual model for the prediction of Spam, a collection of X comment is used. These samples were extracted and hand-annotated by X using five highly popular videos on "YouTube" and contain beside the author, the date of writing and an Id the actual comment and label. Other than comparable dataset like X, all the X spam comments and Y non-spam comments are written in English. While this fact implies already a loss of potential useable information it simplifies the actual processing of the data dramatically: Beside some emoticons, the texts do not utilize advanced and therefore hardly handleable Unicode features. Some comments contain additional makeup data for formatting purposes in form of HTML tags. The data is divided into five comma-separated CSV files, where quotes are used to escape comma in the actual text.
Due to its nature, trained data mining models should have the ability to correctly classify data which vary from those they were trained on. A common problem is the so-called "overfitting": In this case, the neural networks learn not context behind the actual features but the features itself. A separation of training and testing data is therefore essential; in this paper, a ratio of 70% training and 30% testing data was used. Moreover, most machine learning algorithm contains a partly random element and may differ highly between different testing sets. To find a reliable outcome, all the following experiments were applied at least three times; the variance of the actual results is stated beside the used evaluation metric. Wherever applicable, a 3-fold cross validation where used, guaranteeing that the algorithm trains all the time on different data. If such a cross-validation was not usable due to given constraints, the data was at least shuffled before each trial. At the end, the following datasets were used:
- ...

This separation provides not only information of the actual generalizability of a model but also about the differences in performance with a different amount of samples.

### Preprocessing

Other than many other data mining tasks, Text intelligence is applied on unstructured data rather than structured. It is essential to understand the difference between both approaches and the steps needed to generated knowledge out of them: Like in the processing of images, the actual data is not simply represented by an ordered list of datasets containing a specific information in a precisely defined datatype which is required by the algorithm. Instead, it is a bunch of different words which are most likely highly different from the deterministic and unique nature numbers are. Even worse, the type of text itself is more comparable with Twitter tweets rather than reliable sources like or medical reports, which are often used as the foundation of natural language processing tasks. Extensive preprocessing is, therefore, necessary to standardize the data in a format a computer may understand it and gain good results in terms of precision and recall of the model. Given these constraints, before the actual evaluation of the algorithms, a data pipeline was developed utilizing the well known "sklearn" and "nltk" packages.

#### Step 1: Loading and Tokenization

In order to allow a flexible approach in the choice of the data sources, the above-mentioned datasets were constructed as instances of an abstract class "DataSource", encapsulating common logic in the actual parsing of the file and providing an convenient way to access training and test datasets represented as tuples of the actual comment and a boolean describing its type. The inclusion of other provided metadata like the name of the author or time of the comment shows no significant improvements in the classification in smaller experiments was therefore excluded.
After the actual process of loading the CSV files, the Unicode characters of no further interest and the HTML tags were removed in order to prepare the data for the following steps. While the actual tokenization, the conversion from a single string into a list of words, requires in the common case just a splitting at whitespace, the noisy nature of the comments requires a more elaborate strategy in the generation of further usable entities. In order to be able to process typical internet slang like URLs, emoticons and simple ASCII art like arrows, the TwitterTokenizer of the NLTK package was used. The resulting list of entities was afterward used as the starting point for further processing.

#### Step 2: Preprocessing

For a convenient integration into the Scikit-learn workflow, the different preprocessors were developed as subclasses of an overall "Preprocessor" subclass. Afterwards, the different algorithms were integrated into a single Scikit-learn "Transformer", where they were individually activatable using boolean switches. Using this architecture, the different preprocessors were accessible as hyperparameters for the optimization using a grid-search in the following steps in order to find their optimal combination. In the following, the complete preprocessing pipeline of the data is described.

Preprocessing 1: Standardization
In the first preprocessing step, some kind of standardization was performed to reduce the amount of lexically different but semantical equal data significantly. This process is important for two reasons: On the one hand, it extends the raw input with semantic information usable for the classification algorithm. As an example, many spam comments contain URL or YouTube channels being advertised. While humans have the ability to easily identify these entities through the presence of the characteristic parts "http" and (often) ".com", for an algorithm every of these parts looks completely different. On the other hand, further preprocessing algorithms may be irritated by a number of special chars used in the "noisy" way we humans use social media. Therefore, in this step, utilizing different regular expressions, URLs, numbers, and emoticons were replaced by a fixed constant. Afterwards, any other special character beside the normal punctuation got removed.

Preprocessing 2: Slang removal
In this step, the wide amount of slang words and spelling mistakes got reduced. For doing that, first multiple times repeated character like in "loooooove" got removed. Due to the fact that almost any English term contains maximal two times the same letters behind each other, the overall amount of proper English words were untouched by replacing only three or more occurrences. In a next step, common slang words and spelling mistakes gets removed. The basis of this was the collection generated by X (...). Even if this list was generated automatically and is not fully trustworthy; this step reduced again the amount of variance in the text.

Preprocessing 3: POS tagging and lemmatization
In the next step, all the words in the now rather cleaned sentences got reduced to their common form. The algorithm of choice was, in this case, the Stanford Lemmatizer. Other than the rather dumb stemmer, the algorithm utilized additional knowledge about the actual word it is applied to improve the actual result. Therefore, the sentences where POS-tagged according to their structure utilizing the Penrose corpus as their source of data. After the classification by the Perceptron Tagger provided by the NLTK package, this knowledge was used to lemmatize the comment word by word.

Preprocessing 4: Stop word removal
In this step, every word known for its minimal information value got removed after their successful consideration for the actual sentence structure in the POS-tagging step before. While words like "to" may be important and rather common in our natural way of communication, they do not carry any significant data for the actual classification of the elements. The deletion is, therefore, an easy and efficient way to reduce again the number of words the algorithms have to handle. 

Preprocessing 4: Stemming
Often, stemming is used as a less complicated alternative to a lemmatization: Instead of reducing a word to a valid English term for a phrase, its reduction to a word stem is normally easier to implement. Nevertheless, the usage besides the lemmatization has a reason: The combination of both approaches is used for an optimal reduction of features being misspelled or otherwise wrongly classified. In the end of this step, the comments consisted out of a list of not longer syntactic correct words.

Preprocessing 5: Lowercase
In the next step, the capitalization of all words was standardized to a complete lowercase variant. After the tagging and stemming, the type of the first character did not carry any interesting meaning. By lowercasing all the characters, equal words perceived by a human were also equal in the byte representation, the text algorithm are applied on.

Preprocessing 6: Finalization
In the final step, the last remaining punctation necessary in the former steps like dots, commas, and hyphens got removed. Only the actual lowercase characters remained and represented the source of features used in the vectorization and classification of the following algorithms.

#### Step 3: Vectorization

Neither Recurrent Neural Networks, not other data mining algorithm works on real words. While these are convenient to use for humans and allow us to communicate in a simple manner, the different encodings and size would result in a rather inefficient way of handling it. Even for us humans, it seems understandable, that a representation where each distinct word matches a different concept is an optimized solution. Therefore, every machine learning algorithm used in text processing needs a function which is able to transform the words in a unique manner into a binary representation. Classical way of handling this is a classical, hash function: \forall W:V(W) \rightarrow N \land \forall W2 \ne W: V(W2) \ne V(W).

The actual implementation of the function may differ and is an implementation detail: Beside a classical "Bag of Word"-approach, where each word is simply represented by a unique number getting incremented through its calculation, fast hash algorithms with low potential of hash collision may be used for the same task, eliminating the necessity of storing an actual list of already founded words. 

In the following experiment, after the vectorization of the preprocessed words, two different formats were used to store in that way generated data. Reason for this decision was the fact that RNNs requires a different input than other algorithms: While those just take a vector of values with a fixed length matching the number of words to be considered sequential, other machine learning algorithms get a (dense) vector with the length of all words as input, where an element unequal zero marks this word as "used". The  Python library scikit-learn provides an already developed class for the latter task, for the RNN, a custom Bag of word matching the above-stated requirements was developed.

### Algorithms

Recurrent neural networks are, like the name already included, a specialized family of artificial neural networks. With the increasing popularity of feed-forward neural networks in the last decades due to their often proven excellent performance in a wide range of problems, RNNs were conducted as a more powerful tool especially for tasks involving the use of sequences of data rather than only data itself. To understand their specialty is is rather important to understand the difference between them and classical feed-forward neuronal networks. While former generates a fixed output vector out of a fixed input vector and a fixed number of computational steps, RNNs work on sequences of these vectors. Foundation of this is their ability to have an interior state which gets adapted between different samples and allows therefore further consideration of spatial frequency and common pattern. This ability was successfully used in speech recognition, machine translation and even the generation of text. For a rough evaluation of the performance of a "typical" RNN in the classification of Spam, a rather simple architecture with only a single hidden layer was used. In order to reduce the source of possible mistakes and improve the performance of the network, the python toolkit Keras was utilized; using the highly optimized Tensorflow framework as its backend. 

Unlike the other data mining tools, the input was not handled as a dense vector with a length of the complete number of words, but as a 16 number long sequence of the actual word indices. Longer sentences were truncated, shorter comments padded with zeros. In the first Embedding layer, each integer of this input vector was itself turned into a dense vector. After this transformation, an optimal Dropout layer was added before the actual artificial neurons: RNNs tend to easily overfit the data, by adding a certain amount of random noise it is possible to reduce this danger and improve the prediction results; the actual amount given in percentage was designed to be one of the hyperparameters. 
The following layer with actual RNN neurons afterward and their numbers were designed in a flexible manner, too: Besides the classical, simple RNN neurons described above, LSTM and GRU neurons were evaluated in this experiment. Both were designed to face the "long-term dependency problem": While RNNs are able to easily find dependencies between near elements, further context commonly needed in language processing tasks are not possible. By adding additional complexity to the actual neurons, Hochreiter \& Schmidhuber (1997) created LSTMs which may deal with that kind of "long-time memory". GRUs, published recently by Y, target the same problem but try to handle it in a less complex way. Even if studies show both types of advanced cells perform rather similar, it seems worth to explore if the GRUs may be better in dealing with the rather small amount of samples provided in the experiment.
After the following Dropout layer, the results are directly mapped to a single neuron of a Dense layer used as output. Due to the nature of the binary classification, the sigmoid function is used as the activation function. For the Backpropagation as part of the training process and under respect of the binary output of the dense layer, Keras' "binary_crossentropy" and beside this the "Adam" optimizer with the default parameter described in its paper is used.

Beside the neuronal network architecture, two other families of classifier were used to generate a valid ground truth for the evaluation. 
Random forests are generally one of the best performing algorithms in the family of decision trees. Inspired by the human reasoning process, those trees classify upon a sequential ordered number of (often binary) decisions. While these approach is rather powerfull, it has in general a problem with overfitting. Random forests try to minimize the problem by evaluating not a single, but a specific amount of individual trees and generate its prediction according to the predictions generated by the different trees.
Naive Bayes is the the third and last utilized algorithm. Even if it assumes conditional independence between the features, these family of algorithm performs surprisingly well on the classification tasks. Different members like Multinomial Naive Bayes, binarized Multinomial Naive Bayes and Bernoulli Naive Bayes focus on different types of features; in the following, Multinomial Naive Bayes is used according to its focus on multiple occurrences of a word for a classification.

## Results

Before the actual evaluation of the classifier in terms of their performance begun, a run without any preprocessing was performed. All the algorithms are highly dependent on their hyperparameters, therefore a grid search was used to find the global optimum. For doing that, the classifier was three times trained and evaluated with all possible combinations of the parameters on as different training and testing sets as possible. By the repetition and the following averaging, the reliability of the results is optimized. As the measurement for further analysis, the F-Measure was used. Unlike the accuracy considering only the rate of detected true-positives, the alternative reflects precision and recall and is therefore far better suitable for contexts, where the ratio between the classes is not exactly equal. In the underlying table, the averaged F-scores, precisions and recalls of the best performing model on different datasets are stated. It is important to mention that all the averages were calculated after the individual calculations, therefore the averaged precision and recall may not result in the presented F-Score value. The variance describes the variance in the F-Score between the three trials.

| Type  | Dataset       | F-Score | Variance | Precision | Recall |
| ----- | ------------- | ------- | -------- | --------- | ------ |
| Bayes | Single        | 81.4%   | +/- 5.8% | 83.6%     | 82.9%  |
|       | Splitted      | 76.4%   | -        | 71.7%     | 73.7%  |
|       | Multi         | 89.8%   | +/- 2.1% | 89.5%     | 89.4%  |
|       | MultiSplitted | 84.5%   | -        | 84.7%     | 83.4%  |
| RF    | Single        | 81.6%   | +/- 1.9% | 86.1%     | 82.9%  |
|       | Splitted      | 68%     | +/- 0.5% | 64.3%     | 65.7%  |
|       | Multi         | 89%     | +/- 1.8% | 90.3%     | 89.7%  |
|       | MultiSplitted | 87.1%   | +/- 0.2% | 86.8%     | 86.8%  |
| RNN   | Single        | 77.7%   | +/- 8.7% | 77.1%     | 76.9%  |
|       | Splitted      | 67.9%   | +/- 5%   | 67.8%     | 68%    |
|       | Multi         | 88.9%   | +/- 5.6% | 88.6%     | 88.4%  |
|       | Multisplitted | 81.1%   | +/- 6.3% | 80.5%     | 78.9%  |

With an average F-score over all datasets from 84.35%, the Random Forest algorithm performed in this instance of the experiment slighly better than the Naive Bayes approach with its 83.025%; the RNN followed at least with 78.27% performance. Nevertehless, it would be wrong to speak of a clear winner: Adding the measured variance from up to 9% in the calculation, the maximal difference between the performance of the algorithms is far big enough to claim a statistical significant difference between them. Instead, one should claim that at least Random Forest and Naive Bayes performed rather similar, the performance of the RNN is slighly infirior.

While these results claims about the performance in general, a finer-grained interpretation of the difference datasets is more worthy:

When being applied on a single dataset, the Random Forest algorithm performed at best. These result resutls not only from the minimal better performance of 81.6% in comparisment to the 81.4% derived by Naive Bayes. Moreover, it is resulting from the fact that the actual variance when applied on the different fold was almost three times as small and therefore far more stable that Naive Bayes. In general, it seams that the Random forest was very good in detecting real true positives, as shown by the high precision, but not as good in detecting all of them (measured by the recall). In absolute terms, the RNN performed worst at this dataset. Not only the rather small F-score shows that, but rather the rather high variance. It seems, that it was not possible on the small dataset to generate a model deliviring good results.

On the splitted dataset, where the model was first trained on the dataset of spam under one window and afterwards applied on the test sets of other ones, the actually results differs drastically. This task is interesting under the premise, that it actual measure the generalizibility of a model not only on former unseen data of the same source but also on a new one - and therefore crutial for real world applience. Instead of the cross validation, due to the constraints towards the task the training and testing was performed on the same two, randomized datasets. This time, Naive Bayes performed constantly and clearly best. The determinism of the performance can be seen on the completely missing variance, one may therefore conclude that the implementation of Multinomal Naive Bayes performes independenlty of the actual order of the samples. The Random Forest algorithm and the RNN performed with 68% rather similar. The gap between the first place on the first place and the last one on the second dataset seems to show that the Random forest has a problem with overfitting the data: The extracted features used for its classification seems to be only usable on the actual training and not on the testing set.

On the 3-fold cross validated dataset with all available samples, all algorithms was capable of showing their best performance around 89%. In general, the variance of all models was reduced which may be a hint that the increased amount of data leads to better generalizing models. While the actual improvement for the Random Forest under the context of its rather low variance was quite suble, both Naive Bayes and the RNN profit clearly.

On the last dataset, where the algorithms leared on three datasets and were tested on a fourth one, again Random forest performed best. Woth mentioned is not only its in comparisment to the other algorithm high F-Score of 87.1%, but the rather suble difference in classfing performance between the fourth and the third dataset. While the drop between dataset one and two was dramatically, the increased amount of data seems to signifiant reduce the problem of overfitting; a result with fits the common opinion between data scientists. The difference in performance of the Naive Bayes classifier were comparable with the one between the first pair of datasets; again, the appliance on the same randomized datasets leads to the same performance. The RNN performed not that sucessfull, but was capable of compensating the drop of performance until a certain extend with the additional data, that the differences were not that high like between the first two datasets.
