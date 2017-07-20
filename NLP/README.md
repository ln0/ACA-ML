I left the preprocessing file just as it was. I firstly tried removing all the punctuation and the numbers but, in fact, these decreased the accuracy. 

As for the sentiment.sh file, the process of finding a good set of parameters was tough and a little slow. But there were only a few parameters that actually could increase the precision of the model. Most of the change was made by wordNgrams.

For the nearest future (after it finishes downloading), I plan to use wikipedia pretrained vectors hoping that it will further increase the accuracy. Also, it will be a good idea to use existing libraries to do stemming and lemmatization.
