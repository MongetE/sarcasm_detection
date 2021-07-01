# Sarcasm detection in Newspaper Headlines

This repository makes use of the News Headline Dataset for Sarcasm Detection,
published on Kaggle. Instead of Tweets, it is made of headlines from online
newspapers, such as The Onion, known to be a satirical newspaper, and the
Huffpost, a more traditional newspaper. [More about the dataset](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection).

The associated task is thus to build a model to detect sarcasm.

## Getting started

A notebook with definitions of sarcasm, analysis of the dataset and experiments
with machine learning models can be found in the notebooks folder, or [here](afaire).

### Requirements

- Keras==2.4.3  
- tqdm==4.60.0  
- numpy==1.18.5  
- matplotlib==3.4.1  
- wordcloud==1.8.1  
- pandas==1.2.4  
- spacy==3.0.5  

Required libraries can be installed using `pip install -r requirements.txt`.

*Notes*: SpaCy requires a language model to run. I used `en_core_web_sm` for
speed, but any model following the Penn Tree Bank tags can be used. Spacy models
for English are `en_core_web_sm`,  `en_core_web_md`, `en_core_web_lg` and
`en_core_web_trf`. Install one of them using `python -m spacy download [model_name]`.

Alternatively, you can run `sh setup.sh`. It will create a new virtual
environment (Python 3.8 is required) and download the smallest model
(`en_core_web_sm`).

## References

- Misra, R., Arora, P. (2019). Sarcasm Detection using Hybrid Neural Network.
arXiv preprint arXiv:1908.07414
- Misra, R. Grover, J. (2021). Sculpting Data for ML: The first act of Machine
Learning. 978-0-578-83125-1
