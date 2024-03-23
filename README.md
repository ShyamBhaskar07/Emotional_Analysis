1. Collecting Dataset from Kaggle
2. Text Preprocessing
     1. Removing noise
     2. Removing Stopwords
     3. Normalization
     4. Lemmatization
     5. Find nearest Dictionary words or use contractions (because we are dealing with chat data)
3. Feature Selection and Extraction using TF-IDF Matrix
4. Training and Feeding the Model (Naive Bayes)
5. Performance Metrics (Accuracy of our Model : 72%)

Our basic emotions:
Anger,happy,sad,fear,surprise,love

We have oversampled dataset towards sad emotion by using sad_dataset.
We have built model in two ways, using ML models(ex. Naive Bayes) and without using ML Models (pure NLP Techniques)


Install the required packaged and run 2 files withML.py and withoutML.py to detect the emotion for a given query
