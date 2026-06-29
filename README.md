# Sentimental-analysis
# Financial-Sentiment-Analysis - Invest smarter with sentiments

<h3>Machine Learning</h3>
<p>A Machine Learning Project to predict the sentiments (positive, negative, or neutral) on real-time financial news headlines through unsupervised textual data of big tech companies using NLTK, K-Means Clustering Algorithm, and LSTM RNN and thus performing the mismatch or divergence between both the models.</p>
<h3>Come, Visit the Site for a Better Choice of Stock Investment!&#127881;</h3>

# Tech Stack<br>
<details>
  <summary>Data Collection</summary>  <br>
  
  > Python's BeautifulSoup module is used to scrape real-time financial news data from finviz.
  > Following that, the headlines are cleaned and prepared for analysis.

</details>

<details>
  <summary>Preprocessing</summary>  <br>
  
  > Lemmatization and stopword elimination are two preprocessing techniques used on the headlines to get the data ready for sentiment analysis.

</details>

  # Analysis Techniques<br>
<details>
<summary>KMeans Clustering</summary>  <br>

  > CountVectorizer is used to create the bag of words model.

  > used to group similar data points together in a process known as clustering. There are three clusters for sentiments like <b>Positive, Negative, and Neutral</b>.


</details>

<details>
<summary>Vader lexicon NLTK</summary>  <br>
  
  > A suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It supports classification, tokenization, parsing, and semantic reasoning functionalities.

  > Sentiment analysis is done using the NLTK library, more especially the VADER sentiment analyzer. Headlines are categorised as good, negative, or neutral based on the computation of sentiment scores.

  > Results are visualized using matplotlib.
  <img src="https://github.com/Nehagarg816/Financial-Sentiment-Analysis-ML/assets/111566521/9c5ce2ef-a0fa-41d8-85df-4c0264501cec" alt="image" width=500/>

</details>

<details>
<summary>LSTM RNN(Recurrent Neural Network)</summary>  <br>

  > A type of RNN with higher memory power to remember the outputs of each node for a more extended period to produce the outcome for the next node efficiently.
  
  > The preprocessed data is used to train an LSTM RNN model for more sophisticated sentiment analysis. Based on the patterns that were learned from the text input through K-Means clustering, the model classifies attitudes as positive, negative, or neutral, thereby confirming its predictions.

  > Results for the RNN model are visualized using a pie chart in Matplotlib.
  <img src="https://github.com/Nehagarg816/Financial-Sentiment-Analysis-ML/assets/111566521/8f4d2362-8172-478d-bd59-7b953d2675cb" alt="image" width=500>

</details>

<h3>Mismatch Between both models</h3>
<details>
  <summary>KMeans v/s RNN</summary>
  <img src="https://github.com/Nehagarg816/Financial-Sentiment-Analysis-ML/assets/111566521/933e0949-9b49-41b8-89b1-cdd3b41a7c33" alt="image" width=500>
</details>


