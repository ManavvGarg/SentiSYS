# SentiSYS 🤖 - Real-time Speech Sentiment Analysis

## Brief:
SentiSYS is a real-time sentiment analysis tool that utilizes EmoRoBERTa, a pre-trained transformer-based model, to perform sentiment analysis on both text and speech inputs. The tool allows users to interact with it using two input methods: microphone and text. Users can speak or type their text, and SentiSYS will analyze the sentiment, providing the emotional tone as:

`admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise and neutral.` 

The tool integrates a robust speech recognition system, enabling users to use their voice to convey sentiments. It also demonstrates the power of transformer-based models in natural language processing tasks, such as sentiment analysis.

**Importance of Sentiment Analysis:**

1. **Customer Insights:** Sentiment analysis helps businesses understand their customers' feelings and opinions about products, services, or brand experiences. Analyzing sentiment from customer feedback, reviews, and social media posts can guide businesses in improving products and services to meet customer needs better.

2. **Brand Reputation Management:** Monitoring sentiment online helps organizations proactively address negative sentiment and maintain a positive brand image. Early detection of negative sentiments allows businesses to respond promptly to customer issues and improve customer satisfaction.

3. **Market Research:** Sentiment analysis allows companies to gather insights from public opinions, identifying emerging trends, and gaining a competitive advantage. Understanding market sentiment helps businesses make data-driven decisions and develop effective marketing strategies.

4. **Social Media Monitoring:** Sentiment analysis is essential for tracking brand mentions and customer feedback on social media platforms. It enables businesses to engage with their audience effectively and address concerns or complaints in real-time.

5. **Risk Management:** In the financial sector, sentiment analysis can be used to gauge market sentiment and assess potential risks. Analyzing sentiments from news articles and social media can help investors make informed decisions and manage risks in their portfolios.

6. **Political Analysis:** Sentiment analysis is used in political campaigns to gauge public opinion about candidates and policies. It helps political parties understand voter sentiments and tailor their strategies accordingly.

7. **Customer Support:** Sentiment analysis can be integrated into customer support systems to automatically classify customer queries based on their sentiment. It allows support teams to prioritize and address critical issues promptly.

## What is EmRoBERTa?
The model used in the SentiSYS project is called "EmoRoBERTa," which is a variant of the RoBERTa model specifically fine-tuned for sentiment analysis tasks. Let's break down the components of this model:

RoBERTa (Robustly Optimized BERT): RoBERTa is a transformer-based language model introduced by Facebook AI. It is built upon the BERT (Bidirectional Encoder Representations from Transformers) architecture. RoBERTa employs a large-scale unsupervised pretraining process, significantly improving the performance of BERT by optimizing hyperparameters, training data size, and training duration.

Tokenizer: The EmoRoBERTa model uses the "RobertaTokenizerFast" from the Hugging Face library. The tokenizer converts raw text into input tensors that can be processed by the EmoRoBERTa model. It splits the text into subwords, adds special tokens, and encodes them in a format suitable for the transformer model.

TFRobertaForSequenceClassification: This is the specific variant of the RoBERTa model used for sentiment analysis. It has a classification head attached to the pre-trained RoBERTa base, allowing it to predict sentiment labels based on input text. In this case, the classification head consists of three output units corresponding to the sentiment labels "positive," "neutral," and "negative."

Pipeline: The Hugging Face library's "pipeline" function allows easy access to pre-trained models for specific tasks. In this case, the model is accessed using the "pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')" command. It provides a convenient interface for performing sentiment analysis directly on the input text.

The EmoRoBERTa model used in SentiSYS has been fine-tuned on a sentiment analysis task to predict emotions or sentiments present in a given piece of text. Fine-tuning involves training the model on a labeled dataset of text samples, where each sample is labeled with its corresponding sentiment (positive, neutral, or negative). By learning from this labeled data, the model becomes capable of associating the input text with one of the three sentiment labels during inference.

Overall, EmoRoBERTa is a powerful transformer-based model that leverages pre-training and fine-tuning techniques to perform sentiment analysis with high accuracy and efficiency. It forms the backbone of the SentiSYS project, enabling real-time sentiment analysis on both text and speech inputs.

## Link to the model below (All rights to their respective owners of external elements/models/modules used in this project):
https://huggingface.co/arpanghoshal/EmoRoBERTa

# Run it locally

1. Clone the repository.
2. Go to root directory and run `pip install -r requirements.txt`.
3. Run `python run.py [mic/text]` Choose whether to use mic or text.
