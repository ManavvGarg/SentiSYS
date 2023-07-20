import speech_recognition as sr
import concurrent.futures
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import argparse
import sys
import os
import logging

def load_model():
    tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
    model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
    model_ = pipeline("sentiment-analysis", model="arpanghoshal/EmoRoBERTa")
    return model_


def recognize_and_analyze_audio(audio_data, model, recognizer):
    # Perform speech recognition
    try:
        text = recognizer.recognize_google(audio_data, language="en-US")

        # Print the recognized text
        print("You said:", text)

        # Get sentiment analysis result
        sentiment = model(text)
        print("Sentiment:", sentiment[0]['label'],'\n\n')

        return text

    except sr.UnknownValueError:
        # If the speech cannot be recognized
        print("Could not understand audio")
        return None

    except sr.RequestError as e:
        # If there's an issue with the API or internet connection
        print("Error occurred: {0}".format(e))
        return None

def real_time_speech_sentiment_analysis(model, input_method):
    if input_method == "mic":
        print("Welcome to SentiSYS 🤖\n\n")
        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Open the microphone for capturing audio
        with sr.Microphone() as source:
            print("Listening...")

            # Adjust for ambient noise (optional)
            recognizer.adjust_for_ambient_noise(source)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []

                while True:
                    try:
                        # Listen for audio and convert it to text
                        audio_data = recognizer.listen(source)

                        # Submit the recognition task to the ThreadPoolExecutor
                        future = executor.submit(recognize_and_analyze_audio, audio_data, model, recognizer)
                        futures.append(future)

                        # Check if the user said "end listening"
                        if any(f.result() == "end listening" for f in futures if f.result() is not None):
                            print("Stopped listening.\nThank you for using SentiSYS 🤖\n\n")
                            break

                    except KeyboardInterrupt:
                        # Stop listening when the user presses Ctrl+C
                        print("Stopped listening.")
                        break

            print("Thank you for using SentiSYS 🤖.")
    
    elif input_method == "text":
        print("Welcome to SentiSYS 🤖\n\n")
        while True:
            text = input("Enter your text (type 'end listening' to stop): ")
            if text.lower() == "end listening":
                print("Stopped listening.\nThank you for using SentiSYS 🤖\n\n")
                break
            
            # Get sentiment analysis result
            sentiment = model(text)
            print("Sentiment:", sentiment[0]['label'],'\n\n')
    
    else:
        print("Invalid input method. Please choose 'mic' or 'text'.")

if __name__ == "__main__":
    # Disable TensorFlow and Transformers library logging for levels lower than ERROR
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser(description="Real-time Speech Sentiment Analysis")
    parser.add_argument("input_method", choices=["mic", "text"], help="Choose the input method: 'mic' for microphone, 'text' for typing the text.")
    args = parser.parse_args()

    model = load_model()
    real_time_speech_sentiment_analysis(model, args.input_method)
