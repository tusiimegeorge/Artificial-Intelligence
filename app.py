import streamlit as st
from fastai.text.all import *

class TextClassifierApp:
    """
    A Streamlit app for classifying Swahili news articles using ULMFiT.

    Attributes:
        learn (Learner): The FastAI learner object for the text classifier.

    Methods:
        load_model(): Loads the pre-trained model with a spinner indicating the loading process.
        predict(text: str) -> str: Predicts the class of the given text.
        run(): Runs the Streamlit app, providing the user interface for text classification.
    """
    def __init__(self):
        """
        Initializes the TextClassifierApp by loading the model.
        """
        self.learn = None
        self.load_model()

    @st.cache_resource
    def load_model(self):
        """
        Loads the pre-trained model and shows a spinner during the loading process.

        Returns:
            None
        """
        with st.spinner('Model is being loaded...'):
            self.learn = load_learner('models/text_classifier_model.pkl')

    def predict(self, text):
        """
        Predicts the class of the given text.

        Args:
            text (str): The text to classify.

        Returns:
            str: The predicted class.
        """
        pred_class, pred_idx, outputs = self.learn.predict(text)
        return pred_class

    def run(self):
        """
        Runs the Streamlit app, providing the user interface for text classification.

        Returns:
            None
        """
        st.title('ULMFiT Swahili News Article Classifier')

        st.markdown("""
        ULMFiT (Universal Language Model Fine-tuning) is an effective transfer learning method for NLP tasks.
        """)

        user_text = st.text_area('Enter text for classification')

        if st.button('Classify'):
            if user_text:
                pred_class = self.predict(user_text)
                st.write(f"Input text belongs to: {pred_class}")
            else:
                st.write("Please enter text to classify.")

if __name__ == '__main__':
    app = TextClassifierApp()
    app.run()
