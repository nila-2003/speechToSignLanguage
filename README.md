# Overview
<hr>
The project utilizes a combination of deep learning techniques, including transformers and attention mechanisms, to build a model capable of translating speech to sign language. It involves processing input audio signals, extracting relevant features, and generating corresponding sign language gestures.

# Features

Speech Processing : The system takes input audio signals as speech input.<br>
Deep Learning Model : Utilizes transformer-based architectures for translation tasks.<br>
Sign Language Generation : Translates speech into sign language gestures.<br>
Training and Testing : Includes functionalities for training and testing the model.<br>
Command-Line Interface : Provides a command-line interface for easy interaction.<br>


# Set Up and Installation

1. Clone the repository
```
git clone https://github.com/nila-2003/speechToSignLanguage.git
```
2. navigate to the project directory
```
cd speechToSignLanguage
```
3. Installing the dependencies
```
pip install -r requirements.txt
```
4. Training the model
```
python __main__.py train ./Configs/Base.yaml
```
5. Testing the model
```
python __main__.py test ./Configs/Base.yaml ./Models/best.ckpt
```
