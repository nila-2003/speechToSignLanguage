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



import yaml
import json
from pactman import Consumer, Provider

def swagger_to_pact(swagger_file, pact_file):
    with open(swagger_file, 'r') as file:
        swagger_data = yaml.safe_load(file)

    consumer = Consumer('MyConsumer')
    provider = Provider('MyProvider')
    pact = consumer.has_pact_with(provider)

    paths = swagger_data.get('paths', {})
    for path, path_item in paths.items():
        for method, details in path_item.items():
            if method in ['get', 'post', 'put', 'delete', 'patch']:  # HTTP methods
                interaction_name = details.get('summary', f'{method.upper()} {path}')
                pact.upon_receiving(interaction_name)

                # Handle request
                request = {
                    'method': method.upper(),
                    'path': path,
                }

                # Add query parameters for GET requests
                if method.lower() == 'get' and 'parameters' in details:
                    query_params = {}
                    for param in details['parameters']:
                        if param['in'] == 'query':
                            query_params[param['name']] = 'string'  # Simplified, adjust as needed
                    if query_params:
                        request['query'] = query_params

                # Add request body for POST requests
                if method.lower() == 'post' and 'requestBody' in details:
                    content = details['requestBody']['content']
                    if 'application/json' in content:
                        request['headers'] = {'Content-Type': 'application/json'}
                        request['body'] = {}  # Simplified, you might want to generate a more specific body

                pact.with_request(**request)

                # Handle responses
                responses = details.get('responses', {})
                for status_code, response_details in responses.items():
                    response = {
                        'status': int(status_code),
                    }

                    if 'content' in response_details:
                        content = response_details['content']
                        if 'application/json' in content:
                            response['headers'] = {'Content-Type': 'application/json'}
                            response['body'] = {}  # Simplified, you might want to generate a more specific body

                    pact.will_respond_with(**response)

    with open(pact_file, 'w') as file:
        json.dump(pact.json(), file, indent=2)

# Usage
swagger_to_pact('path/to/swagger.yaml', 'path/to/pact.json')
