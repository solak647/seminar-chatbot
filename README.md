# Seminar Chatbot - Extracting the personality of a person from a chat conversation

The purpose of this project is to extract the personality of a person using Transformers on labeled text data of the Big Five model.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
conda create --name <env> --file requirements.txt
or
install pandas, numpy, pytorch, transformers, simpletransformers, (optional) ktrain
```

## Usage

Data can be found in the `data` folder. There is 3 datasets:
```
Essays
FriendsPersonality (but finally not used)
myPersonality
```

Source code can be found in the `src` folder.
* Useful code, result used for the paper:
  * `data.py`: contains code to preprocess and format the data
  * `simpletransformers_train.py`: file used to train every model with different parameters. Almost all good results come from this file
  * `simpletransformers.ipynb`: Test of simpletransformers before implementation in the python file
  * `Seminar_Chatbot_Demo.ipynb`: File used for the demo, using the model of cCON and cEXT, on google colab.
  * `main.py`: self-made training using RoBERTa model
  * `predict.py`: predict data using model get from `main.py`
* Test code, finally not used for the paper:
  * `ktrain.ipynb` and `ktrainv2.ipynb`: Test of using the ktrain library
  * `test1.ipynb`, `test2.ipynb`, `v1.ipynb`, `v2.ipynb`: Test file of different model

To test the training, use `simpletransformers_train.py` and uncomment / change the different array at the beginning of the file. To test a model, use `Seminar_Chatbot_Demo.ipynb` and change the folder dir and the name of the model when creating `ClassificationModel`
## Authors

* **Thomas Schaller**

## License

This project is licensed under the MIT License

## Acknowledgments
This is paper is a student work, written for Seminar "Chatbots and Conversational Agents" of the university of Fribourg, under the supervision of Jacky Casas and Prof. Dr. Elena Mugellini, HumanTech Institute, University of Applied Sciences of Western Switzerland, Switzerland, 2020.
