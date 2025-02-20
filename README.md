[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mIB_EBL1)
# Commands

## Virtual environment creation

It's highly recommended to use a virtual environment for this assignment.

Virtual environment creation (you may also use venv):

```{sh}
conda create -n cs5740a1_310 python=3.10
conda activate cs5740a1_310
python -m pip install -r requirements.txt
```

## Train and predict commands

Example command for the original code (subject to change, if additional arguments are added):
Below is the updated commands for MLP model and perceptron model.

```{sh}
# Example for perceptron model that is used for test set results
python perceptron.py -f 'bow+punc+sentilex' -d sst2 -l 0.01 -e 7 --save_test_predictions --error_analysis
python perceptron.py -f 'bow+discgroup+lex_complexity’ -d newsgroups -l 0.01 -e 7

# Example for MLP model that is used for test set results (See multilayer_perceptron.py for more details on the arguments)
python multilayer_perceptron.py --data sst2 --epochs 12 --learning_rate 0.001 --max_vocab_size 10000 --max_length 250 -hd '200,100,50' -a 'relu' -o 'adam' -dr 0.0	
python multilayer_perceptron.py --data newsgroups --epochs 12 --learning_rate 0.001 --max_vocab_size 20000 --max_length 400 -hd '256' -a 'relu' -o 'adam' -dr 0.0 --save_test_predictions --error_analysis
```

## Commands to run unittests

It's recommended to ensure that your code passes the unittests before submitting it.
The commands can be run from the root directory of the project.

```{sh}
pytest
pytest tests/test_perceptron.py
pytest tests/test_multilayer_perceptron.py
```

## Submission

Please do NOT commit any code that changes the following files and directories:

- tests/
- .github/
- pytest.ini

Otherwise, your submission may be flagged by GitHub Classroom autograder.

Please DO commit your code changes in other python files. The autograder will every time you push to the main branch.

Please DO commit your output labels in results/ following the same name and content format. Our leaderboard periodically pulls your outputs and computes accuracy against hidden test labels. <https://github.com/cornell-cs5740-sp25/leaderboards/>
