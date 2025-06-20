## Chord Classification with Regression
This is a model that classifies guitar chord sounds based on [chroma features](https://en.wikipedia.org/wiki/Chroma_feature) and Regression.

## Chord Chroma Features
![chrome features-min](https://github.com/user-attachments/assets/b90ef970-1c8c-4410-b835-9a2cc6eda34f)

## Demo
https://github.com/user-attachments/assets/659f807b-421c-40d8-9ecb-0506418ec5ab


## How to Install
1. install python 3.10
```
brew install python@3.10
```

1. create virtual environment
```
python3.10 -m venv .venv
```

1. activate virtual environment
```
source .venv/bin/activate
```

1. install packages
```
pip install -r requirements.txt
```

## Training
```
make training
```

## Run
```
make run
```