## Chord Classification
This project classifies guitar chord sounds using [chroma features](https://en.wikipedia.org/wiki/Chroma_feature) with two models: CNN and Regression.

## Chord Chroma Features
![chrome features-min](https://github.com/user-attachments/assets/b90ef970-1c8c-4410-b835-9a2cc6eda34f)

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

## Regression Model
```
cd regression && make training && make run
```

https://github.com/user-attachments/assets/659f807b-421c-40d8-9ecb-0506418ec5ab


## CNN Model
```
cd cnn && make training && make run
```
https://github.com/user-attachments/assets/ad17ac67-3311-4cb4-91ad-631ff1d4ae83