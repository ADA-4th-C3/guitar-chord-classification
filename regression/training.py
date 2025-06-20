import pickle
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_path = 'data/training.csv'

# 1. CSV 불러오기
df = pd.read_csv(csv_path)

# 2. 입력(X)과 레이블(y) 분리
X = df.iloc[:, 1:].values  # 12개의 chroma feature
y = df.iloc[:, 0].values   # label

# 3. 레이블을 숫자로 변환
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 4. 학습/검증 데이터 분할
X_train = X
y_train = y_categorical

# 5. 모델 구성
model = Sequential([
    Dense(32, input_shape=(12,), activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. 학습
history = model.fit(X_train, y_train, epochs=50, batch_size=16)

# Plot training accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save plot
plt.tight_layout()
plt.savefig('model/training_history.png')
plt.close()

# 7. 모델 저장
os.makedirs('model', exist_ok=True)
model.save('model/chroma_classifier.h5')

# 8. 레이블 인코더 저장 (선택 사항)
os.makedirs('model', exist_ok=True)
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
