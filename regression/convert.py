import coremltools as ct
from tensorflow.keras.models import load_model

# 1. Keras 모델 로드
keras_model = load_model("model/chroma_classifier.h5")

# 2. 입력 shape: 12개의 chroma feature 벡터 (배치 크기 1 포함)
input_shape = (12,)

# 3. CoreML 변환
mlmodel = ct.convert(
    keras_model,
    inputs=[ct.TensorType(shape=(1, *input_shape))],  # 배치 크기 1, (12,)
    source="tensorflow"
)

# 4. 변환된 모델 저장
mlmodel.save("model/ChromaClassifier.mlpackage")
