import coremltools as ct
from tensorflow.keras.models import load_model

# 모델 로드
keras_model = load_model("model/model.h5")

# 입력 shape
input_shape = (128, 128, 3)

# CoreML 변환
mlmodel = ct.convert(
    keras_model,
    inputs=[ct.ImageType(shape=(1, *input_shape), scale=1/255.0)],
    source="tensorflow"
)

# 저장
mlmodel.save("model/GuitarChordClassifier.mlpackage")
