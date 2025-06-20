import os
import librosa
import streamlit as st
import numpy as np
import sounddevice as sd
import pandas as pd
from tensorflow.keras.models import load_model
from generate import extract_chroma

MODEL_PATH = 'model/chroma_classifier.h5'  # 학습된 모델 파일 경로
SAMPLE_RATE = 44100  # 오디오 샘플링 레이트 (generate.py와 동일하게)
DURATION = 1  # 1초 단위로 음원 분석
TRAINING_DATA_PATH = 'data/training'
df = pd.read_csv(os.path.join(TRAINING_DATA_PATH, "labels.csv"))
CLASS_NAMES = sorted(df[df.columns[0]].unique())
CMAP = 'gray_r'  # librosa 색상
print(CLASS_NAMES)


@st.cache_resource
def load_keras_model():
    """학습된 Keras 모델을 로드합니다."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        st.error(
            f"'{MODEL_PATH}' 파일이 현재 폴더에 있는지, TensorFlow/Keras가 설치되었는지 확인하세요.")
        return None


def preprocess_audio_chunk(audio_chunk, sr, noise_gate):
    """
    실시간으로 들어온 오디오 청크(Numpy 배열)에서 chroma 특징을 추출합니다.
    모델이 raw chroma 배열을 입력으로 사용하도록 변경되었습니다.
    """
    # 노이즈 제거
    y = audio_chunk
    if len(y) < 2048:  # 충분한 신호가 없으면 처리하지 않음 (STFT 최소 크기)
        return None

    y[np.abs(y) < noise_gate] = 0
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # chroma shape : (12, 87)
    chroma_mean = np.mean(chroma, axis=1)
    return chroma_mean


def chroma_to_model_input(chroma):
    return np.expand_dims(chroma, axis=0)

 # --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🎸 실시간 기타 코드 분류")
st.write(f"{DURATION}초 단위로 마이크 입력을 받아 코드를 예측합니다.")

# 노이즈 게이트 슬라이더
NOISE_GATE = st.slider("🔊 노이즈 게이트 임계값", min_value=0.0,
                       max_value=0.1, value=0.01, step=0.001)

# 모델 로드
model = load_keras_model()

if model:
    # 녹음 상태 관리를 위해 session_state 사용
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False

    col1, col2 = st.columns([1, 2])

    with col1:
        start_button = st.button("▶️ 예측 시작", type="primary")
        stop_button = st.button("⏹️ 예측 중지")

    if start_button:
        st.session_state.is_recording = True
    if stop_button:
        st.session_state.is_recording = False

    with col2:
        st.info("예측이 시작되면 아래에 결과가 표시됩니다.")

    # 결과 표시를 위한 placeholder
    placeholder = st.empty()

    if st.session_state.is_recording:
        st.toast("녹음을 시작합니다...", icon="🎤")

    # 실시간 예측 루프
    while st.session_state.is_recording:
        try:
            # 1. 실시간 오디오 녹음
            audio_chunk = sd.rec(int(DURATION * SAMPLE_RATE),
                                 samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()  # 녹음이 끝날 때까지 대기
            audio_chunk = audio_chunk.flatten()

            # 2. 오디오 전처리
            chroma = preprocess_audio_chunk(
                audio_chunk, sr=SAMPLE_RATE, noise_gate=NOISE_GATE)

            if chroma is not None:
                # 3. 모델 입력 형태로 변환
                model_input = chroma_to_model_input(chroma)

                # 4. 코드 예측
                prediction = model.predict(model_input)[0]
                predicted_index = np.argmax(prediction)
                predicted_label = CLASS_NAMES[predicted_index]
                confidence = prediction[predicted_index]

                # 5. 결과 시각화
                with placeholder.container():
                    st.subheader("📊 예측 결과")
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric(label="예측된 코드", value=f"{predicted_label}",
                                  help=f"신뢰도: {confidence:.2%}")

                        st.write("📈 **코드별 신뢰도**")
                        # 신뢰도 데이터를 DataFrame으로 만들어 st.bar_chart에 전달
                        confidence_df = pd.DataFrame({
                            'Code': CLASS_NAMES,
                            'Confidence': prediction
                        }).set_index('Code')
                        st.bar_chart(confidence_df)
            else:
                with placeholder.container():
                    st.warning("소리가 너무 작아 분석할 수 없습니다. 더 크게 연주해주세요.", icon="🔇")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            st.session_state.is_recording = False
            break

    if not st.session_state.is_recording:
        st.toast("예측이 중지되었습니다.")
