import os
import tempfile
import streamlit as st
import whisper
from moviepy import VideoFileClip


st.set_page_config(page_title="Transcriptor de video/audio", page_icon="🎙️")
st.title("🎙️ Transcriptor de video o audio a texto")
st.write("Sube un video o un audio, lo transcribo y luego descargas el TXT.")


VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"]
AUDIO_EXTS = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"]


@st.cache_resource(max_entries=1)
def cargar_modelo(nombre_modelo):
    return whisper.load_model(nombre_modelo)


def transcribir_archivo_subido(uploaded_file, modelo_nombre="base", language="es"):
    with tempfile.TemporaryDirectory() as temp_dir:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if not ext:
            raise ValueError("No se pudo identificar la extensión del archivo.")

        ruta_entrada = os.path.join(temp_dir, f"archivo_subido{ext}")

        with open(ruta_entrada, "wb") as f:
            f.write(uploaded_file.getbuffer())

        modelo = cargar_modelo(modelo_nombre)

        # Si es video, extraemos audio
        if ext in VIDEO_EXTS:
            ruta_audio = os.path.join(temp_dir, "audio_temp.wav")

            video = VideoFileClip(ruta_entrada)

            if video.audio is None:
                video.close()
                raise ValueError("El video no contiene audio.")

            audio = video.audio
            audio.write_audiofile(ruta_audio, logger=None)
            audio.close()
            video.close()

            resultado = modelo.transcribe(ruta_audio, language=language)

        # Si es audio, se transcribe directo
        elif ext in AUDIO_EXTS:
            resultado = modelo.transcribe(ruta_entrada, language=language)

        else:
            raise ValueError("Formato no soportado.")

        return resultado["text"]


uploaded_file = st.file_uploader(
    "Sube tu video o audio",
    type=[
        "mp4", "mov", "avi", "mkv", "m4v", "webm",
        "mp3", "wav", "m4a", "aac", "flac", "ogg", "wma"
    ]
)

modelo_nombre = st.selectbox(
    "Modelo Whisper",
    ["tiny", "base"],
    index=1
)

language = st.selectbox(
    "Idioma del audio",
    ["es", "en", "auto"],
    index=0
)

if uploaded_file is not None:
    st.info(f"Archivo cargado: {uploaded_file.name}")

    ext = os.path.splitext(uploaded_file.name)[1].lower()

    if ext in VIDEO_EXTS:
        st.video(uploaded_file)
    elif ext in AUDIO_EXTS:
        st.audio(uploaded_file)

    if st.button("Transcribir"):
        try:
            with st.spinner("Procesando y transcribiendo..."):
                idioma = None if language == "auto" else language
                texto = transcribir_archivo_subido(
                    uploaded_file,
                    modelo_nombre=modelo_nombre,
                    language=idioma
                )

            nombre_base = os.path.splitext(uploaded_file.name)[0]
            nombre_txt = f"{nombre_base}_transcripcion.txt"

            st.success("Transcripción completada")
            st.text_area("Texto transcrito", texto, height=350)

            st.download_button(
                label="Descargar TXT",
                data=texto,
                file_name=nombre_txt,
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")