import os
import tempfile
import subprocess
import streamlit as st
import whisper


st.set_page_config(page_title="Transcriptor de video/audio", page_icon="🎙️")
st.title("🎙️ Transcriptor de video o audio a texto")
st.write("Sube un video o un audio, lo transcribo y luego descargas el TXT.")


VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"]
AUDIO_EXTS = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"]


@st.cache_resource(max_entries=1)
def cargar_modelo(nombre_modelo):
    return whisper.load_model(nombre_modelo)


def convertir_a_wav_16k(ruta_entrada, ruta_salida):
    """
    Convierte cualquier audio o video a WAV mono 16kHz usando ffmpeg.
    """
    comando = [
        "ffmpeg",
        "-y",
        "-i", ruta_entrada,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        ruta_salida
    ]

    proceso = subprocess.run(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if proceso.returncode != 0:
        raise RuntimeError(
            "No se pudo convertir el archivo a WAV.\n\n"
            f"Detalle ffmpeg:\n{proceso.stderr}"
        )


def transcribir_archivo_subido(uploaded_file, modelo_nombre="base", language="es"):
    with tempfile.TemporaryDirectory() as temp_dir:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if not ext:
            raise ValueError("No se pudo identificar la extensión del archivo.")

        if ext not in VIDEO_EXTS and ext not in AUDIO_EXTS:
            raise ValueError("Formato no soportado.")

        ruta_entrada = os.path.join(temp_dir, f"archivo_subido{ext}")
        ruta_audio_wav = os.path.join(temp_dir, "audio_normalizado.wav")

        # Guardar archivo subido
        with open(ruta_entrada, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if not os.path.exists(ruta_entrada):
            raise ValueError("No se pudo guardar el archivo temporal.")

        if os.path.getsize(ruta_entrada) == 0:
            raise ValueError("El archivo subido quedó vacío.")

        # Convertir SIEMPRE a WAV 16k mono
        convertir_a_wav_16k(ruta_entrada, ruta_audio_wav)

        if not os.path.exists(ruta_audio_wav):
            raise ValueError("No se generó el audio WAV temporal.")

        if os.path.getsize(ruta_audio_wav) == 0:
            raise ValueError("El audio convertido quedó vacío.")

        # Validar que Whisper sí pueda leerlo
        audio_array = whisper.audio.load_audio(ruta_audio_wav)
        if len(audio_array) == 0:
            raise ValueError("Whisper recibió un audio vacío después de la conversión.")

        modelo = cargar_modelo(modelo_nombre)

        kwargs = {
            "fp16": False
        }

        if language is not None:
            kwargs["language"] = language

        resultado = modelo.transcribe(ruta_audio_wav, **kwargs)

        texto = resultado.get("text", "").strip()

        if not texto:
            raise ValueError("La transcripción salió vacía.")

        return texto


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
            st.error(f"Ocurrió un error:\n{e}")

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