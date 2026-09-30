import streamlit as st
import librosa
import numpy as np
import pandas as pd
import snowflake.connector


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Music Genre Classification")
st.write(
    "Upload a music file and let the Snowflake ML model predict its genre."
)


# ---------------------------------------------------------
# SNOWFLAKE CONNECTION
# ---------------------------------------------------------
def get_connection():
    return snowflake.connector.connect(
        account=st.secrets["snowflake"]["account"],
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        warehouse="COMPUTE_WH",
        database="MUSIC_GENRE_DB",
        schema="ML_SCHEMA",
        role="ACCOUNTADMIN"
    )


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_features(audio_file):

    # Load audio
    y, sr = librosa.load(
        audio_file,
        sr=22050,
        mono=True,
        duration=30
    )

    features = {}

    # -----------------------------------------------------
    # Basic feature
    # -----------------------------------------------------
    features["LENGTH"] = len(y)

    # -----------------------------------------------------
    # Chroma STFT
    # -----------------------------------------------------
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features["CHROMA_STFT_MEAN"] = np.mean(chroma)
    features["CHROMA_STFT_VAR"] = np.var(chroma)

    # -----------------------------------------------------
    # RMS
    # -----------------------------------------------------
    rms = librosa.feature.rms(y=y)

    features["RMS_MEAN"] = np.mean(rms)
    features["RMS_VAR"] = np.var(rms)

    # -----------------------------------------------------
    # Spectral Centroid
    # -----------------------------------------------------
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features["SPECTRAL_CENTROID_MEAN"] = np.mean(
        spectral_centroid
    )

    features["SPECTRAL_CENTROID_VAR"] = np.var(
        spectral_centroid
    )

    # -----------------------------------------------------
    # Spectral Bandwidth
    # -----------------------------------------------------
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features["SPECTRAL_BANDWIDTH_MEAN"] = np.mean(
        spectral_bandwidth
    )

    features["SPECTRAL_BANDWIDTH_VAR"] = np.var(
        spectral_bandwidth
    )

    # -----------------------------------------------------
    # Spectral Rolloff
    # -----------------------------------------------------
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features["ROLLOFF_MEAN"] = np.mean(rolloff)
    features["ROLLOFF_VAR"] = np.var(rolloff)

    # -----------------------------------------------------
    # Zero Crossing Rate
    # -----------------------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y)

    features["ZERO_CROSSING_RATE_MEAN"] = np.mean(zcr)
    features["ZERO_CROSSING_RATE_VAR"] = np.var(zcr)

    # -----------------------------------------------------
    # Harmony and Percussive components
    # -----------------------------------------------------
    y_harm, y_perc = librosa.effects.hpss(y)

    features["HARMONY_MEAN"] = np.mean(y_harm)
    features["HARMONY_VAR"] = np.var(y_harm)

    features["PERCEPTR_MEAN"] = np.mean(y_perc)
    features["PERCEPTR_VAR"] = np.var(y_perc)

    # -----------------------------------------------------
    # Tempo
    # -----------------------------------------------------
    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    features["TEMPO"] = float(
        np.asarray(tempo).reshape(-1)[0]
    )

    # -----------------------------------------------------
    # MFCC 1 - 20
    # -----------------------------------------------------
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    for i in range(20):

        features[f"MFCC{i + 1}_MEAN"] = np.mean(
            mfcc[i]
        )

        features[f"MFCC{i + 1}_VAR"] = np.var(
            mfcc[i]
        )

    # Convert to DataFrame
    return pd.DataFrame([features])


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a music file",
    type=["wav", "mp3", "ogg", "flac"]
)


# ---------------------------------------------------------
# PROCESS FILE
# ---------------------------------------------------------
if uploaded_file is not None:

    # Audio player
    st.audio(uploaded_file)

    # Prediction button
    if st.button("🎯 Predict Genre"):

        try:

            # -------------------------------------------------
            # STEP 1: FEATURE EXTRACTION
            # -------------------------------------------------
            with st.spinner("Extracting audio features..."):

                features_df = extract_features(
                    uploaded_file
                )

            st.success(
                "Audio features extracted successfully!"
            )

            # Show extracted features
            with st.expander("View extracted features"):

                st.dataframe(
                    features_df,
                    use_container_width=True
                )

            # -------------------------------------------------
            # STEP 2: CONNECT TO SNOWFLAKE
            # -------------------------------------------------
            with st.spinner(
                "Connecting to Snowflake..."
            ):

                conn = get_connection()

                cursor = conn.cursor()

            # -------------------------------------------------
            # STEP 3: CREATE TEMPORARY TABLE
            # -------------------------------------------------
            with st.spinner(
                "Sending features to Snowflake..."
            ):

                columns = ", ".join(
                    f'"{col}" FLOAT'
                    for col in features_df.columns
                )

                cursor.execute(
                    f"""
                    CREATE OR REPLACE TEMPORARY TABLE
                    APP_AUDIO_FEATURES
                    (
                        {columns}
                    )
                    """
                )

                # Prepare values
                values = tuple(
                    features_df.iloc[0].tolist()
                )

                placeholders = ", ".join(
                    ["%s"] * len(values)
                )

                cursor.execute(
                    f"""
                    INSERT INTO APP_AUDIO_FEATURES
                    VALUES ({placeholders})
                    """,
                    values
                )

            # -------------------------------------------------
            # STEP 4: PREDICT USING SNOWFLAKE ML
            # -------------------------------------------------
            with st.spinner(
                "Running Snowflake ML prediction..."
            ):

                cursor.execute(
                    """
                    SELECT
                        MUSIC_GENRE_DB.ML_SCHEMA.MUSIC_GENRE_MODEL!
                        PREDICT(
                            INPUT_DATA => {*}
                        ):class::STRING AS PREDICTED_GENRE
                    FROM APP_AUDIO_FEATURES
                    """
                )

                result = cursor.fetchone()

                predicted_genre = result[0]

            # -------------------------------------------------
            # STEP 5: CLOSE CONNECTION
            # -------------------------------------------------
            cursor.close()
            conn.close()

            # -------------------------------------------------
            # STEP 6: DISPLAY RESULT
            # -------------------------------------------------
            st.success(
                "Prediction completed!"
            )

            st.markdown(
                "## 🎶 Predicted Genre"
            )

            st.header(
                predicted_genre.upper()
            )

        except Exception as e:

            # Try to close connection if an error occurs
            try:
                cursor.close()
            except:
                pass

            try:
                conn.close()
            except:
                pass

            st.error(
                "Something went wrong."
            )

            st.code(
                str(e)
            )