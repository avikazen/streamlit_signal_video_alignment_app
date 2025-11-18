import tempfile
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks

from app_helpers import butterworth_filter, standardize

BASE_DIR = Path(__file__).resolve().parents[1]
PHY_PHOX_DIR = BASE_DIR / "data" / "phyphox"
DEFAULT_SESSION = "JT 5 hurdles - three step - right leg  2025-11-11 17-36-33"
DEFAULT_VIDEO = PHY_PHOX_DIR / DEFAULT_SESSION / (DEFAULT_SESSION + ".MOV")
DEFAULT_SENSOR = PHY_PHOX_DIR / DEFAULT_SESSION / (DEFAULT_SESSION + ".xls")


@st.cache_data(show_spinner=False)
def load_sensor_dataframe(source_path: str) -> pd.DataFrame:
    path = Path(source_path)
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_csv(path, delim_whitespace=True, engine="python")
    except Exception:
        return pd.read_csv(path)


def infer_column(df: pd.DataFrame, token: str) -> str:
    lowered = token.lower()
    for column in df.columns:
        if lowered in str(column).lower():
            return column
    raise KeyError(f"Could not find a column containing '{token}'")


def calc_samplerate(time_vec: np.ndarray) -> float:
    if len(time_vec) < 2:
        return 1.0
    duration = float(time_vec[-1] - time_vec[0])
    if duration <= 0:
        return len(time_vec)
    return len(time_vec) / duration


def discover_axis_columns(df: pd.DataFrame) -> dict[str, str]:
    tokens = ["x", "y", "z"]
    mapping: dict[str, str] = {}
    for column in df.columns:
        lowered = str(column).lower()
        for token in tokens:
            if token in lowered and token.upper() not in mapping:
                mapping[token.upper()] = column
    return mapping


def build_axis_signals(df: pd.DataFrame, axis_columns: dict[str, str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    time_col = infer_column(df, "time")
    raw_time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    time_mask = np.isfinite(raw_time)
    time_data = raw_time[time_mask]
    signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for axis_label, column in axis_columns.items():
        raw_signal = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        masked_signal = raw_signal[time_mask]
        axis_mask = np.isfinite(masked_signal)
        if axis_mask.sum() < 2:
            continue
        axis_times = time_data[axis_mask]
        normalized = standardize(masked_signal[axis_mask])
        signals[axis_label] = (axis_times, normalized)
    return signals


def detect_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    filtered = butterworth_filter(signal, lowcut=10, highcut=49, fs=fs, order=4)
    standardized = standardize(filtered)
    peaks, _ = find_peaks(standardized, distance=40, prominence=1)
    return peaks


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".bin"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getbuffer())
    temp_file.flush()
    temp_path = Path(temp_file.name)
    temp_file.close()
    return temp_path


def get_video_metadata(video_path: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps else 0.0
    cap.release()
    return fps, duration


def read_video_frame(video_path: Path, time_sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(time_sec, 0.0) * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def crop_frame_vertically(frame: np.ndarray, target_ratio: float) -> np.ndarray:
    height, width = frame.shape[:2]
    desired_height = int(round(width / target_ratio))
    if desired_height <= 0 or desired_height >= height:
        return frame
    top = (height - desired_height) // 2
    return frame[top : top + desired_height]


def plot_signal_window(axis_data: dict[str, tuple[np.ndarray, np.ndarray]], center_time: float, window: float) -> plt.Figure:
    half = max(window / 2, 0.1)
    start_time = center_time - half
    end_time = center_time + half
    fig, ax = plt.subplots(figsize=(6, 3))
    plotted = False
    for label, (times, signal) in axis_data.items():
        mask = (times >= start_time) & (times <= end_time)
        if mask.any():
            ax.plot(times[mask], signal[mask], label=label)
            plotted = True
    if not plotted:
        ax.plot([], [])
    ax.axvline(center_time, color="red", linestyle="--", label="Aligned time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Standardized accel")
    ax.set_title("Sensor window around selected frame")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_full_trace(axis_data: dict[str, tuple[np.ndarray, np.ndarray]], primary_axis: str, peaks: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    for label, (times, signal) in axis_data.items():
        ax.plot(times, signal, label=label)
    if peaks.size and primary_axis in axis_data:
        primary_times, primary_signal = axis_data[primary_axis]
        ax.scatter(primary_times[peaks], primary_signal[peaks], color="#d62728", marker="x", label="Peaks")
        for idx in range(1, len(peaks)):
            delta = primary_times[peaks[idx]] - primary_times[peaks[idx - 1]]
            ax.text(
                primary_times[peaks[idx]],
                primary_signal[peaks[idx]] + 0.5,
                f"Δ {delta:.2f}s",
                fontsize=8,
                ha="center",
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Standardized accel")
    ax.set_title("Full accelerometer recording with detected peaks")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Hurdle Alignment Studio", layout="wide")
    st.title("Frame + Accelerometer Alignment")
    st.markdown(
        """
        Upload your video and accelerometer data files.
        """
    )

    with st.sidebar:
        st.header("Inputs & controls")
        video_upload = st.file_uploader("Video file", type=["mov", "mp4", "avi"], key="video_upload")
        sensor_upload = st.file_uploader("Accelerometer data", type=["xls", "xlsx", "csv", "txt"], key="sensor_upload")

    if video_upload is not None:
        video_path = save_uploaded_file(video_upload)
        video_source = video_upload.name
    else:
        video_path = DEFAULT_VIDEO if DEFAULT_VIDEO.exists() else None
        video_source = DEFAULT_VIDEO.name if video_path else "(none)"

    if sensor_upload is not None:
        sensor_path = save_uploaded_file(sensor_upload)
        sensor_source = sensor_upload.name
    else:
        sensor_path = DEFAULT_SENSOR if DEFAULT_SENSOR.exists() else None
        sensor_source = DEFAULT_SENSOR.name if sensor_path else "(none)"

    st.sidebar.divider()
    st.sidebar.markdown(f"**Video:** {video_source}")
    st.sidebar.markdown(f"**Sensor:** {sensor_source}")

    if video_path is None or sensor_path is None:
        st.warning("Upload video and data files.")
        return

    try:
        df = load_sensor_dataframe(str(sensor_path))
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        st.error(f"Unable to load accelerometer data: {exc}")
        return

    axis_columns = discover_axis_columns(df)
    if not axis_columns:
        st.error("No accelerometer axes detected in the uploaded file.")
        return

    axis_signals = build_axis_signals(df, axis_columns)
    if not axis_signals:
        st.error("Accelerometer data does not contain enough valid samples for any axis.")
        return

    axis_options = list(axis_signals.keys())
    default_axis = "Y" if "Y" in axis_options else axis_options[0]
    if "axis_picker" not in st.session_state:
        st.session_state.axis_picker = [default_axis]
    axes_info = ", ".join(f"{axis}: {axis_columns[axis]}" for axis in axis_options if axis in axis_columns)

    _, duration = get_video_metadata(video_path)
    duration = max(duration, 1.0)
    default_video_time = min(1.0, duration)
    video_time = default_video_time
    default_offset = 4.75
    window_width = 2.0

    with st.container():
        frame_col, signal_col = st.columns([1.5, 1])
        with frame_col:
            st.subheader("Video frame")
            image_placeholder = st.empty()
            video_time = st.slider(
                "Video time (s)",
                min_value=0.0,
                max_value=float(duration),
                value=video_time,
                step=0.1,
                key="video_time_slider",
            )
            frame = read_video_frame(video_path, video_time)
            if frame is not None:
                frame = crop_frame_vertically(frame, target_ratio=2.0)
                image_placeholder.image(frame, use_container_width=True)
            else:
                image_placeholder.info("Frame not available at selected time.")
        with signal_col:
            st.subheader("Sensor window")
            selected_axes = st.multiselect(
                "Select axes to display", axis_options, default=st.session_state.axis_picker, key="axis_picker"
            )
            if not selected_axes:
                st.warning("Select at least one axis to display.")
                return

            if "offset_input" not in st.session_state:
                st.session_state.offset_input = f"{default_offset:.2f}"
            try:
                offset = float(st.session_state.offset_input)
            except ValueError:
                st.warning("Enter a numeric offset (e.g., 4.75).")
                offset = default_offset
                st.session_state.offset_input = f"{offset:.2f}"

            if "window_slider" not in st.session_state:
                st.session_state.window_slider = window_width
            window_width = st.session_state.window_slider

            sensor_time = video_time - offset
            selected_axis_data = {axis: axis_signals[axis] for axis in selected_axes}
            snippet_fig = plot_signal_window(selected_axis_data, sensor_time, window_width)
            st.pyplot(snippet_fig)

            def adjust_offset(delta: float) -> None:
                try:
                    current = float(st.session_state.offset_input)
                except (KeyError, ValueError):
                    current = default_offset
                new_value = round(current + delta, 2)
                st.session_state.offset_input = f"{new_value:.2f}"

            label_col, input_col, dec_col, inc_col = st.columns([1.5, 1.2, 0.6, 0.6])
            with label_col:
                st.write("Signal offset (s)")
            with input_col:
                st.text_input("", key="offset_input", label_visibility="collapsed")
            with dec_col:
                st.button(
                    "-0.1 s",
                    key="offset_decrease",
                    use_container_width=True,
                    on_click=adjust_offset,
                    kwargs={"delta": -0.1},
                )
            with inc_col:
                st.button(
                    "+0.1 s",
                    key="offset_increase",
                    use_container_width=True,
                    on_click=adjust_offset,
                    kwargs={"delta": 0.1},
                )

            width_label_col, width_slider_col = st.columns([1.2, 2])
            with width_label_col:
                st.write("Sensor window width (s)")
            with width_slider_col:
                st.slider(
                    "",
                    min_value=0.5,
                    max_value=5.0,
                    value=window_width,
                    step=0.25,
                    key="window_slider",
                    label_visibility="collapsed",
                )

    selected_axes = st.session_state.axis_picker
    if not selected_axes:
        return
    selected_axis_data = {axis: axis_signals[axis] for axis in selected_axes}
    primary_axis = selected_axes[0]
    primary_times, primary_signal = selected_axis_data[primary_axis]
    fs = calc_samplerate(primary_times)
    peaks = detect_peaks(primary_signal, fs)

    st.divider()
    st.subheader("Full accelerometer trace")
    full_fig = plot_full_trace(selected_axis_data, primary_axis, peaks)
    st.pyplot(full_fig)


if __name__ == "__main__":
    main()
