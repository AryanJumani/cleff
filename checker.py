import tempfile
from pathlib import Path
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import demucs.api
import asyncio
import json
import librosa
import numpy as np
from music21.musicxml.m21ToXml import GeneralObjectExporter

from music21 import (
    stream,
    note,
    tempo,
    meter,
    metadata,
    instrument,
    midi,
    duration as m21duration,
)

# --- Global Setup ---
# Create a static directory to serve the output files from
OUTPUT_DIR = Path("separated_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize the Separator model once when the server starts.
separator = demucs.api.Separator(
    model="htdemucs_6s",
    device="mps",
    segment=7.8,
)

app = FastAPI(title="Demucs Live Update API")

# --- CORS Middleware ---
# Allow the frontend (even when opened as a local file) to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Live Update Generator Function ---
async def separation_generator(file: UploadFile):
    """
    An async generator that yields progress updates and final file paths.
    """
    # Use a temporary file to handle the upload safely
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=file.filename
    ) as temp_input_file:
        shutil.copyfileobj(file.file, temp_input_file)
        temp_input_path = Path(temp_input_file.name)

    try:
        # --- Separation Step ---
        yield f"data: Starting separation for '{file.filename}'...\n\n"
        _, separated = await asyncio.to_thread(
            separator.separate_audio_file, temp_input_path
        )
        yield "data: ✅ Separation complete.\n\n"
        yield "data: Switching to saving mode...\n\n"

        # --- Saving Step ---
        # Save files into a unique subfolder within our main static output directory
        song_output_folder = OUTPUT_DIR / temp_input_path.stem
        song_output_folder.mkdir(exist_ok=True)

        final_stem_paths = {}

        for stem_name, stem_waveform in separated.items():
            yield f"data:   - Saving {stem_name}...\n\n"
            # The path for saving the file
            output_path = song_output_folder / f"{stem_name}.mp3"
            # The relative URL path for the frontend to use
            url_path = f"{OUTPUT_DIR.name}/{temp_input_path.stem}/{stem_name}.mp3"

            await asyncio.to_thread(
                demucs.api.save_audio,
                stem_waveform,
                str(output_path),
                samplerate=separator.samplerate,
            )
            final_stem_paths[stem_name] = url_path

        yield f"data: \n✅ All stems saved successfully.\n\n"

        # --- Final message with JSON data ---
        # This is a structured way to send the final file paths to the frontend
        final_data = json.dumps({"status": "done", "paths": final_stem_paths})
        yield f"data: {final_data}\n\n"

    finally:
        # Clean up the temporary input file
        temp_input_path.unlink()


# --- API Endpoint ---
@app.post("/separate-live")
def separate_audio_live_endpoint(file: UploadFile = File(...)):
    """
    This endpoint takes a file and returns a live stream of progress updates.
    """
    return StreamingResponse(separation_generator(file), media_type="text/event-stream")


def extract_notes_with_onsets(
    audio_path: Path,
    fmin_note="C1",
    n_bins=72,
    n_fft=2048,
    overlap=0.5,
    mag_exp=4,
    cqt_threshold_db=-61,
    onset_prepost=6,
    backtrack=True,
):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    hop_length = int(n_fft * (1 - overlap))
    C = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=librosa.note_to_hz(fmin_note),
        n_bins=n_bins,
    )
    C_mag = np.abs(C) ** mag_exp
    CdB = librosa.amplitude_to_db(C_mag, ref=np.max)
    CdB_thresh = CdB.copy()
    CdB_thresh[CdB_thresh < cqt_threshold_db] = -120.0
    onset_env = librosa.onset.onset_strength(
        S=CdB_thresh, sr=sr, aggregate=np.mean, hop_length=hop_length
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack,
        pre_max=onset_prepost,
        post_max=onset_prepost,
        units="frames",
    )
    onset_boundaries = np.concatenate([[0], onset_frames, [CdB.shape[1]]])
    onset_times = librosa.frames_to_time(onset_boundaries, sr=sr, hop_length=hop_length)
    tempo_bpm, _ = librosa.beat.beat_track(
        y=None, sr=sr, onset_envelope=onset_env, hop_length=hop_length
    )
    tempo_bpm = float(np.atleast_1d(tempo_bpm)[0])
    if tempo_bpm <= 0 or np.isnan(tempo_bpm):
        tempo_bpm = 60.0
    freqs = librosa.cqt_frequencies(
        n_bins=n_bins, fmin=librosa.note_to_hz(fmin_note), bins_per_octave=12
    )
    notes_info = []
    for i in range(len(onset_boundaries) - 1):
        b0 = onset_boundaries[i]
        b1 = onset_boundaries[i + 1]
        if b1 <= b0:
            continue
        segment = CdB_thresh[:, b0:b1]
        peak_vals = np.max(segment, axis=1)
        peak_val = peak_vals.max()
        if peak_val <= cqt_threshold_db or np.isneginf(peak_val):
            notes_info.append((None, onset_times[i], onset_times[i + 1], 0.0))
        else:
            bin_idx = int(np.argmax(peak_vals))
            hz = freqs[bin_idx]
            midi_num = int(round(librosa.hz_to_midi(hz)))
            vel = float(
                np.clip(
                    (peak_vals[bin_idx] - CdB.min()) / (CdB.max() - CdB.min()), 0.0, 1.0
                )
            )
            vel_midi = int(round(vel * 127))
            notes_info.append((midi_num, onset_times[i], onset_times[i + 1], vel_midi))
    return notes_info, float(tempo_bpm)


def build_musicxml_from_notes(notes_info, tempo_bpm, title="transcription"):
    s = stream.Score()
    p = stream.Part()
    p.append(meter.TimeSignature("4/4"))
    p.append(tempo.MetronomeMark(number=round(tempo_bpm, 3)))
    s.insert(0, metadata.Metadata())
    s.metadata.title = title
    for midi_num, t0, t1, vel in notes_info:
        dur_sec = max(0.0, t1 - t0)
        if dur_sec <= 0.01:
            continue
        sec_per_beat = 60.0 / tempo_bpm
        ql = dur_sec / sec_per_beat
        ql = round(ql * 16) / 16.0
        if ql <= 0:
            ql = 0.25
        if midi_num is None:
            r = note.Rest()
            r.duration.quarterLength = ql
            p.append(r)
        else:
            n = note.Note(midi=int(midi_num))
            n.duration.quarterLength = ql
            p.append(n)
    s.append(p)
    exporter = GeneralObjectExporter(s)
    xml_str = exporter.parse()
    return xml_str


def audio_to_musicxml_string(
    audio_path: Path,
    fmin_note="C1",
    n_bins=72,
    n_fft=2048,
    overlap=0.5,
    mag_exp=4,
    cqt_threshold_db=-61,
    onset_prepost=6,
    backtrack=True,
):
    notes_info, tempo_bpm = extract_notes_with_onsets(
        audio_path,
        fmin_note=fmin_note,
        n_bins=n_bins,
        n_fft=n_fft,
        overlap=overlap,
        mag_exp=mag_exp,
        cqt_threshold_db=cqt_threshold_db,
        onset_prepost=onset_prepost,
        backtrack=backtrack,
    )
    xml_str = build_musicxml_from_notes(notes_info, tempo_bpm, title=audio_path.stem)
    return xml_str


@app.get("/transcribe")
async def transcribe_existing_stem(stem: str = Query(...)):
    stem_path = Path(stem)
    if not stem_path.exists():
        stem_path = Path(OUTPUT_DIR) / stem
    if not stem_path.exists():
        return {"error": f"Stem file not found: {stem}"}

    xml_str = await asyncio.to_thread(audio_to_musicxml_string, stem_path)
    if isinstance(xml_str, bytes):
        xml_str = xml_str.decode("utf-8")

    return JSONResponse(content={"status": "ok", "xml": xml_str})


# --- Static File Server ---
# This line mounts the 'separated_audio' directory, making the output files
# accessible to the browser at http://.../separated_audio/
app.mount(f"/{OUTPUT_DIR.name}", StaticFiles(directory=OUTPUT_DIR), name="separated")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
