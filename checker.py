import tempfile
from pathlib import Path
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import demucs.api
import asyncio
import json
import librosa
import numpy as np
from music21 import stream, note, tempo, meter

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


def audio_to_sheet_music(audio_path: Path, output_path_musicxml: Path):
    y, sr = librosa.load(str(audio_path))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    notes = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        pitch = pitches[idx, t]
        if pitch > 0:
            notes.append(librosa.hz_to_midi(pitch))

    if not notes:
        raise RuntimeError("No notes detected in audio.")

    frame_duration = librosa.get_duration(y=y, sr=sr) / pitches.shape[1]
    note_events, current_note, start_time = [], notes[0], 0

    for i in range(1, len(notes)):
        if notes[i] != current_note:
            dur = (i - start_time) * frame_duration
            note_events.append({"midi": current_note, "duration_sec": dur})
            current_note, start_time = notes[i], i
    dur = (len(notes) - start_time) * frame_duration
    note_events.append({"midi": current_note, "duration_sec": dur})

    score = stream.Score()
    part = stream.Part()
    part.append(meter.TimeSignature("4/4"))
    part.append(tempo.MetronomeMark(number=bpm))
    sec_per_beat = 60.0 / bpm

    for ev in note_events:
        if ev["duration_sec"] < 0.1:
            continue
        n = note.Note(midi=int(round(ev["midi"])))
        ql = ev["duration_sec"] / sec_per_beat
        n.duration.quarterLength = round(ql * 4) / 4.0
        part.append(n)

    score.append(part)
    score.write(fp=str(output_path_musicxml))
    return output_path_musicxml


@app.get("/transcribe")
async def transcribe_existing_stem(stem: str = Query(...)):
    stem_path = Path(stem)
    if not stem_path.exists():
        stem_path = Path(OUTPUT_DIR) / stem
    if not stem_path.exists():
        return {"error": f"Stem file not found: {stem}"}

    output_xml = stem_path.with_suffix(".musicxml")
    await asyncio.to_thread(audio_to_sheet_music, stem_path, output_xml)
    return FileResponse(output_xml, filename=output_xml.name)


# --- Static File Server ---
# This line mounts the 'separated_audio' directory, making the output files
# accessible to the browser at http://.../separated_audio/
app.mount(f"/{OUTPUT_DIR.name}", StaticFiles(directory=OUTPUT_DIR), name="separated")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
