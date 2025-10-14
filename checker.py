import tempfile
from pathlib import Path
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import demucs.api
import asyncio
import json

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


# --- Static File Server ---
# This line mounts the 'separated_audio' directory, making the output files
# accessible to the browser at http://.../separated_audio/
app.mount(f"/{OUTPUT_DIR.name}", StaticFiles(directory=OUTPUT_DIR), name="separated")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
