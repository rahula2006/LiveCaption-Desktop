
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel


# ---------------- CONFIGURATION ----------------
SAMPLE_RATE = 16000
CHUNK_SECONDS = 8
MODEL_SIZE = "base"       # "tiny" = faster; "small" = potentially more accurate
SILENCE_THRESHOLD = 0.003
SILENCE_SECONDS = 30
# ------------------------------------------------


class FloatingTranscriber:
    def __init__(self, root):
        self.root = root
        self.root.title("Floating Transcriber")
        self.root.geometry("560x440+100+100")
        self.root.minsize(380, 300)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1e1e2e")

        self.running = False
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.transcript = []
        self.lock = threading.Lock()
        self.started_at = None
        self.auto_stop = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready — press Start, then play your video.")
        self.output_path = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TButton",
            font=("Segoe UI", 10),
            padding=7,
            background="#89b4fa",
            foreground="#11111b"
        )

        header = tk.Frame(self.root, bg="#313244", height=42)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header, text="  🎙  FLOATING TRANSCRIBER",
            bg="#313244", fg="#cdd6f4",
            font=("Segoe UI", 11, "bold")
        ).pack(side="left", padx=8)

        # Drag the window using its header.
        header.bind("<Button-1>", self.start_move)
        header.bind("<B1-Motion>", self.do_move)

        controls = tk.Frame(self.root, bg="#1e1e2e")
        controls.pack(fill="x", padx=12, pady=10)

        self.start_button = ttk.Button(
            controls, text="▶ START", command=self.toggle
        )
        self.start_button.pack(side="left", padx=(0, 6))

        ttk.Button(
            controls, text="Clear", command=self.clear
        ).pack(side="left", padx=6)

        ttk.Button(
            controls, text="Save As", command=self.save_as
        ).pack(side="left", padx=6)

        tk.Checkbutton(
            self.root,
            text="Auto-stop after 30 seconds of silence",
            variable=self.auto_stop,
            bg="#1e1e2e", fg="#cdd6f4",
            selectcolor="#313244",
            activebackground="#1e1e2e",
            activeforeground="#cdd6f4",
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=15, pady=(0, 7))

        tk.Label(
            self.root, textvariable=self.status,
            bg="#1e1e2e", fg="#a6e3a1",
            font=("Segoe UI", 9), anchor="w",
            wraplength=520
        ).pack(fill="x", padx=15, pady=(0, 8))

        body = tk.Frame(self.root, bg="#1e1e2e")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        scrollbar = ttk.Scrollbar(body)
        scrollbar.pack(side="right", fill="y")

        self.text = tk.Text(
            body, wrap="word",
            bg="#11111b", fg="#cdd6f4",
            insertbackground="white",
            font=("Segoe UI", 10),
            padx=10, pady=10,
            relief="flat",
            yscrollcommand=scrollbar.set
        )
        self.text.pack(fill="both", expand=True)
        scrollbar.config(command=self.text.yview)

        self.text.insert(
            "1.0",
            "Your transcript will appear here.\n"
            "Start the recorder before playing your video.\n\n"
        )

    def start_move(self, event):
        self.drag_x = event.x
        self.drag_y = event.y

    def do_move(self, event):
        x = self.root.winfo_x() + event.x - self.drag_x
        y = self.root.winfo_y() + event.y - self.drag_y
        self.root.geometry(f"+{x}+{y}")

    def set_status(self, message):
        self.root.after(0, self.status.set, message)

    def toggle(self):
        if not self.running:
            self.start()
        else:
            self.stop()

    def start(self):
        self.running = True
        self.stop_event.clear()
        self.audio_queue = queue.Queue()
        self.transcript = []
        self.started_at = datetime.now()
        self.output_path = None

        self.text.delete("1.0", "end")
        self.start_button.config(text="■ STOP & SAVE")
        self.set_status("Starting — initializing local speech model...")
        
        # Each worker has one responsibility.
        threading.Thread(
            target=self.capture_audio, daemon=True
        ).start()

        threading.Thread(
            target=self.transcribe_audio, daemon=True
        ).start()

    def stop(self):
        if not self.running:
            return

        self.stop_event.set()
        self.start_button.config(state="disabled")
        self.set_status("Stopping capture; finishing queued audio and saving...")

    def capture_audio(self):
        """Capture desktop speaker output using Windows loopback."""
        try:
            speaker = sc.default_speaker()
            if speaker is None:
                raise RuntimeError(
                    "No default speaker was found. Check Windows sound settings."
                )

            microphone = sc.get_microphone(
                id=str(speaker.name), include_loopback=True
            )

            self.set_status(
                f"Capturing system audio from: {speaker.name}"
            )

            samples = []
            samples_in_chunk = 0
            silent_seconds = 0
            block_frames = SAMPLE_RATE

            with microphone.recorder(
                samplerate=SAMPLE_RATE
            ) as recorder:

                while not self.stop_event.is_set():
                    audio = recorder.record(numframes=block_frames)

                    if audio is None or len(audio) == 0:
                        continue

                    # Convert stereo/multichannel audio to mono.
                    audio = np.asarray(audio, dtype=np.float32)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)

                    # Optional silence-based auto-stop.
                    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))

                    if self.auto_stop.get():
                        if rms < SILENCE_THRESHOLD:
                            silent_seconds += 1
                        else:
                            silent_seconds = 0

                        if silent_seconds >= SILENCE_SECONDS:
                            self.set_status(
                                "Silence detected — finishing transcript..."
                            )
                            self.stop_event.set()
                            break

                    samples.append(audio)
                    samples_in_chunk += len(audio)

                    if samples_in_chunk >= SAMPLE_RATE * CHUNK_SECONDS:
                        chunk = np.concatenate(samples)
                        self.audio_queue.put(chunk)
                        samples = []
                        samples_in_chunk = 0

                # Send any remaining audio after Stop.
                if samples:
                    self.audio_queue.put(np.concatenate(samples))

        except Exception as exc:
            self.set_status(f"Audio capture error: {exc}")
        finally:
            # Signal the transcription worker that no more audio is coming.
            self.audio_queue.put(None)

    def transcribe_audio(self):
        """Load Whisper and transcribe each queued audio chunk in order."""
        try:
            self.set_status(
                f"Loading Whisper '{MODEL_SIZE}' model on CPU..."
            )

            model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8"
            )

            self.set_status("Listening — play your video now.")

            while True:
                chunk = self.audio_queue.get()

                if chunk is None:
                    break

                try:
                    segments, _ = model.transcribe(
                        chunk,
                        vad_filter=True,
                        beam_size=3
                    )

                    # Consume the generator while the model is active.
                    for segment in segments:
                        sentence = segment.text.strip()
                        if not sentence:
                            continue

                        start = max(0.0, float(segment.start))
                        end = max(start, float(segment.end))
                        line = f"[{self.format_time(start)}] {sentence}"

                        with self.lock:
                            self.transcript.append(line)

                        self.root.after(
                            0, self.append_text, line
                        )

                except Exception as exc:
                    self.set_status(
                        f"Transcription warning: {exc}"
                    )

            self.root.after(0, self.finish)

        except Exception as exc:
            self.root.after(0, self.model_error, str(exc))

    @staticmethod
    def format_time(seconds):
        total = int(seconds)
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    def append_text(self, line):
        self.text.insert("end", line + "\n")
        self.text.see("end")

    def finish(self):
        self.running = False
        self.start_button.config(
            text="▶ START", state="normal"
        )

        if not self.transcript:
            self.status.set(
                "Finished, but no speech was recognized. Check the audio device."
            )
            return

        try:
            folder = Path.cwd() / "transcripts"
            folder.mkdir(parents=True, exist_ok=True)

            stamp = self.started_at.strftime("%Y%m%d_%H%M%S")
            self.output_path = folder / f"transcript_{stamp}.txt"

            with open(self.output_path, "w", encoding="utf-8") as file:
                file.write("\n".join(self.transcript) + "\n")

            self.status.set(
                f"Saved {len(self.transcript)} segments: {self.output_path}"
            )

        except Exception as exc:
            self.status.set(f"Could not save transcript: {exc}")

    def model_error(self, error):
        self.running = False
        self.stop_event.set()
        self.start_button.config(text="▶ START", state="normal")
        self.status.set(f"Model error: {error}")

        messagebox.showerror(
            "Transcription error",
            "Whisper could not start or finish.\n\n"
            f"{error}\n\n"
            "Check your internet connection for the first model download "
            "and ensure faster-whisper is installed."
        )

    def clear(self):
        if self.running:
            messagebox.showinfo(
                "Transcriber running",
                "Stop the transcription before clearing the text."
            )
            return

        self.text.delete("1.0", "end")
        self.transcript = []
        self.status.set("Text cleared. Ready.")

    def save_as(self):
        content = self.text.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Save transcript", "There is no text to save.")
            return

        path = filedialog.asksaveasfilename(
            title="Save transcript",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if path:
            try:
                with open(path, "w", encoding="utf-8") as file:
                    file.write(content + "\n")
                self.status.set(f"Saved: {path}")
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))

    def close(self):
        if self.running:
            self.stop_event.set()
            self.root.destroy()
        else:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FloatingTranscriber(root)
    root.mainloop()
