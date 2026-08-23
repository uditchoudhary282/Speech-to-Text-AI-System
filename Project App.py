import gradio as gr
from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tempfile, os
from scipy.io.wavfile import write

model = WhisperModel("small", compute_type="int8")
an = SentimentIntensityAnalyzer()

def transcribe_plus(audio, export_srt=True, denoise=False, add_summary=False):
    if audio is None:
        return "", "neutral (0.000)", None, "", ""

    sr, data = audio
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(f.name, sr, data)
    path = f.name

    if denoise:
        try:
            import librosa, noisereduce as nr, soundfile as sf
            y, sr2 = librosa.load(path, sr=None)
            y2 = nr.reduce_noise(y=y, sr=sr2)
            sf.write(path, y2, sr2)
        except Exception:
            pass

    segments, info = model.transcribe(path, vad_filter=True, word_timestamps=False)
    os.unlink(path)

    text = " ".join(s.text for s in segments).strip()
    vs = an.polarity_scores(text)["compound"] if text else 0.0
    lab = "positive" if vs >= 0.05 else ("negative" if vs <= -0.05 else "neutral")
    sentiment = f"{lab} ({vs:.3f})"

    srt_file = None
    if export_srt and text:
        def fmt(x):
            h = int(x // 3600); m = int((x % 3600) // 60); s = x % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
        lines = []
        t = 1
        for seg in segments:
            lines.append(f"{t}\n{fmt(seg.start)} --> {fmt(seg.end)}\n{seg.text.strip()}\n")
            t += 1
        with open("out.srt", "w") as g:
            g.write("\n".join(lines))
        srt_file = "out.srt"

    summary = ""
    if add_summary and text:
        try:
            from transformers import pipeline
            summ = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summ(text[:2000], max_length=70, min_length=20, do_sample=False)[0]["summary_text"]
        except Exception:
            summary = ""

    return text, sentiment, srt_file, (info.language if text else ""), summary

demo = gr.Interface(
    fn=transcribe_plus,
    inputs=[
        gr.Audio(sources=["microphone"], streaming=False, label="mic"),
        gr.Checkbox(label="export SRT", value=True),
        gr.Checkbox(label="denoise", value=False),
        gr.Checkbox(label="add summary", value=False),
    ],
    outputs=[
        gr.Textbox(label="transcript"),
        gr.Textbox(label="sentiment"),
        gr.File(label="subtitles (.srt)"),
        gr.Textbox(label="language"),
        gr.Textbox(label="summary"),
    ],
    title="speech to text model with ai summary and denoising",
    description="Speak or upload audio to get transcript, sentiment, optional denoising, summary, and SRT export."
)

if __name__ == "__main__":
    demo.launch()
