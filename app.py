import gradio as gr 
import time

model = whisper.load_model("base")

def inference(audio):
    
    time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(base_model.device)

    # decode the audio
    options = whisper.DecodingOptions(without_timestamps=True)
    result = whisper.decode(base_model, mel, options)
    return result.text


gr.Interface(
    title = 'Whisper-app', 
    fn=inference, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch()