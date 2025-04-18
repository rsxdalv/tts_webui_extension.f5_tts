# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

# MIT License

# Copyright (c) 2024 Yushen CHEN

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import OrderedDict
import json
import re
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from pydub import AudioSegment

from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.list_dir_models import unload_model_button



from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

DEFAULT_TTS_MODEL = "F5-TTS_v1"

# Default model configuration as JSON
DEFAULT_MODEL_CONFIG = json.dumps(
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
)

# Default model choice JSON structure
DEFAULT_MODEL_CHOICE_JSON = json.dumps({
    "type": DEFAULT_TTS_MODEL,
    "custom": {
        "model_path": "",
        "vocab_path": "",
        "model_config": DEFAULT_MODEL_CONFIG
    }
})

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    ),
]

@manage_model_state("f5_tts_vocoder")
def load_vocoder2():
    return load_vocoder()


# Use a single model state namespace for all TTS models
@manage_model_state("f5_tts_model")
def load_f5tts(model_name="F5-TTS_v1"):
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


@manage_model_state("f5_tts_model")
def load_e2tts(model_name="E2-TTS"):
    ckpt_path = str(
        cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors")
    )
    E2TTS_model_cfg = dict(
        dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1
    )
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)


@manage_model_state("f5_tts_model")
def load_custom(model_name, vocab_path="", model_cfg=None):
    # model_name is the path to the model checkpoint
    ckpt_path = model_name.strip()
    vocab_path = vocab_path.strip() if vocab_path else ""

    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path and vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)



def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(
        ref_audio_orig, ref_text, show_info=show_info
    )

    # Load the appropriate model using the managed model state
    ema_model = None

    # Parse the model choice JSON
    try:
        model_data = json.loads(model)
        model_type = model_data["type"]

        if model_type == DEFAULT_TTS_MODEL or model_type == "F5-TTS":
            ema_model = load_f5tts("F5-TTS_v1")
        elif model_type == "E2-TTS":
            ema_model = load_e2tts("E2-TTS")
        elif model_type == "Custom":
            # Get custom model configuration from the JSON
            custom_data = model_data["custom"]
            custom_model_path = custom_data["model_path"]
            custom_vocab_path = custom_data["vocab_path"]
            custom_model_config = custom_data["model_config"]

            if not custom_model_path:
                gr.Warning("Custom model path is not set. Please configure it in the Model Selection section.")
                return gr.update(), gr.update(), ref_text

            ema_model = load_custom(custom_model_path, vocab_path=custom_vocab_path, model_cfg=json.loads(custom_model_config))
        else:
            # Default to F5-TTS for unknown model types
            gr.Warning(f"Unknown model type: {model_type}. Defaulting to {DEFAULT_TTS_MODEL}.")
            ema_model = load_f5tts("F5-TTS_v1")
    except Exception as e:
        # Handle parsing errors or other exceptions
        gr.Warning(f"Error loading model: {str(e)}. Defaulting to {DEFAULT_TTS_MODEL}.")
        ema_model = load_f5tts("F5-TTS_v1")

    vocoder = load_vocoder2()

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text


def generate_podcast(
    script,
    speaker1_name,
    ref_audio1,
    ref_text1,
    speaker2_name,
    ref_audio2,
    ref_text2,
    model,
    remove_silence,
):
    # Split the script into speaker blocks
    speaker_pattern = re.compile(
        f"^({re.escape(speaker1_name)}|{re.escape(speaker2_name)}):", re.MULTILINE
    )
    speaker_blocks = speaker_pattern.split(script)[1:]  # Skip the first empty element

    generated_audio_segments = []

    for i in range(0, len(speaker_blocks), 2):
        speaker = speaker_blocks[i]
        text = speaker_blocks[i + 1].strip()

        # Determine which speaker is talking
        if speaker == speaker1_name:
            ref_audio = ref_audio1
            ref_text = ref_text1
        elif speaker == speaker2_name:
            ref_audio = ref_audio2
            ref_text = ref_text2
        else:
            continue  # Skip if the speaker is neither speaker1 nor speaker2

        # Generate audio for this block
        audio, _, ref_text = infer(
            ref_audio, ref_text, text, model, remove_silence, 0.15, 32, 1.0, print
        )

        # Convert the generated audio to a numpy array
        sr, audio_data = audio

        # Save the audio data as a WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sr)
            audio_segment = AudioSegment.from_wav(temp_file.name)

        generated_audio_segments.append(audio_segment)

        # Add a short pause between speakers
        pause = AudioSegment.silent(duration=500)  # 500ms pause
        generated_audio_segments.append(pause)

    # Concatenate all audio segments
    final_podcast = sum(generated_audio_segments)

    # Export the final podcast
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        podcast_path = temp_file.name
        final_podcast.export(podcast_path, format="wav")

    return podcast_path


def ui_app_credits():
    gr.Markdown("""
# Credits

* [mrfakename](https://github.com/fakerybakery) for the original [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) for initial chunk generation and podcast app exploration
* [jpgallegoar](https://github.com/jpgallegoar) for multiple speech-type generation & voice chat
""")


def create_model_selection_ui():
    """Create a reusable model selection component that can be added to any tab."""
    # Parse the default model choice JSON
    default_model_data = json.loads(DEFAULT_MODEL_CHOICE_JSON)

    # Create a JSON state to store the model configuration
    model_choice_json = gr.State(DEFAULT_MODEL_CHOICE_JSON)

    with gr.Accordion("Model Selection", open=False):
        with gr.Row():
            model_type = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS", "Custom"],
                label="Choose TTS Model",
                value=default_model_data["type"],
                info="Select the model to use"
            )
            with gr.Column(scale=1):
                unload_btn = unload_model_button("tts_model")

        with gr.Group(visible=default_model_data["type"] == "Custom") as custom_model_group:
            gr.Markdown("### Custom Model Configuration")
            custom_model_path_input = gr.Textbox(
                label="Model Path",
                value=default_model_data["custom"]["model_path"],
                placeholder="Path to model checkpoint (local path or Hugging Face path starting with hf://)",
                info="Example: hf://username/model-name/model.safetensors or C:/path/to/model.safetensors"
            )
            custom_vocab_path_input = gr.Textbox(
                label="Vocabulary Path (Optional)",
                value=default_model_data["custom"]["vocab_path"],
                placeholder="Path to vocabulary file (leave empty to use default)",
                info="Example: hf://username/model-name/vocab.txt or C:/path/to/vocab.txt"
            )
            custom_model_config_input = gr.Textbox(
                label="Model Configuration (JSON)",
                value=default_model_data["custom"]["model_config"],
                info="JSON configuration for the model architecture"
            )

        # Show/hide custom model configuration based on selection
        def update_custom_model_visibility(model_type):
            return gr.update(visible=model_type == "Custom")

        model_type.change(
            update_custom_model_visibility,
            inputs=[model_type],
            outputs=[custom_model_group]
        )

        # Update model choice JSON when any input changes
        def update_model_choice_json(model_type, path, vocab_path, config, current_json):
            # Parse the current JSON
            model_data = json.loads(current_json)

            # Update the model type
            model_data["type"] = model_type

            # Update custom model data if needed
            if model_type == "Custom":
                model_data["custom"]["model_path"] = path
                model_data["custom"]["vocab_path"] = vocab_path
                model_data["custom"]["model_config"] = config

            # Return the updated JSON as a string
            return json.dumps(model_data)

        # Connect all input changes to update the JSON
        model_type.change(
            update_model_choice_json,
            inputs=[model_type, custom_model_path_input, custom_vocab_path_input, custom_model_config_input, model_choice_json],
            outputs=[model_choice_json]
        )

        custom_model_path_input.change(
            update_model_choice_json,
            inputs=[model_type, custom_model_path_input, custom_vocab_path_input, custom_model_config_input, model_choice_json],
            outputs=[model_choice_json]
        )

        custom_vocab_path_input.change(
            update_model_choice_json,
            inputs=[model_type, custom_model_path_input, custom_vocab_path_input, custom_model_config_input, model_choice_json],
            outputs=[model_choice_json]
        )

        custom_model_config_input.change(
            update_model_choice_json,
            inputs=[model_type, custom_model_path_input, custom_vocab_path_input, custom_model_config_input, model_choice_json],
            outputs=[model_choice_json]
        )

    return model_choice_json


def ui_app_tts():
    gr.Markdown("# Batched TTS")
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10)

    # Add model selection component
    model_choice = create_model_selection_ui()

    generate_btn = gr.Button("Synthesize", variant="primary")
    with gr.Accordion("Advanced Settings", open=False):
        ref_text_input = gr.Textbox(
            label="Reference Text",
            info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
            value=False,
        )
        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        nfe_slider = gr.Slider(
            label="NFE Steps",
            minimum=4,
            maximum=64,
            value=32,
            step=2,
            info="Set the number of denoising steps.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )

    audio_output = gr.Audio(label="Synthesized Audio")
    spectrogram_output = gr.Image(label="Spectrogram")

    generate_btn.click(
        infer,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            model_choice,  # Use the local model choice
            remove_silence,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider,
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input],
    )


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments


def ui_app_podcast():
    with gr.Blocks() as app_podcast:
        gr.Markdown("# Podcast Generation")
        speaker1_name = gr.Textbox(label="Speaker 1 Name")
        ref_audio_input1 = gr.Audio(
            label="Reference Audio (Speaker 1)", type="filepath"
        )
        ref_text_input1 = gr.Textbox(label="Reference Text (Speaker 1)", lines=2)

        speaker2_name = gr.Textbox(label="Speaker 2 Name")
        ref_audio_input2 = gr.Audio(
            label="Reference Audio (Speaker 2)", type="filepath"
        )
        ref_text_input2 = gr.Textbox(label="Reference Text (Speaker 2)", lines=2)

        script_input = gr.Textbox(
            label="Podcast Script",
            lines=10,
            placeholder="Enter the script with speaker names at the start of each block, e.g.:\nSean: How did you start studying...\n\nMeghan: I came to my interest in technology...\nIt was a long journey...\n\nSean: That's fascinating. Can you elaborate...",
        )

        # Add model selection component
        podcast_model_choice = create_model_selection_ui()

        podcast_remove_silence = gr.Checkbox(
            label="Remove Silences",
            value=True,
        )
        generate_podcast_btn = gr.Button("Generate Podcast", variant="primary")
        podcast_output = gr.Audio(label="Generated Podcast")

        def podcast_generation(
            script,
            speaker1,
            ref_audio1,
            ref_text1,
            speaker2,
            ref_audio2,
            ref_text2,
            model,
            remove_silence,
        ):
            return generate_podcast(
                script,
                speaker1,
                ref_audio1,
                ref_text1,
                speaker2,
                ref_audio2,
                ref_text2,
                model,
                remove_silence,
            )

        generate_podcast_btn.click(
            podcast_generation,
            inputs=[
                script_input,
                speaker1_name,
                ref_audio_input1,
                ref_text_input1,
                speaker2_name,
                ref_audio_input2,
                ref_text_input2,
                podcast_model_choice,  # Use the local model choice
                podcast_remove_silence,
            ],
            outputs=podcast_output,
        )

# Keep track of autoincrement of speech types, no roll back
speech_type_count = 1

def ui_app_emotional():
    # New section for multistyle generation
    gr.Markdown(
        """
    # Multiple Speech-Type Generation

    This section allows you to generate multiple speech types or multiple people's voices. Enter your text in the format shown below, and the system will generate speech using the appropriate type. If unspecified, the model will use the regular speech type. The current speech type will be used until the next speech type is specified.
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Example Input:**
            {Regular} Hello, I'd like to order a sandwich please.
            {Surprised} What do you mean you're out of bread?
            {Sad} I really wanted a sandwich though...
            {Angry} You know what, darn you and your little shop!
            {Whisper} I'll just go back home and cry now.
            {Shouting} Why me?!
            """
        )

        gr.Markdown(
            """
            **Example Input 2:**
            {Speaker1_Happy} Hello, I'd like to order a sandwich please.
            {Speaker2_Regular} Sorry, we're out of bread.
            {Speaker1_Sad} I really wanted a sandwich though...
            {Speaker2_Whisper} I'll give you the last one I was hiding.
            """
        )

    gr.Markdown(
        "Upload different audio clips for each speech type. The first speech type is mandatory. You can add additional speech types by clicking the 'Add Speech Type' button."
    )

    gr.Markdown(
        "Emotional is limited to 9 speech types due to low performance causing the entire app to crash"
    )
    # Regular speech type (mandatory)
    with gr.Row() as regular_row:
        with gr.Column():
            regular_name = gr.Textbox(value="Regular", label="Speech Type Name")
            regular_insert = gr.Button("Insert Label", variant="secondary")
        regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")
        regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=2)

    # Regular speech type (max 100)
    max_speech_types = 9
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_audios = [regular_audio]
    speech_type_ref_texts = [regular_ref_text]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]

    # Additional speech types (99 more)
    for i in range(max_speech_types - 1):
        with gr.Row(visible=False) as row:
            with gr.Column():
                name_input = gr.Textbox(label="Speech Type Name")
                delete_btn = gr.Button("Delete Type", variant="secondary")
                insert_btn = gr.Button("Insert Label", variant="secondary")
            audio_input = gr.Audio(label="Reference Audio", type="filepath")
            ref_text_input = gr.Textbox(label="Reference Text", lines=2)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Speech Type")

    # Function to add a speech type
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning("Exhausted maximum number of speech types. Consider restart the app.")
        return row_updates

    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # Function to delete a speech type
    def delete_speech_type_fn():
        return gr.update(visible=False), None, None, None

    # Update delete button clicks
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[speech_type_rows[i], speech_type_names[i], speech_type_audios[i], speech_type_ref_texts[i]],
        )

    # Text input for the prompt
    gen_text_input_multistyle = gr.Textbox(
        label="Text to Generate",
        lines=10,
        placeholder="Enter the script with speaker names (or emotion types) at the start of each block, e.g.:\n\n{Regular} Hello, I'd like to order a sandwich please.\n{Surprised} What do you mean you're out of bread?\n{Sad} I really wanted a sandwich though...\n{Angry} You know what, darn you and your little shop!\n{Whisper} I'll just go back home and cry now.\n{Shouting} Why me?!",
    )

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            speech_type_name = speech_type_name or "None"
            updated_text = current_text + f"{{{speech_type_name}}} "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
        )

    # Add model selection component
    emotional_model_choice = create_model_selection_ui()

    with gr.Accordion("Advanced Settings", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="Remove Silences",
            value=True,
        )

    # Generate button
    generate_multistyle_btn = gr.Button("Generate Multi-Style Speech", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Synthesized Audio")

    def generate_multistyle_speech(
        gen_text,
        *args,
    ):
        speech_type_names_list = args[:max_speech_types]
        speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
        remove_silence = args[3 * max_speech_types]
        # Collect the speech types and their audios into a dict
        speech_types = OrderedDict()

        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_style = "Regular"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in speech_types:
                current_style = style
            else:
                gr.Warning(f"Type {style} is not available, will use Regular as default.")
                current_style = "Regular"

            try:
                ref_audio = speech_types[current_style]["audio"]
            except KeyError:
                gr.Warning(f"Please provide reference audio for type {current_style}.")
                return [None] + [speech_types[style]["ref_text"] for style in speech_types]
            ref_text = speech_types[current_style].get("ref_text", "")

            # Generate speech for this segment
            audio_out, _, ref_text_out = infer(
                ref_audio, ref_text, text, emotional_model_choice, remove_silence, 0, 32, 1.0, print
            )  # show_info=print no pull to top when generating
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_style]["ref_text"] = ref_text_out

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return [(sr, final_audio_data)] + [speech_types[style]["ref_text"] for style in speech_types]
        else:
            gr.Warning("No audio generated.")
            return [None] + [speech_types[style]["ref_text"] for style in speech_types]

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            remove_silence_multistyle,
        ],
        outputs=[audio_output_multistyle] + speech_type_ref_texts,
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["style"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


def ui_core():
    gr.Markdown(
        """
# E2/F5 TTS

This is a local web UI for [F5 TTS](https://github.com/SWivid/F5-TTS) with advanced batch processing support. This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The checkpoints currently support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 12s with  ✂  in the bottom right corner (otherwise might have non-optimal auto-trimmed result).

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<12s). Ensure the audio is fully uploaded before generating.**
"""
    )
    with gr.Tabs():
        with gr.Tab("TTS"):
            ui_app_tts()
        with gr.Tab("Podcast"):
            ui_app_podcast()
        with gr.Tab("Multi-Style"):
            ui_app_emotional()
        with gr.Tab("Credits"):
            ui_app_credits()


def ui_app():
    with gr.Blocks() as app:
        ui_core()
    return app
