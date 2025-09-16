import gradio as gr


def extension__tts_generation_webui():
    f5_tts_ui()
    return {
        "package_name": "extension_f5_tts",
        "name": "F5-TTS",
        "requirements": "git+https://github.com/rsxdalv/extension_f5_tts@main",
        "description": "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/SWivid/F5-TTS",
        "extension_website": "https://github.com/rsxdalv/extension_f5_tts",
        "extension_platform_version": "0.0.1",
    }


def f5_tts_ui():
    from .gradio_app import ui_core

    ui_core()


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        extension__tts_generation_webui()
    demo.launch(
        server_port=7770,
    )
