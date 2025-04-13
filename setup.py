
import setuptools
import re
import os

# Get version from main.py
def get_version_from_main():
    main_path = os.path.join('extension_f5_tts', 'main.py')
    if not os.path.exists(main_path):
        return "0.0.0"  # Default version if main.py not found

    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Use regex to find the version in the extension__tts_generation_webui function
    version_match = re.search(r'"version":\s*"([^"]+)"', content)
    if version_match:
        return version_match.group(1)
    return "0.0.0"  # Default version if not found

setuptools.setup(
	name="extension_f5_tts",
    packages=setuptools.find_namespace_packages(),
	version=get_version_from_main(),
	author="rsxdalv",
	description="F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.",
	url="https://github.com/rsxdalv/extension_f5_tts",
    project_urls={},
    scripts=[],
    install_requires=[
        "gradio",
        "f5_tts @ git+https://github.com/SWivid/F5-TTS@main",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
