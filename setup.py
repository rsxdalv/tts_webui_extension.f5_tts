
import setuptools

setuptools.setup(
	name="extension_f5_tts",
    packages=setuptools.find_namespace_packages(),
	version="0.1.1",
	author="rsxdalv",
	description="F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching.",
	url="https://github.com/rsxdalv/extension_f5_tts",
    project_urls={},
    scripts=[],
    install_requires=[
        "gradio",
        "f5_tts @ git+https://github.com/rsxdalv/F5-TTS@main",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
