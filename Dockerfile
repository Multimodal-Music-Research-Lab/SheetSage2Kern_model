FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies:
#   ffmpeg / libsm6      - audio/video decoding for datasets & librosa
#   fluidsynth           - MIDI synthesis (pretty_midi / music21)
#   rubberband-cli       - required by pyrubberband (data augmentation)
#   git, build-essential - build tooling for pip packages
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsm6 \
        fluidsynth \
        rubberband-cli \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install pybind11

WORKDIR /workspace

# Install all Python dependencies (torch, lightning, datasets, muq, ...) in one shot.
# This installs the exact stack the code was developed against, so the image is
# ready to train/evaluate straight after `docker build` — no manual pip steps.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project source
COPY . .
