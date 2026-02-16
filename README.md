# Grand Brother: Edge AI Identity System 👁️

**PhD Research Project: Distributed Knowledge Generation at the Edge**
*Author: Andrés (PhD Candidate) | Context: School Environment Analysis*

## Overview
This project implements a hybrid Computer Vision architecture running on a **Raspberry Pi 5**. It combines **YOLOv8** (for robust human detection) and **HOG/CNN** (for precise face recognition) to monitor student interactions in real-time without compromising privacy.

**Key Features:**
* **Edge Computing:** All video processing happens on-device. No video stream leaves the local network.
* **Privacy Preserving:** Only metadata (timestamps, IDs) is logged; images are discarded immediately.
* **Hybrid Architecture:** Uses YOLOv8n for speed (5 FPS on CPU) and dlib for identity verification.

## Hardware Requirements
* Raspberry Pi 5 (8GB RAM recommended)
* Active Cooling System (Required for YOLO inference)
* USB Camera / Pi Camera Module 3

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/Dyrtull/grand-brother-edge.git](https://github.com/Dyrtull/grand-brother-edge.git)
   cd grand-brother-edge
