# GesturePhysics

A real-time interactive physics simulation controlled entirely by hand gestures. Built on Python in PyCharm using computer vision and hand-tracking to let users grab, throw, and interact with physics-simulated objects with no mouse or keyboard required.

Developed as a final project for an **Intro to AI** course.

---

## Overview

This project uses MediaPipe's hand landmark detection to track finger positions in real time via webcam. Users can pinch to grab physics-enabled boxes, throw them around with momentum, and watch material-specific behavior play out as rubber bounces, metal sinks, and glass shatters on hard impact.

Interaction data is logged throughout the session and visualized as a chart on exit.

---

## Development Approach

This project was built using a **prompt-engineering-first workflow**. Rather than writing the codebase by hand, I applied concepts from the course, like how language models reason, where they make mistakes, and how context and specificity affect output, to direct an AI assistant through the full development process.

My role was less about writing syntax and more about:

- Designing the system architecture and feature set
- Prompting for edge case handling (e.g., two hands grabbing simultaneously, box-box collision resolution, fragment lifetime cleanup)
- Identifying where generated code was inefficient or incorrect and prompting targeted fixes
- Evaluating outputs critically and iterating

The split was roughly **35% me, 65% AI-generated**, the 35% was the part that mattered: knowing what to ask for, how to structure each request, catching what was wrong, and understanding the output well enough to direct the next step.

---

## Features

- Real-time hand tracking — detects up to 2 hands simultaneously
- Pinch-to-grab — index finger + thumb pinch picks up and moves objects
- Momentum-based throwing — release velocity tracked over time and applied on drop
- Three physics materials:
  - **Glass** — shatters into fragments on high-speed floor impact
  - **Rubber** — high bounce, lower gravity
  - **Metal** — heavy, minimal bounce
- Fragment particle system — glass shards fade out over ~1 second
- Box-box collision with stacking resolution
- Session analytics — grab events logged with timestamps, shown as a bar chart on exit
- `R` to reset, `Q` to quit

---

## Tech Stack

| Library | Purpose |
|---|---|
| `opencv-python` | Webcam capture and rendering |
| `mediapipe` | Hand landmark detection |
| `numpy` | Frame buffer |
| `pandas` | Grab event logging |
| `matplotlib` | Session analytics chart |

---

## Installation

```bash
git clone https://github.com/Olek-Z/GesturePhysics.git
cd GesturePhysics
pip install opencv-python mediapipe numpy pandas matplotlib
python main.py
```

Requires Python 3.8+ and a working webcam.

---

## Controls

| Key | Action |
|---|---|
| Pinch | Grab and move a box |
| Release pinch | Throw with momentum |
| `R` | Reset all boxes |
| `Q` | Quit and show analytics chart |
