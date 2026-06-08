# 🫁 DysphagiaGuard

> **Real-time swallowing disorder detection and aspiration prevention system** — an IoT-powered, AI-assisted wearable designed to protect patients with dysphagia from life-threatening silent aspiration.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [System Architecture & Data Flow](#system-architecture--data-flow)
- [Core Features](#core-features)
- [Classification Engine](#classification-engine)
- [Real-World Applications](#real-world-applications)
- [Challenges & Optimizations](#challenges--optimizations)
- [Future Roadmap](#future-roadmap)
- [Results](#results)

---

## Overview

Dysphagia (swallowing disorder) affects over **8 million people** in India and is a leading cause of aspiration pneumonia — one of the most preventable yet fatal complications in stroke, Parkinson's, and ALS patients. Current clinical assessment (Videofluoroscopy) is expensive, radiation-intensive, and episodic — offering no real-time protection between clinical visits.

**DysphagiaGuard** is a neck-worn, multi-sensor IoT device paired with an Android app that continuously classifies swallowing events in real time — distinguishing safe swallows, unsafe swallows (laryngeal penetration), and coughs (silent aspiration reflex) — and alerts caregivers instantly.

---

## Technology Stack

### Firmware (ESP32 Hardware)

| Component | Technology |
|---|---|
| Language | C++ / Arduino Framework |
| Microcontroller | ESP32 |
| Networking | `AsyncTCP` + `ESPAsyncWebServer` (async WebSocket) |
| Storage | `Preferences.h` (Non-Volatile Storage for session persistence) |
| Sensors | Analog microphone (pharyngeal acoustics) + MPU6050 IMU (laryngeal motion) |

### Android Application

| Component | Technology |
|---|---|
| Language | Kotlin |
| UI Framework | Jetpack Compose + Material 3 + Glassmorphism |
| Architecture | MVVM + Kotlin Coroutines + `StateFlow` |
| Networking | `OkHttp3` (persistent WebSocket client) |
| Local Database | Room (SQLite) — patient profiles & event history |
| PDF Generation | `iTextG` |
| Build System | Gradle (Kotlin DSL) |

---

## System Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HARDWARE LAYER (ESP32)                       │
│                                                                     │
│   [Mic (ADC)]  ──►  Peak Envelope ──►┐                              │
│   [MPU6050]    ──►  IMU RMS       ──►├──► Classification ──► JSON   │
│                     ZCR           ──►┘    Engine (50ms)    Broadcast│
└─────────────────────────────┬───────────────────────────────────────┘
                              │  WebSocket  ws://192.168.4.1/ws
                              │  100ms broadcast interval
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ANDROID APPLICATION                           │
│                                                                     │
│  OkHttp3           MonitorViewModel          Jetpack Compose UI     │
│  WebSocket  ──►   StateFlow / Flow   ──►    LiveMonitorScreen       │
│  Client            (Coroutines)              WaveformCanvas         │
│                         │                    AI Assistant           │
│                         ▼                    Alert History          │
│                   Room Database                                     │
│                   (SQLite)        ──►        PDF Report (iTextG)    │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 1 — Hardware Acquisition & Signal Processing

The ESP32 runs a continuous **20Hz (50ms) windowed analysis** loop:

1. **Sensor Polling** — Simultaneous sampling of the analog microphone (pharyngeal sounds) and MPU6050 IMU (laryngeal acceleration).
2. **Feature Extraction** per window:
   - **IMU RMS** — Root Mean Square of motion data; indicates laryngeal movement magnitude.
   - **Mic Envelope** — Peak acoustic amplitude; detects pharyngeal sound bursts.
   - **ZCR (Zero-Crossing Rate)** — Proxy for signal turbulence; critical for detecting wet gurgles and coughs.
3. **Classification** — The three features are combined against thresholds to label the window (see [Classification Engine](#classification-engine)).

### Phase 2 — WebSocket Transmission

- The ESP32 operates as a **WiFi Access Point** (`DysphagiaGuard-AP`).
- Every **100ms**, it serializes the current classification, raw sensor values, and session totals into a **JSON packet** and broadcasts it over WebSocket.

```json
{
  "event": "UNSAFE",
  "imu_rms": 0.87,
  "mic_env": 412,
  "zcr": 0.12,
  "session_safe": 4,
  "session_unsafe": 3,
  "session_cough": 1
}
```

### Phase 3 — Android Reception & Reactive UI

1. `OkHttp3` WebSocket client receives the JSON and parses it into a `SwallowEventData` Kotlin object.
2. `MonitorViewModel` updates its `StateFlow` — no polling, fully reactive.
3. **Jetpack Compose** UI components recompose automatically: waveform updates, color changes, haptic feedback — all within milliseconds of the ESP32 classification.

---

## Core Features

### 🔬 Cough vs. Swallow Classifier (The Clinical Twist)

Silent aspiration — food entering the lungs without triggering a visible swallow — is one of the most dangerous and underdiagnosed events in dysphagia patients. The only physiological reflex is often a **subtle cough**.

- **Duration threshold**: Swallows require coordinated muscle activity lasting **>100ms**. Coughs are explosive diaphragm contractions lasting **<80ms**.
- Coughs are tracked separately, highlighted in **orange**, trigger a **double-pulse vibration**, and contribute to the **Aspiration Risk Ratio**.

### 🤖 Context-Aware AI Clinical Assistant

- Reads **live session data** from `MonitorViewModel` and generates contextual safety guidance.
- Logic: If `unsafe_count >= 3`, the assistant issues a strict stop-feeding warning.
- Explains clinical terms (e.g., "Silent Aspiration", "Laryngeal Penetration") in plain language for non-medical caregivers.

### 🎭 Live Demo Engine (`DemoDataSource.kt`)

- Kotlin Coroutines-powered software simulation that mathematically generates realistic sensor signals:
  - **Safe swallows** → smooth bell-curve IMU profile
  - **Unsafe swallows** → high-amplitude IMU + mic burst
  - **Coughs** → rapid noisy double-peak
- Injected into the same UI data pipeline as real hardware — indistinguishable from live data.

### 🔁 Bi-Directional Manual Triggering

- SAFE / UNSAFE / COUGH buttons on `LiveMonitorScreen` inject simulated events for demonstration.
- When connected to hardware, these commands are **sent back to the ESP32 via WebSocket** (`ws.send("COUGH")`), syncing physical LED/motor feedback with the app in real time.

### 🚨 Emergency SMS Fallback

- After **3 consecutive UNSAFE events**, `MonitorViewModel` fires an `Intent.ACTION_SENDTO`.
- Opens the system SMS app with the caregiver's number and a pre-filled distress message — bypassing Android's background SMS restrictions.

### 🗄️ Persistent Storage & PDF Report Generation

- Every event is persisted to **Room (SQLite)**.
- `SessionRepository` computes session totals and duration on session close.
- `DailyReportScreen` pulls historical data; `iTextG` generates a formatted PDF report shareable with a Speech-Language Pathologist (SLP).

---

## Classification Engine

| Event | IMU RMS | Mic Envelope | Duration | ZCR | Meaning |
|---|---|---|---|---|---|
| `SAFE` | Moderate | Low–Moderate | 100–200ms | Low | Normal coordinated swallow |
| `UNSAFE` | High | High | 100–200ms | Low | Wet gurgle / laryngeal penetration |
| `COUGH` | Very High | High | < 80ms | High | Turbulent airflow / aspiration reflex |
| `NOISE` | — | — | Too short | — | Filtered out; low confidence |
| `IDLE` | Near-zero | Near-zero | — | — | Baseline; no event |

---

## Real-World Applications

| Domain | Use Case |
|---|---|
| 🏥 **Smart Hospital Monitoring** | Continuous non-invasive swallowing monitoring in ICUs; reduces manual supervision load |
| 👵 **Assisted Living & Elder Care** | Enables safe feeding for elderly patients; reduces dependency on constant caregiver presence |
| 🏠 **Remote Healthcare Ecosystem** | Real-time patient monitoring at home; supports telemedicine and remote diagnosis |
| 🧠 **Neurological Disorder Management** | Tracks swallowing ability over time for stroke, Parkinson's, and ALS patients |
| 🍽️ **Smart Feeding Systems** | Integrates with assistive feeding devices to ensure safe intake during meals |

---

## Challenges & Optimizations

### Design
- Accurate sensor placement on the neck region for reliable signal capture, balancing wearability against signal precision in a compact form factor.

### Hardware & Wiring
- Managing multiple interfaces (I2C, SPI, ADC) on ESP32 simultaneously.
- Ensuring stable power from Li-Po battery without voltage drops; maintaining common ground and noise isolation.

### Signal Processing
- Handling noisy and inconsistent microphone signals → **applied smoothing filters** to reduce noise.
- Synchronizing multi-sensor data in real time → **optimized sensor placement** based on laryngeal motion dynamics.

### Software
- Debugging the full end-to-end pipeline (ESP32 → Room DB → UI).
- Implemented **lightweight rule-based classification** for fast embedded processing.
- Designed an **efficient real-time data pipeline** with minimal Compose recomposition overhead.
- Optimized SQL queries for time-series dashboard performance.

---

## Future Roadmap

### 🧠 Adaptive Learning System
- Self-learning model that adapts thresholds to individual swallowing patterns over time.
- Personalized baselines built from user history stored in Room DB.

### ⚡ Early Aspiration Prediction
- Predict unsafe swallowing events **before** they fully occur using trend analysis.
- Shift from reactive detection to **preventive alerts**.

### 👥 Multi-User & Patient Profiling
- Support multiple patient profiles in a single system.
- Long-term health trend analysis and cross-session reporting.

---

## Results

> *"The system accurately captures multi-sensor data and classifies swallowing events with real-time visualization and alerts."*

- ✅ Real-time event classification at **10 Hz** with sub-100ms UI latency
- ✅ Three-class detection: Safe, Unsafe, Cough — clinically meaningful differentiation
- ✅ Fully functional Android app with persistent history, AI assistant, and PDF export
- ✅ Hardware-software bidirectional sync validated end-to-end
- ✅ Demo mode enables reliable presentation without physical hardware dependency

---

*DysphagiaGuard — Protecting every swallow, in real time.*
