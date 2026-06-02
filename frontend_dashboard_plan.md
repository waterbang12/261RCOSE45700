# Frontend Requirements for Unmanned Physical AI Environment Control

## Purpose of This Document

This document explains what frontend screens may be needed for the real product, why each screen exists, and which parts should be discussed with teammates/advisors before implementation.

The project is not just a web dashboard. It is an unmanned-store physical AI system:

```text
camera / sensor / voice input
→ event signal
→ state machine / AI decision
→ actuator command
→ Arduino or real environment-control hardware
```

Therefore, the frontend should support three different users:

1. Customer using the unmanned kiosk
2. Administrator/operator monitoring the store
3. System/developer during demo and debugging

---

## Product Frontend Split

Recommended frontend structure:

```text
/customer   Customer kiosk UI
/admin      Administrator dashboard
/demo       Developer/demo control panel
```

The frontend should not directly control Arduino or the camera. It should send requests to the backend, and the backend should route everything through the state machine and actuator layer.

```text
Frontend
→ FastAPI backend
→ state machine / decision logic
→ Arduino actuator
→ physical environment
```

---

# 1. Customer Kiosk Frontend

## What It Is

The customer kiosk is the screen that a visitor sees inside the unmanned store.

It should provide simple interaction with the environment, similar to a digital staff assistant.

## Why It Is Needed

In an unmanned store, there is no employee immediately available. The kiosk gives the customer a way to:

- Request help
- Report discomfort
- Ask for environment changes
- Understand what the system is doing
- Receive feedback after AI/control actions

This is especially important if the system controls real hardware such as fan, lighting, AC, relay, ventilation, or warning devices.

## Required Features

### 1. Main Status Screen

Display:

- Store or zone name
- Current state: Normal / Alert / Cooling / Help requested
- Current environment status if available
- System availability

Example:

```text
Welcome
Zone: Entrance
Environment: Normal
AI monitoring: Active
Actuator state: NORMAL
```

Reason:

Customers need to know the system is active and their request can be handled.

---

### 2. Voice Input Button

Display a large microphone button:

```text
Press to speak
```

Example customer speech:

```text
"너무 더워요"
"도움이 필요해요"
"조명이 너무 어두워요"
"문제가 있어요"
```

Flow:

```text
Customer voice
→ stt.py
→ text transcript
→ customer agent / local rule
→ response + possible actuator action
```

Reason:

Voice is natural for kiosk interaction, and the project already has `stt.py` connected.

---

### 3. Manual Request Buttons

Voice can fail, so buttons are also needed.

Suggested buttons:

- Too hot
- Too cold
- Need help
- Report problem
- Emergency
- Cancel / Reset

Reason:

Buttons make the kiosk more reliable for demos and real users. They also help users who do not want to speak.

---

### 4. Transcript and AI Response Area

Display:

```text
You said: 너무 더워요
AI response: 온도를 낮추겠습니다. 잠시만 기다려 주세요.
Action: Cooling requested
```

Reason:

The customer needs feedback that the system understood the request.

---

### 5. Environment Action Feedback

Display what changed:

```text
Fan: ON
Cooling request: Sent
Arduino command: ALERT
```

For the current demo:

```text
ALERT → Arduino LED ON
NORMAL → Arduino LED OFF
```

Reason:

The project is about physical AI control, so the UI should show the connection between request and physical action.

---

### 6. Privacy / Camera Notice

If camera detection is used, the kiosk should show a simple notice:

```text
This store uses camera/sensor-based AI monitoring for safety and environment control.
```

Reason:

Camera-based AI requires user trust and may require a visible notice.

---

## Customer Kiosk MVP

Minimum version:

- Microphone button
- Transcript text
- AI response text
- Buttons: Too Hot / Too Cold / Need Help / Emergency
- Current actuator state: NORMAL / ALERT

---

# 2. Administrator Dashboard

## What It Is

The admin dashboard is for the store owner, manager, or operator.

It is not meant for the customer.

## Why It Is Needed

The administrator needs to monitor the system, review events, override AI actions, and understand whether the physical environment-control system is working.

## Required Features

### 1. Store Overview

Display:

```text
Entrance Zone: Normal
Customer Area: Heat discomfort detected
Arduino: Connected
Camera: Connected
OpenAI: Available
Decision mode: Hybrid
```

Reason:

Admin needs a quick view of the whole system state.

---

### 2. Camera / Sensor Event Panel

Display:

- Camera feed or latest frame
- Mock/camera/sensor event source
- Event type
- Confidence score
- Store zone

Example:

```text
Event: heat_sweat_discomfort
Zone: entrance_zone
Confidence: 0.90
Source: camera
```

Reason:

Admin needs to understand why the system made a decision.

---

### 3. Event Log

Display chronological logs:

```text
14:02:11 event generated: heat_sweat_discomfort
14:02:11 state machine received event
14:02:11 local rule decision: ALERT
14:02:11 Arduino response: OK ALERT LED ON
14:02:18 OpenAI explanation generated
```

Reason:

Logs are necessary for debugging, review, and advisor presentation.

---

### 4. Manual Override Controls

Admin controls:

- Force ALERT
- Force NORMAL
- Turn fan ON
- Turn fan OFF
- Reset Arduino
- Disable automation

Reason:

Physical systems need human override. The AI should not be the only way to control hardware.

---

### 5. Decision Mode Selector

Options:

- Local Rule
- OpenAI Agent
- Hybrid
- Mock Agent

Recommended default:

```text
Hybrid
```

Meaning:

- Local Rule: fastest physical control
- OpenAI Agent: slower but better explanation
- Hybrid: local control first, OpenAI background analysis
- Mock Agent: offline testing

Reason:

The latency experiment showed that OpenAI is too slow for immediate actuator control, but still useful for explanation and reports.

---

### 6. Latency Monitor

Display:

- Total latency
- Queue latency
- Agent decision latency
- Arduino round-trip latency
- OpenAI latency if used

Example:

```text
Total: 45ms
Queue: 1ms
Decision: 2ms
Arduino: 0.3ms
```

Reason:

Latency is a major evaluation point for physical AI control.

---

### 7. System Health Panel

Display:

- Arduino connected / mock mode
- Camera connected / disconnected
- Microphone available / unavailable
- OpenAI available / unavailable
- Last error

Reason:

This helps operators know if the system can be trusted.

---

## Admin Dashboard MVP

Minimum version:

- Event log
- Arduino status
- Manual ALERT / NORMAL buttons
- Decision mode selector
- Latency table
- Camera/mock event status

---

# 3. Demo / Developer Control Panel

## What It Is

This is a presentation/debug screen for the team.

It can be separate from the real product UI.

## Why It Is Needed

During a school presentation, real camera detection, microphone, OpenAI, and Arduino may not always behave perfectly. A demo panel allows the team to prove each pipeline step reliably.

## Required Features

### 1. Manual Event Triggers

Buttons:

- Customer discomfort
- Heat / sweat discomfort
- Occlusion detected
- Abnormal behavior
- Unsafe environment
- Reset / Normal

Reason:

This allows demo without depending on real camera detection.

---

### 2. Pipeline Visualization

Show:

```text
Event created
→ queued
→ state machine received
→ decision made
→ actuator command sent
→ Arduino response received
```

Reason:

This explains the architecture better than terminal logs.

---

### 3. Side-by-Side Decision Comparison

Optional but useful:

```text
Local Rule result: ALERT, 5ms
OpenAI result: ALERT, 8000ms
Hybrid result: ALERT immediately + explanation later
```

Reason:

This directly supports the conclusion that physical control should be local/hybrid.

---

## Demo Panel MVP

Minimum version:

- Manual event buttons
- Pipeline status
- Arduino status
- Latest latency result
- Local/OpenAI/Hybrid selector

---

# Suggested Backend Endpoints

The frontend should call backend APIs like:

```text
GET  /status
GET  /events
GET  /latency
GET  /video-feed
POST /event
POST /actuator/alert
POST /actuator/normal
POST /mode
POST /voice/transcribe
```

The backend should own the actual logic:

```text
frontend request
→ backend
→ state machine
→ decision layer
→ actuator
```

---

# Suggested Technology Stack

Recommended for this project:

```text
Backend: FastAPI
Frontend: HTML + JavaScript or React
Realtime updates: Server-Sent Events or WebSocket
Camera stream: MJPEG or latest-frame polling
Voice input: browser mic or backend mic → stt.py
Arduino control: arduino_actuator.py
Latency tracking: latency_profiler.py
AI decision: local rule / OpenAI / hybrid
```

For fastest school-demo implementation:

```text
FastAPI + simple HTML dashboard
```

React can be added later if the UI needs to look more polished.

---

# Recommended First Build

Build these first:

## Customer Page

- Big microphone button
- Request buttons
- Transcript
- AI response
- Current actuator state

## Admin Page

- Event log
- Arduino status
- ALERT / NORMAL override
- Decision mode selector
- Latency table

## Demo Page

- Manual event buttons
- Pipeline status
- Latest latency result

---

# Advisor Feedback Questions

Ask advisors these questions:

1. Should the first frontend focus on the customer kiosk, admin dashboard, or demo dashboard?
2. For a real unmanned store, what should the customer be allowed to control directly?
3. Should customers see camera/detection status, or should that be admin-only?
4. Is voice input important enough for the MVP, or should buttons come first?
5. Should emergency/help requests bypass AI and immediately notify admin?
6. Should physical actuator control use local rule by default, with OpenAI only in the background?
7. What environment device should replace the Arduino LED first: fan, relay, warning light, or IR AC control?
8. Should the admin be able to override all AI decisions?
9. Is AWS/cloud backend necessary for the project scope, or should it remain local for now?
10. What privacy notice is required if camera-based AI is used in the kiosk?

---

# Key Recommendation

The frontend should support the real product as a kiosk system, not only as a debug dashboard.

Recommended real product structure:

```text
Customer kiosk:
voice + buttons + response + environment feedback

Admin dashboard:
monitoring + override + logs + latency + system health

Demo panel:
manual events + pipeline visualization + comparison
```

Recommended control architecture:

```text
local rule / on-device AI
→ immediate Arduino or environment control
→ OpenAI/AWS background explanation and reporting
```

This gives fast physical response while still using AI for higher-level reasoning.
