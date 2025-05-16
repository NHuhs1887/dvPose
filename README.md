# dvPose

**dvPose** is a research-oriented repository for collecting and comparing methods for **online human pose estimation** using **neuromorphic event cameras**, specifically the **Inivation DVXplorer** and **DAVIS346**.

This project leverages the [`hpe-core`](https://github.com/event-driven-robotics/hpe-core/tree/178ac06fbf35f4b6ea9aca1a2a35bb6906b35213) library, which provides a framework for high-frequency human pose estimation with event cameras. It uses a **pretrained MoveNet model** adapted for asynchronous data streams.

---

## Overview

Modern event-based cameras, like DVXplorer and DAVIS346, capture changes in brightness at a microsecond resolution, enabling ultra-low latency and low power computation — ideal for real-time pose estimation tasks. This repository is designed to evaluate and compare various pose estimation methods that process such data streams online.

### Key Features

- Event-driven Pose Estimation: Utilizes asynchronous data from neuromorphic cameras.
- MoveNet Integration: Based on a pretrained MoveNet model for pose inference.
- hpe-core Integration: Builds upon the [hpe-core library](https://github.com/event-driven-robotics/hpe-core/tree/178ac06fbf35f4b6ea9aca1a2a35bb6906b35213) to provide a high-frequency, low-latency pose estimation pipeline.
- Camera Support: Tested with both DVXplorer and DAVIS346 cameras.
- Evaluation and Comparison: Designed to collect results from multiple algorithms for comparative analysis.

---

## Related Paper

This repository is heavily inspired by and builds upon the findings from the following paper:

**[MoveEnet: Online High-Frequency Human Pose Estimation With an Event Camera – CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Goyal_MoveEnet_Online_High-Frequency_Human_Pose_Estimation_With_an_Event_Camera_CVPRW_2023_paper.pdf)**  
*by Kunal Goyal, Ryad Benosman, Alessio Paolillo, and Yulia Sandamirskaya*

---

## Installation

> Detailed setup instructions coming soon.

Basic requirements:
- Python 3.8+
- [hpe-core](https://github.com/event-driven-robotics/hpe-core)
- PyTorch
- dv-processing SDK (for Inivation cameras)

---
