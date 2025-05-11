# Designing an Affordable Motion Capture System for Multi-Robot Research and Education

**2025 E90 Project Report**  
**Student:** Hojune Kim  
**Advisor:** Matt Zucker

---

## 📋 Abstract

This project proposes a low-cost motion capture system for real-time tracking of multiple mobile robots in indoor environments, aimed at research and educational applications. While commercial systems like Vicon and OptiTrack cost $50,000–$150,000, our system uses three ceiling-mounted 1080p USB cameras and AprilTags to deliver approximately 10 cm spatial accuracy at 30 Hz. The system architecture includes camera calibration, pose estimation via solvePnP, coordinate fusion, and TCP-based data streaming. The system has been validated through real-to-simulation and simulation-to-real experiments involving multiple robots, providing an affordable and scalable platform for real-time multi-robot tracking.

---

## 🎯 Introduction

Multi-Robot Systems (MRS) hold vast potential across autonomous exploration, disaster response, collaborative construction, logistics, and environmental monitoring. However, real-time and accurate localization remains a critical challenge.  
This project addresses the lack of accessible motion capture solutions at smaller institutions like Swarthmore College.  
Using affordable components and open-source tools, we designed a practical system for multi-robot tracking with an operating frequency of 30 Hz and spatial accuracy of ~10 cm.  

The pipeline includes:  
- Camera calibration (intrinsic + extrinsic)  
- Pose estimation with Perspective-n-Point (PnP)  
- Coordinate fusion from multiple views  
- Real-time TCP data streaming  

Our system enables hands-on robotics experimentation without the financial barrier of commercial alternatives.

---

## 🛠️ Dependencies

- [AprilTag library](https://github.com/AprilRobotics/apriltag)  
- OpenCV  
- Python 3.x  
- NumPy  

---

## 📂 Project Structure

code/
├── 1_calibration/ # Camera calibration scripts
├── 2_server/ # Real-time TCP position server
├── 3_generate_results/ # Data analysis and result visualization


---

## 🚀 How to Use

1. Clone this repository and install the dependencies.  
2. Download and install the [AprilTag library](https://github.com/AprilRobotics/apriltag).  
3. Run camera calibration scripts in `code/1_calibration/`.  
4. Start the position server with `code/2_server/`.  
5. Collect data and generate analysis using `code/3_generate_results/`.  

Detailed instructions and example datasets will be added soon.

---

## 💡 Research Impact

This project aims to democratize motion capture capabilities for MRS research by providing an accessible and affordable alternative to expensive commercial systems. It empowers student researchers and smaller institutions to move beyond simulations and conduct meaningful experiments with physical robots.

---
