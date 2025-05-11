# Designing an Affordable Motion Capture System for Multi-Robot Research and Education

**2025 E90 Project Report**  
**Student:** Hojune Kim  
**Advisor:** Matt Zucker

---

## 🎯 Introduction
This project addresses the lack of accessible motion capture solutions at smaller institutions like Swarthmore College or student-led robotics club. Using Apriltag library, we designed a practical system for multi-robot tracking with an operating frequency of 30 Hz and spatial accuracy of ~5 cm for covering 2.5 m × 4.3 m room space using three logitech webcams.

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
```
code/
├── 1_calibration/ # Camera calibration scripts
├── 2_server/ # Real-time TCP position server
├── 3_generate_results/ # Data analysis and result visualization
```
---

## 🚀 How to Use

1. Download and install the [AprilTag library](https://github.com/AprilRobotics/apriltag).  
2. Code for both intrinsic and extrinsic calibrations can be found in `code/1_calibration/`.  
   - Make sure to modify variables such as chessboard size, camera names, folder paths, etc., to match your setup.
3. Use the code in `code/2_server/` to host the server and establish connections with client devices.
4. Collect data and perform analysis using the scripts in `code/3_generate_results/`.

Detailed instructions will be added soon.

---

## 💡 Research Impact

This project aims to democratize motion capture capabilities for MRS research by providing an accessible and affordable alternative to expensive commercial systems. It empowers student researchers and smaller institutions to move beyond simulations and conduct meaningful experiments with physical robots.

---
