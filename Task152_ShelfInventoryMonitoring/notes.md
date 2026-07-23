# Task 152: AI-Based Shelf Inventory Monitoring System

## Task Overview

This project implements a state-of-the-art, **AI-Based Shelf Inventory Monitoring System** to monitor retail and supermarket shelves, track stock levels (detecting empty or low-stock shelves), track customer interactions, and maintain transaction audit records. It integrates:
1. **YOLOv8** for real-time detection of people (customers/staff) and retail product items (bottles, cups, fruits, books, etc.).
2. **Class-Aware IoU Tracker** to assign persistent IDs and compile movement trajectories.
3. **Multi-Shelf Region of Interest (ROI) Boundaries**: Supports multiple independent polygonal shelf zones (e.g., `Shelf_Left`, `Shelf_Right`, `Upper_Shelf`, `Lower_Shelf`) with customizable normalized vertices.
4. **Behavior Analysis & Stock Change Logic**:
   - **Low Stock & Empty Alerts**: Continuously monitors the product count in each ROI. Triggers warning alerts and logs events immediately when count falls to/below `low_stock_limit` or reaches `0`.
   - **Customer Interaction Tracking**: Detects when a customer (`person` box) overlaps with a shelf ROI, logging customer arrival and departure.
   - **Stock Depletion (Purchases)**: Evaluates stock count changes before and after customer interaction. A net count decrease logs a depletion event (e.g., product removed).
   - **Stock Restocking**: A net count increase logs a replenishment/restocking event.
5. **Alerts, Evidence snapshots, and Logging**:
   - Generates and writes warn/critical alerts to stdout.
   - Appends all events to a persistent `incident_log.csv` (includes timestamps, event type, shelf name, item counts, and frame numbers).
   - Captures and saves full-resolution JPG evidence snapshots for every event transition in the outputs.
   - Exports overall metrics in `summary_stats.json`.
6. **Analytics Dashboard**: Generates a matplotlib-based PNG dashboard showing event distributions and inventory level timelines per shelf.

---

## Folder Architecture & Alignment

The project is structured inside the workspace as follows:

```text
Task152_ShelfInventoryMonitoring/
├── Code/
│   ├── detect.py                   # Main pipeline command-line script
│   ├── utils.py                    # Helper module (frame reader, tracker, logging, dashboarding)
│   └── shelf_inventory_monitoring.ipynb # Jupyter Notebook for Google Colab/local visualization
├── Inputs/
│   ├── shelf_0.mp4                 # Video 1: Supermarket shelf surveillance (1920x1080, 30 FPS, 1268 frames)
│   ├── shelf_1.mp4                 # Video 2: Close-up retail store shelf monitoring (1280x720, 30 FPS, 1211 frames)
│   └── shelf_2.mp4                 # Video 3: WiFi camera retail shelf tracking (1280x720, 30 FPS, 2509 frames)
├── Models/
│   └── yolov8n.pt                  # Pre-trained YOLOv8 weights (6.2 MB)
└── Outputs/
    ├── shelf_0_annotated.mp4       # Annotated output video for Video 1
    ├── shelf_1_annotated.mp4       # Annotated output video for Video 2
    ├── shelf_2_annotated.mp4       # Annotated output video for Video 3
    ├── incident_log.csv            # Structured CSV log database (310 rows across runs)
    ├── summary_stats.json          # Compiled summary metrics JSON
    ├── analytics_dashboard.png     # Rendered matplotlib dashboard plots
    ├── evidence_frames/            # JPG snapshots saved during transitions (310 snapshots)
    │   ├── shelf_Shelf_Left_INITIAL_STOCK_20260723_224911_frame_0.jpg
    │   └── ...
    └── inspect_frames/             # Sample periodic frame overlays for notebook display
        ├── frame_0.jpg
        ├── frame_50.jpg
        └── ...
```

---

## Verification & Execution Outcomes

We verified the pipeline on **3 real-world CCTV/video dataset footages**:

### 1. Video 1: Supermarket Shelf Surveillance (`shelf_0.mp4`)
*   **Properties**: 1920x1080 | 30.00 FPS | 1268 frames
*   **Execution Command**:
    ```bash
    python Task152_ShelfInventoryMonitoring/Code/detect.py --video Task152_ShelfInventoryMonitoring/Inputs/shelf_0.mp4
    ```
*   **Outcome**: **32 events logged**. The system successfully monitored two shelves (`Shelf_Left` and `Shelf_Right`). Since the shelves initially had no products detected in the specific ROI zones, it correctly initialized with `EMPTY` stock alerts. It tracked **14 customer interactions** as people approached and left the shelves, showing live HUD purple boxes and path lines.
*   **Performance**: ~10.39 FPS average processing speed.

### 2. Video 2: On-Shelf Availability Optimization (`shelf_1.mp4`)
*   **Properties**: 1280x720 | 30.00 FPS | 1211 frames
*   **Execution Command**:
    ```bash
    python Task152_ShelfInventoryMonitoring/Code/detect.py --video Task152_ShelfInventoryMonitoring/Inputs/shelf_1.mp4
    ```
*   **Outcome**: **94 events logged** (126 cumulative). The system monitored the `Main_Display_Shelf` zone, tracking multiple stock level transitions as products were handled.
    - Successfully triggered **39 Low Stock alerts** and **31 Empty alerts** as stock counts fluctuated.
    - Detected **2 Stock Restocking events** (e.g. stock replenished from 0 to 6 items at frame 1177) and **17 Customer Interactions**.
*   **Performance**: ~10.21 FPS average processing speed.

### 3. Video 3: WiFi Shelf Monitoring Camera (`shelf_2.mp4`)
*   **Properties**: 1280x720 | 30.00 FPS | 2509 frames
*   **Execution Command**:
    ```bash
    python Task152_ShelfInventoryMonitoring/Code/detect.py --video Task152_ShelfInventoryMonitoring/Inputs/shelf_2.mp4
    ```
*   **Outcome**: **184 events logged** (310 cumulative). Monitored two vertical levels (`Upper_Shelf` and `Lower_Shelf`).
    - Successfully captured **113 Low Stock alerts** and **94 Empty alerts** across shelves.
    - Registered **6 Stock Restockings** and **32 Customer Interactions** with high precision.
*   **Performance**: ~11.13 FPS average processing speed.

---

## How to Run

### Standalone CLI Execution
To execute the pipeline:
```bash
# Process a video with default automatic ROIs
python Task152_ShelfInventoryMonitoring/Code/detect.py --video <path_to_video>

# Process a video with custom shelf ROIs and low stock limits
python Task152_ShelfInventoryMonitoring/Code/detect.py --video Inputs/shelf_0.mp4 --shelf_rois "Shelf_A:0.05,0.2,0.45,0.2,0.45,0.9,0.05,0.9" --low_stock_limit 3
```

### Jupyter Notebook
Open `Task152_ShelfInventoryMonitoring/Code/shelf_inventory_monitoring.ipynb` in Jupyter Notebook, JupyterLab, or upload to Google Colab, mount inputs, and execute all cells to run and visualize outcomes.
