# Task 152: AI-Based Shelf Inventory Monitoring System

An AI-based computer vision system designed to monitor store shelves, detect empty or low-stock shelves in real-time, track customer interaction behaviors, and maintain transaction audit logs.

This system leverages **YOLOv8** for real-time object detection and integrates custom class-aware **Intersection-over-Union (IoU) Tracking** with geometric and temporal heuristics to monitor shelf stock levels and analyze customer interaction events (e.g. stock replenishment or purchase depletions).

---

## Key Features

1. **Object Detection**: Identifies products on shelves (`bottle`, `cup`, `apple`, `orange`, `banana`, `book`, etc.) and people (`person`) using a pre-trained YOLOv8 model.
2. **Class-Aware IoU Tracking**: Tracks products and customers across frames to assign persistent IDs and compile centroid histories.
3. **Multi-Shelf Region of Interest (ROI) Boundaries**: Supports multiple independent polygonal shelf zones (e.g. `Shelf_Left`, `Shelf_Right`, `Upper_Shelf`, `Lower_Shelf`) configured using normalized coordinates.
4. **Behavior & Stock Level Analysis**:
   - **Low Stock & Empty Alerts**: Continuously monitors shelf counts. Triggers warning alerts and logs events immediately when a shelf transitions to or below `low_stock_limit` or reaches `0`.
   - **Customer Interaction Tracking**: Detects when a customer (`person` box) overlaps with a shelf ROI, registering customer arrival and departure.
   - **Stock Depletion (Purchases)**: Computes inventory count difference before and after customer interactions. A net decrease logs a stock depletion event.
   - **Stock Restocking**: A net increase logs a stock replenishment event.
5. **Real-time Alerting & Evidence Logging**:
   - Prints console alerts.
   - Captures and saves full-resolution JPG evidence snapshots for every event transition.
   - Updates a cumulative CSV log database.
6. **Analytics Dashboard**: Generates visual dashboard plots showing event distributions and inventory level timelines per shelf.

---

## Folder Architecture

The project is structured inside the workspace as follows:

```text
Shelf Inventory Monitoring/
├── .gitignore
├── README.md
└── Task152_ShelfInventoryMonitoring/
    ├── notes.md                          # Logs results, video specs, and performance stats
    ├── Code/
    │   ├── utils.py                      # Helper utilities (tracker, geometry, plotting, logging)
    │   ├── detect.py                     # Main pipeline command-line execution script
    │   └── shelf_inventory_monitoring.ipynb # Jupyter Notebook for Google Colab/local visualization
    ├── Models/
    │   └── yolov8n.pt                    # Pre-trained YOLOv8 weights (downloaded/copied)
    └── Outputs/                          # Incident CSV logs, summary statistics, and plots
        ├── evidence_frames/              # JPG snapshots of shelf transitions
        └── ...
```

---

## Getting Started

### 1. Installation

Ensure you have Python 3.8+ installed. Install the necessary libraries:

```bash
pip install ultralytics opencv-python pandas matplotlib
```

### 2. Execution

To run the pipeline on a video file with default ROIs:

```bash
python Task152_ShelfInventoryMonitoring/Code/detect.py --video <path_to_video>
```

You can customize the shelf ROIs and low stock thresholds via CLI flags:

```bash
python Task152_ShelfInventoryMonitoring/Code/detect.py --video Task152_ShelfInventoryMonitoring/Inputs/shelf_0.mp4 --shelf_rois "Shelf_A:0.05,0.2,0.45,0.2,0.45,0.9,0.05,0.9" --low_stock_limit 3
```

Refer to `detect.py --help` for all available options.

### 3. Google Colab / Jupyter Notebook

Open `Task152_ShelfInventoryMonitoring/Code/shelf_inventory_monitoring.ipynb` in Jupyter Notebook, JupyterLab, or upload to Google Colab, mount inputs, and execute all cells sequentially to run and visualize outcomes.
