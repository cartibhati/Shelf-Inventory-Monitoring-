import os
import cv2
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def video_frame_generator(video_path):
    """
    Opens a video file OR a directory containing sequential images, and yields frames 
    sequentially along with sequence metadata.
    
    Args:
        video_path (str): Path to the input video file or image sequence directory.
        
    Yields:
        tuple: (frame, frame_idx, fps, frame_count, width, height)
            - frame (numpy.ndarray): The current video frame.
            - frame_idx (int): The current frame index (0-based).
            - fps (float): Frame rate of the video.
            - frame_count (int): Total number of frames in the sequence.
            - width (int): Width of the video frame.
            - height (int): Height of the video frame.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input path not found at: {video_path}")
        
    # Check if the path is a directory of images
    if os.path.isdir(video_path):
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(video_path, ext)))
            image_files.extend(glob.glob(os.path.join(video_path, ext.upper())))
            
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in directory: {video_path}")
            
        frame_count = len(image_files)
        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            raise IOError(f"Could not read the first image frame: {image_files[0]}")
        height, width = first_frame.shape[:2]
        fps = 30.0 # Default fallback FPS for image sequences
        
        for frame_idx, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Warning: Could not read frame image: {img_path}")
                continue
            yield frame, frame_idx, fps, frame_count, width, height
            
    else:
        # Standard video file input
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"OpenCV was unable to open the video file at: {video_path}")
            
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps is None:
                fps = 30.0
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame, frame_idx, fps, frame_count, width, height
                frame_idx += 1
        finally:
            cap.release()

def calculate_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0.0:
        return 0.0
        
    return intersection_area / union_area

class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=30):
        """
        A lightweight class-aware IoU-based tracker for matching detections across frames.
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracked_objects = {}
        
    def update(self, detections, frame_idx):
        """
        Updates the tracks with new detections from the current frame.
        """
        active_ids = list(self.tracked_objects.keys())
        matches = []
        
        # Calculate IoU between all current detections and existing tracked objects of the same class
        for det_idx, det in enumerate(detections):
            det_bbox = det["bbox"]
            det_cls = det["class_id"]
            for track_id in active_ids:
                track_data = self.tracked_objects[track_id]
                if track_data["class_id"] == det_cls:
                    track_bbox = track_data["bbox"]
                    iou = calculate_iou(det_bbox, track_bbox)
                    if iou >= self.iou_threshold:
                        matches.append((iou, det_idx, track_id))
                        
        # Sort matches by IoU in descending order (greedy matching)
        matches.sort(key=lambda x: x[0], reverse=True)
        
        matched_det_indices = set()
        matched_track_ids = set()
        
        for iou, det_idx, track_id in matches:
            if det_idx in matched_det_indices or track_id in matched_track_ids:
                continue
                
            matched_det_indices.add(det_idx)
            matched_track_ids.add(track_id)
            
            # Update tracked object details
            det = detections[det_idx]
            track_data = self.tracked_objects[track_id]
            track_data["bbox"] = det["bbox"]
            track_data["confidence"] = det["confidence"]
            track_data["lost_frames"] = 0
            track_data["last_seen_frame"] = frame_idx
            
            centroid_x = (det["bbox"][0] + det["bbox"][2]) / 2.0
            centroid_y = (det["bbox"][1] + det["bbox"][3]) / 2.0
            track_data["centroid_history"].append((centroid_x, centroid_y, frame_idx))
            
            if len(track_data["centroid_history"]) > 100:
                track_data["centroid_history"].pop(0)
                
        # Register new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                centroid_x = (det["bbox"][0] + det["bbox"][2]) / 2.0
                centroid_y = (det["bbox"][1] + det["bbox"][3]) / 2.0
                
                self.tracked_objects[self.next_id] = {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "centroid_history": [(centroid_x, centroid_y, frame_idx)],
                    "lost_frames": 0,
                    "first_seen_frame": frame_idx,
                    "last_seen_frame": frame_idx
                }
                self.next_id += 1
                
        # Handle lost tracks
        dead_tracks = []
        for track_id in active_ids:
            if track_id not in matched_track_ids:
                self.tracked_objects[track_id]["lost_frames"] += 1
                if self.tracked_objects[track_id]["lost_frames"] > self.max_lost_frames:
                    dead_tracks.append(track_id)
                    
        for track_id in dead_tracks:
            del self.tracked_objects[track_id]
            
        # Return tracks present in the current frame
        return {
            tid: data for tid, data in self.tracked_objects.items() 
            if data["last_seen_frame"] == frame_idx
        }

def is_inside_polygon(point, polygon):
    """
    Checks if a point (x, y) is inside a polygon using OpenCV's pointPolygonTest.
    """
    poly_arr = np.array(polygon, dtype=np.float32)
    pt = (float(point[0]), float(point[1]))
    result = cv2.pointPolygonTest(poly_arr, pt, False)
    return result >= 0

def log_shelf_event(frame, frame_idx, shelf_name, event_type, current_stock, evidence_dir, log_list, details=""):
    """
    Logs an inventory shelf event, prints an alert, and saves an evidence frame.
    """
    print(f"[ALERT] {event_type} event on '{shelf_name}' (Current Stock: {current_stock}) at frame {frame_idx}. Details: {details}")
    
    # Format timestamp and filename
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence_filename = f"shelf_{shelf_name}_{event_type}_{timestamp_str}_frame_{frame_idx}.jpg"
    evidence_path = os.path.join(evidence_dir, evidence_filename)
    
    # Save frame
    cv2.imwrite(evidence_path, frame)
    
    # Append structured log
    log_list.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_id": f"{shelf_name}_{event_type}_{frame_idx}",
        "shelf_name": shelf_name,
        "event_type": event_type,
        "item_count": int(current_stock),
        "frame_number": int(frame_idx),
        "evidence_filename": evidence_filename,
        "details": details
    })
    
    return evidence_filename

def display_shelf_dashboard(logs_dir, save_plot=True):
    """
    Loads incident records from the CSV file and displays/saves a formatted analytics dashboard.
    """
    csv_path = os.path.join(logs_dir, "incident_log.csv")
    summary_path = os.path.join(logs_dir, "summary_stats.json")
    
    if not os.path.exists(csv_path):
        print("[ERROR] No incident log CSV found. Process a video first.")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Error loading incident log CSV: {e}")
        return
        
    print("\n" + "="*60)
    print("             SHELF INVENTORY MONITORING SYSTEM ANALYTICS")
    print("="*60)
    
    total_events = len(df)
    low_stock_events = len(df[df['event_type'] == 'LOW_STOCK'])
    empty_events = len(df[df['event_type'] == 'EMPTY'])
    depletion_events = len(df[df['event_type'] == 'STOCK_DEPLETION'])
    restock_events = len(df[df['event_type'] == 'STOCK_RESTOCKING'])
    interaction_events = len(df[df['event_type'] == 'CUSTOMER_INTERACTION'])
    
    print(f"Total Logged Events:           {total_events}")
    print(f"Low Stock Alerts Triggered:    {low_stock_events}")
    print(f"Empty Shelf Alerts Triggered:  {empty_events}")
    print(f"Stock Depletions (Purchases):  {depletion_events}")
    print(f"Stock Restockings:             {restock_events}")
    print(f"Customer Interactions:         {interaction_events}")
    print("="*60 + "\n")
    
    summary_stats = {
        "total_events": total_events,
        "low_stock_alerts": low_stock_events,
        "empty_shelf_alerts": empty_events,
        "stock_depletions": depletion_events,
        "stock_restockings": restock_events,
        "customer_interactions": interaction_events
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=4)
        
    if total_events == 0:
        print("Zero events logged. Skipping dashboard plot rendering.")
        return
        
    plt.ioff()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color Palette: Premium Theme
    colors = ['#4A90E2', '#F5A623', '#D0021B', '#7ED321', '#BD10E0']
    
    # Plot 1: Event Type Distribution
    event_counts = df['event_type'].value_counts()
    axes[0].bar(event_counts.index, event_counts.values, color=colors[:len(event_counts)], edgecolor='black', zorder=2)
    axes[0].set_title("Event Distribution by Type", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Event Type", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    
    # Plot 2: Inventory Level over Time per Shelf
    # Group by shelf_name and plot item_count over frame_number
    # We sort by frame_number to ensure correct timeline
    sorted_df = df.sort_values(by='frame_number')
    shelves = sorted_df['shelf_name'].unique()
    
    for i, shelf in enumerate(shelves):
        shelf_df = sorted_df[sorted_df['shelf_name'] == shelf]
        # Filter for inventory change events only (Low Stock, Empty, Depletion, Restocking)
        inv_df = shelf_df[shelf_df['event_type'].isin(['LOW_STOCK', 'EMPTY', 'STOCK_DEPLETION', 'STOCK_RESTOCKING', 'INITIAL_STOCK'])]
        if not inv_df.empty:
            axes[1].step(inv_df['frame_number'], inv_df['item_count'], where='post', label=f"{shelf} Stock", linewidth=2.5, color=colors[i % len(colors)])
            axes[1].scatter(inv_df['frame_number'], inv_df['item_count'], color=colors[i % len(colors)], s=50, zorder=3)
            
    axes[1].set_title("Shelf Stock Levels Timeline", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Frame Number", fontsize=10)
    axes[1].set_ylabel("Product Count", fontsize=10)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_ylim(bottom=-0.5)
    
    plt.tight_layout()
    if save_plot:
        plot_path = os.path.join(logs_dir, "analytics_dashboard.png")
        plt.savefig(plot_path, dpi=150)
        print(f"[INFO] Saved shelf analytics dashboard plot to: {plot_path}")
    plt.close(fig)
