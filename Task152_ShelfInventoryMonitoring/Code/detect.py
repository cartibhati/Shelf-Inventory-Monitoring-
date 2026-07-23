import os
import sys
import cv2
import argparse
import time
import json
import torch
import pandas as pd
import numpy as np

# PyTorch 2.6 compatibility patch for loading YOLO models safely
try:
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass

from ultralytics import YOLO

# Add the directory containing utils to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    video_frame_generator,
    SimpleIoUTracker,
    is_inside_polygon,
    log_shelf_event,
    display_shelf_dashboard
)

def parse_args():
    parser = argparse.ArgumentParser(description="AI-Based Shelf Inventory Monitoring System")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file or image sequence directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save the annotated output video. If empty, auto-saves to Outputs/."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Models", "yolov8n.pt")),
        help="Path to the YOLOv8 model weight file."
    )
    parser.add_argument(
        "--shelf_rois",
        type=str,
        default="",
        help="Custom shelf ROIs in the format 'ShelfName:x1,y1,x2,y2,x3,y3,x4,y4;ShelfName2:...'"
    )
    parser.add_argument(
        "--product_classes",
        type=str,
        default="bottle,cup,apple,orange,banana,book,vase,teddy bear,sports ball,box,handbag,backpack",
        help="Comma-separated list of COCO object classes representing products on shelves."
    )
    parser.add_argument(
        "--person_classes",
        type=str,
        default="person",
        help="Comma-separated list of COCO classes representing people (customers/staff)."
    )
    parser.add_argument(
        "--low_stock_limit",
        type=int,
        default=2,
        help="Default low stock threshold count."
    )
    return parser.parse_args()

def get_default_rois(video_filename):
    """
    Returns default shelf ROIs based on the video name to allow zero-config execution.
    """
    video_lower = video_filename.lower()
    if "shelf_0" in video_lower:
        # Video 0: AI Surveillance for Retail - Showroom and Shelf Monitoring
        # Define two main shelf regions
        return {
            "Shelf_Left": [(0.05, 0.20), (0.42, 0.20), (0.42, 0.90), (0.05, 0.90)],
            "Shelf_Right": [(0.52, 0.20), (0.95, 0.20), (0.95, 0.90), (0.52, 0.90)]
        }
    elif "shelf_1" in video_lower:
        # Video 1: Auchan Optimising On-Shelf Availability
        # A single central shelf area monitoring items
        return {
            "Main_Display_Shelf": [(0.10, 0.25), (0.90, 0.25), (0.90, 0.85), (0.10, 0.85)]
        }
    elif "shelf_2" in video_lower:
        # Video 2: WiFi camera for smart retail shelf monitoring
        # Split into upper shelf and lower shelf
        return {
            "Upper_Shelf": [(0.05, 0.12), (0.95, 0.12), (0.95, 0.48), (0.05, 0.48)],
            "Lower_Shelf": [(0.05, 0.52), (0.95, 0.52), (0.95, 0.92), (0.05, 0.92)]
        }
    else:
        # Fallback ROI: Center of the frame
        return {
            "Center_Shelf": [(0.15, 0.20), (0.85, 0.20), (0.85, 0.85), (0.15, 0.85)]
        }

def parse_custom_rois(roi_str):
    """
    Parses shelf ROIs from the CLI arguments string.
    Format: 'ShelfName:x1,y1,x2,y2,x3,y3,x4,y4;ShelfName2:...'
    """
    rois = {}
    if not roi_str:
        return rois
    try:
        shelf_configs = roi_str.split(';')
        for config in shelf_configs:
            if not config.strip():
                continue
            name, coord_str = config.split(':')
            coords = [float(c) for c in coord_str.split(',') if c.strip()]
            if len(coords) % 2 != 0 or len(coords) < 6:
                raise ValueError("Each ROI must have at least 3 vertices (6 coordinates) and be even-numbered.")
            vertices = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            rois[name.strip()] = vertices
    except Exception as e:
        print(f"[ERROR] Failed to parse custom ROIs: {e}")
        print("Falling back to default or file-based ROIs.")
    return rois

def main():
    args = parse_args()
    
    # Setup base output directories relative to script
    code_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.abspath(os.path.join(code_dir, "..", "Outputs"))
    evidence_dir = os.path.join(outputs_dir, "evidence_frames")
    inspect_dir = os.path.join(outputs_dir, "inspect_frames")
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)
    os.makedirs(inspect_dir, exist_ok=True)
    
    # Resolve input path
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"[ERROR] Input path does not exist: {video_path}")
        sys.exit(1)
        
    video_filename = os.path.basename(video_path.rstrip('/\\'))
    video_name_only, _ = os.path.splitext(video_filename)
    
    # Resolve output video path
    if args.output:
        output_video_path = args.output
    else:
        output_video_path = os.path.join(outputs_dir, f"{video_name_only}_annotated.mp4")
        
    # Parse class lists
    product_class_list = [c.strip().lower() for c in args.product_classes.split(',') if c.strip()]
    person_class_list = [c.strip().lower() for c in args.person_classes.split(',') if c.strip()]
    all_allowed_classes = set(product_class_list + person_class_list)
    
    # Parse/Retrieve ROIs
    shelf_rois = parse_custom_rois(args.shelf_rois)
    if not shelf_rois:
        shelf_rois = get_default_rois(video_filename)
        
    print(f"\n[INFO] Initializing Shelf Inventory Monitoring Pipeline")
    print(f" - Input Path:           {video_path}")
    print(f" - Output Video Path:      {output_video_path}")
    print(f" - YOLOv8 Model:           {args.model}")
    print(f" - Low Stock Threshold:    {args.low_stock_limit} items")
    print(f" - Monitored Shelves:      {list(shelf_rois.keys())}")
    
    if not os.path.exists(args.model):
        print(f"[ERROR] Model weights file not found at: {args.model}")
        sys.exit(1)
        
    print("\nLoading YOLOv8 model...")
    model = YOLO(args.model)
    print("[OK] Model loaded successfully.\n")
    
    start_time = time.time()
    processed_frames = 0
    incident_logs = []
    
    # Trackers for people and products
    tracker = SimpleIoUTracker(iou_threshold=0.35, max_lost_frames=30)
    
    # Initialize Shelf Tracking States
    # We want to record historical counts and interaction status
    shelf_states = {}
    for name, vertices in shelf_rois.items():
        shelf_states[name] = {
            "vertices": vertices,
            "prev_stock_count": -1,
            "status": "INITIALIZING",
            "under_interaction": False,
            "interaction_start_count": 0,
            "interaction_cooldown": 0,
            "pixel_polygon": None # Set once resolution is known
        }
        
    try:
        frame_generator = video_frame_generator(video_path)
        out_writer = None
        
        for frame, frame_idx, fps, frame_count, width, height in frame_generator:
            if out_writer is None:
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                print(f"Video specs: {width}x{height} pixels | {fps:.2f} FPS | {frame_count} total frames")
                print("Processing frames...")
                
                # Scale ROIs to pixel resolution
                for name, state in shelf_states.items():
                    state["pixel_polygon"] = [(int(pt[0] * width), int(pt[1] * height)) for pt in state["vertices"]]
                    
            # Run YOLOv8 detection
            results = model.predict(frame, verbose=False)
            result = results[0]
            
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                names = result.names
                
                for idx in range(len(boxes)):
                    class_name = names[int(classes[idx])].lower()
                    if class_name in all_allowed_classes and confs[idx] >= 0.25:
                        detections.append({
                            "bbox": list(boxes[idx]),
                            "confidence": float(confs[idx]),
                            "class_id": int(classes[idx]),
                            "class_name": class_name
                        })
                        
            # Update IoU tracker
            current_tracks = tracker.update(detections, frame_idx)
            
            annotated_frame = frame.copy()
            
            # Count products and identify people
            products_in_frame = []
            people_in_frame = []
            
            for track_id, track in current_tracks.items():
                cls_name = track["class_name"]
                bbox = track["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                # Use bottom-center for person containment
                person_by = bbox[3]
                
                if cls_name in product_class_list:
                    products_in_frame.append((cx, cy, track_id, track))
                elif cls_name in person_class_list:
                    people_in_frame.append((cx, person_by, track_id, track))
                    
            # Process each Shelf ROI
            active_alerts = []
            for name, state in shelf_states.items():
                pixel_poly = state["pixel_polygon"]
                pixel_poly_np = np.array(pixel_poly, dtype=np.int32)
                
                # 1. Count products inside this shelf ROI
                current_stock = 0
                items_inside = []
                for cx, cy, tid, track in products_in_frame:
                    if is_inside_polygon((cx, cy), pixel_poly):
                        current_stock += 1
                        items_inside.append((tid, track))
                        
                # 2. Check if any person overlaps with this shelf ROI
                person_overlapping = False
                for px, py, pid, track in people_in_frame:
                    # Check if bottom-center of the person or any corner is inside the ROI
                    # Alternatively, check box intersection or proximity
                    p_bbox = track["bbox"]
                    corners = [
                        (p_bbox[0], p_bbox[1]), (p_bbox[2], p_bbox[1]),
                        (p_bbox[0], p_bbox[3]), (p_bbox[2], p_bbox[3]),
                        (px, py)
                    ]
                    for corner in corners:
                        if is_inside_polygon(corner, pixel_poly):
                            person_overlapping = True
                            break
                    if person_overlapping:
                        break
                        
                # 3. Update interaction status and cooldown
                if person_overlapping:
                    state["interaction_cooldown"] = int(fps * 1.5) # 1.5 seconds cooldown
                    if not state["under_interaction"]:
                        # Customer started interaction
                        state["under_interaction"] = True
                        state["interaction_start_count"] = current_stock
                        log_shelf_event(
                            frame=frame,
                            frame_idx=frame_idx,
                            shelf_name=name,
                            event_type="CUSTOMER_INTERACTION",
                            current_stock=current_stock,
                            evidence_dir=evidence_dir,
                            log_list=incident_logs,
                            details="Customer approached shelf and started interacting."
                        )
                else:
                    if state["interaction_cooldown"] > 0:
                        state["interaction_cooldown"] -= 1
                    elif state["under_interaction"]:
                        # Customer finished interaction: evaluate inventory change
                        state["under_interaction"] = False
                        diff = current_stock - state["interaction_start_count"]
                        if diff < 0:
                            # Stock depletion
                            log_shelf_event(
                                frame=frame,
                                frame_idx=frame_idx,
                                shelf_name=name,
                                event_type="STOCK_DEPLETION",
                                current_stock=current_stock,
                                evidence_dir=evidence_dir,
                                log_list=incident_logs,
                                details=f"Stock decreased from {state['interaction_start_count']} to {current_stock} items."
                            )
                        elif diff > 0:
                            # Stock replenished
                            log_shelf_event(
                                frame=frame,
                                frame_idx=frame_idx,
                                shelf_name=name,
                                event_type="STOCK_RESTOCKING",
                                current_stock=current_stock,
                                evidence_dir=evidence_dir,
                                log_list=incident_logs,
                                details=f"Stock replenished from {state['interaction_start_count']} to {current_stock} items."
                            )
                        else:
                            # Checked but no purchase/refill
                            log_shelf_event(
                                frame=frame,
                                frame_idx=frame_idx,
                                shelf_name=name,
                                event_type="CUSTOMER_INTERACTION_END",
                                current_stock=current_stock,
                                evidence_dir=evidence_dir,
                                log_list=incident_logs,
                                details="Customer left shelf. No stock levels changed."
                            )
                            
                # 4. Check stock transitions (Low Stock, Empty, Normal)
                # Only log on state transitions
                prev_status = state["status"]
                
                if current_stock == 0:
                    new_status = "EMPTY"
                elif current_stock <= args.low_stock_limit:
                    new_status = "LOW_STOCK"
                else:
                    new_status = "NORMAL"
                    
                state["status"] = new_status
                
                # Triggers on status transitions (avoiding initialization logging unless it starts low/empty)
                if prev_status != new_status:
                    if prev_status == "INITIALIZING":
                        # Log initial stock count
                        log_shelf_event(
                            frame=frame,
                            frame_idx=frame_idx,
                            shelf_name=name,
                            event_type="INITIAL_STOCK",
                            current_stock=current_stock,
                            evidence_dir=evidence_dir,
                            log_list=incident_logs,
                            details=f"Initial stock level: {current_stock} items."
                        )
                        # If initially low or empty, trigger alert as well
                        if new_status in ["EMPTY", "LOW_STOCK"]:
                            log_shelf_event(
                                frame=frame,
                                frame_idx=frame_idx,
                                shelf_name=name,
                                event_type=new_status,
                                current_stock=current_stock,
                                evidence_dir=evidence_dir,
                                log_list=incident_logs,
                                details=f"Initial stock is critical: {new_status}."
                            )
                    else:
                        log_shelf_event(
                            frame=frame,
                            frame_idx=frame_idx,
                            shelf_name=name,
                            event_type=new_status,
                            current_stock=current_stock,
                            evidence_dir=evidence_dir,
                            log_list=incident_logs,
                            details=f"Stock transitioned from {prev_status} to {new_status}. Count: {current_stock} items."
                        )
                        
                # Compile alerts for HUD banner
                if new_status == "EMPTY":
                    active_alerts.append(f"SHELF '{name.upper()}' IS EMPTY!")
                elif new_status == "LOW_STOCK":
                    active_alerts.append(f"SHELF '{name.upper()}' IS LOW STOCK!")
                    
                # 5. Draw Shelf HUD elements on the frame
                # Bounding box color coding for shelf:
                # - Blue: Active customer interaction
                # - Red: Shelf Empty
                # - Yellow: Low Stock
                # - Green: Normal Full Stock
                if state["under_interaction"]:
                    shelf_color = (255, 144, 30) # Blue
                    status_lbl = "CUSTOMER INTERACTION"
                elif new_status == "EMPTY":
                    shelf_color = (0, 0, 255) # Red
                    status_lbl = "EMPTY"
                elif new_status == "LOW_STOCK":
                    shelf_color = (0, 255, 255) # Yellow
                    status_lbl = "LOW STOCK"
                else:
                    shelf_color = (0, 255, 0) # Green
                    status_lbl = "NORMAL"
                    
                # Draw semi-transparent fill for shelf
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [pixel_poly_np], shelf_color)
                cv2.addWeighted(overlay, 0.12, annotated_frame, 0.88, 0, annotated_frame)
                
                # Draw outline
                cv2.polylines(annotated_frame, [pixel_poly_np], True, shelf_color, 2, lineType=cv2.LINE_AA)
                
                # Label Shelf
                lbl_text = f"{name} | {status_lbl} (Stock: {current_stock})"
                cv2.putText(annotated_frame, lbl_text, (pixel_poly[0][0], pixel_poly[0][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, shelf_color, 2, lineType=cv2.LINE_AA)
                
                # Draw item circles and labels inside the shelf
                for tid, track in items_inside:
                    tx1, ty1, tx2, ty2 = map(int, track["bbox"])
                    # Draw a nice cyan bounding box around products
                    cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (255, 255, 0), 1)
                    # Dot at center
                    cv2.circle(annotated_frame, (int((tx1+tx2)/2), int((ty1+ty2)/2)), 3, (255, 255, 0), -1)
                    
            # Draw people bounding boxes & path trajectories
            for px, py, pid, track in people_in_frame:
                px1, py1, px2, py2 = map(int, track["bbox"])
                # Purple box for customers/staff
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (180, 0, 180), 2)
                lbl = f"Customer ID {pid}"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                cv2.rectangle(annotated_frame, (px1, py1 - 18), (px1 + lw, py1), (180, 0, 180), -1)
                cv2.putText(annotated_frame, lbl, (px1, py1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                
                # Draw customer trail line
                history = track["centroid_history"]
                for h_idx in range(1, len(history)):
                    pt1 = (int(history[h_idx - 1][0]), int(history[h_idx - 1][1]))
                    pt2 = (int(history[h_idx][0]), int(history[h_idx][1]))
                    cv2.line(annotated_frame, pt1, pt2, (180, 0, 180), 2, lineType=cv2.LINE_AA)
                    
            # Draw header warning banner if critical alerts exist
            if active_alerts:
                cv2.rectangle(annotated_frame, (0, 0), (width, 40), (0, 0, 180), -1)
                warning_msg = f"WARNING: " + " | ".join(active_alerts)
                cv2.putText(annotated_frame, warning_msg, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                
            # Write frame to output video
            out_writer.write(annotated_frame)
            
            # Save periodic inspect frames for reporting
            if frame_idx in [0, 50, 100, 150, 200, 250, 300, 400, 500]:
                inspect_img_path = os.path.join(inspect_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(inspect_img_path, annotated_frame)
                
            processed_frames += 1
            if processed_frames % 50 == 0 or processed_frames == frame_count:
                progress = (processed_frames / frame_count) * 100
                print(f" [INFO] Processed {processed_frames}/{frame_count} frames ({progress:.1f}%)")
                
        # Close output stream
        if out_writer is not None:
            out_writer.release()
            
        if processed_frames == 0:
            print("[ERROR] Processing finished, but no frames were loaded. Check video file.")
            return
            
        # Export CSV log
        log_cols = ["timestamp", "event_id", "shelf_name", "event_type", "item_count", "frame_number", "evidence_filename", "details"]
        log_df = pd.DataFrame(incident_logs, columns=log_cols)
        csv_path = os.path.join(outputs_dir, "incident_log.csv")
        
        # Merge with existing logs to preserve history across video runs
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                existing_df = pd.read_csv(csv_path)
                log_df = pd.concat([existing_df, log_df], ignore_index=True)
                log_df = log_df.drop_duplicates(subset=["timestamp", "event_id"])
            except Exception:
                pass
                
        log_df.to_csv(csv_path, index=False)
        print(f"[INFO] Incident log CSV written/updated: {csv_path}")
        
        # Timing Stats
        elapsed = time.time() - start_time
        avg_fps = processed_frames / elapsed
        
        print("\n" + "="*60)
        print("[SUCCESS] SHELF INVENTORY MONITORING PIPELINE COMPLETED")
        print("="*60)
        print(f"Total Processing Time: {elapsed:.2f} seconds")
        print(f"Average FPS Achieved:  {avg_fps:.2f} FPS")
        print(f"Annotated Video Saved: {output_video_path}")
        print("="*60 + "\n")
        
        # Save analytics dashboard
        display_shelf_dashboard(outputs_dir, save_plot=True)
        
    except Exception as e:
        print(f"[ERROR] Critical pipeline failure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
