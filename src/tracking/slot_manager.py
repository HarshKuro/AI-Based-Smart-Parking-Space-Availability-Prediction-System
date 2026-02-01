"""
Slot Manager - Parking Space State Management
==============================================
Maintains temporal state of parking slots with spatial anchoring.
Aggregates occupancy over time for availability prediction.

Author: Research Team
Date: February 2026
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ParkingSlot:
    """Represents a single parking slot with temporal state."""
    
    def __init__(self, slot_id: int, spatial_anchor: np.ndarray, class_names: List[str]):
        """
        Initialize parking slot.
        
        Args:
            slot_id: Unique identifier
            spatial_anchor: [x1, y1, x2, y2] bounding box
            class_names: List of class names ['free', 'occupied', 'partial']
        """
        self.slot_id = slot_id
        self.spatial_anchor = spatial_anchor
        self.class_names = class_names
        
        # State tracking
        self.current_state = None  # 0=free, 1=occupied, 2=partial
        self.confidence = 0.0
        self.last_updated = None
        
        # Temporal history
        self.state_history = deque(maxlen=100)  # Last 100 observations
        self.confidence_history = deque(maxlen=100)
        
        # Statistics
        self.total_frames = 0
        self.occupied_frames = 0
        self.free_frames = 0
        self.partial_frames = 0
        
    def update(self, state: int, confidence: float):
        """
        Update slot state.
        
        Args:
            state: Class ID (0=free, 1=occupied, 2=partial)
            confidence: Detection confidence [0, 1]
        """
        self.current_state = state
        self.confidence = confidence
        self.last_updated = datetime.now()
        
        self.state_history.append(state)
        self.confidence_history.append(confidence)
        
        self.total_frames += 1
        if state == 0:
            self.free_frames += 1
        elif state == 1:
            self.occupied_frames += 1
        elif state == 2:
            self.partial_frames += 1
    
    def get_occupancy_rate(self) -> float:
        """
        Calculate occupancy rate.
        
        Returns:
            Occupancy rate [0, 1]
        """
        if self.total_frames == 0:
            return 0.0
        return self.occupied_frames / self.total_frames
    
    def get_smoothed_state(self, window: int = 5) -> int:
        """
        Get smoothed state using majority voting over recent frames.
        
        Args:
            window: Number of recent frames to consider
            
        Returns:
            Smoothed state (0, 1, or 2)
        """
        if len(self.state_history) == 0:
            return 0  # Default to free
        
        recent = list(self.state_history)[-window:]
        if not recent:
            return 0
        
        # Majority voting
        counts = {0: 0, 1: 0, 2: 0}
        for state in recent:
            counts[state] = counts.get(state, 0) + 1
        
        return max(counts, key=counts.get)
    
    def is_available(self) -> bool:
        """Check if slot is available (free or partially free)."""
        smoothed = self.get_smoothed_state()
        return smoothed in [0, 2]  # Free or partial


class SlotManager:
    """Manages multiple parking slots with temporal reasoning."""
    
    def __init__(self, class_names: List[str], memory_frames: int = 30, iou_threshold: float = 0.5):
        """
        Initialize slot manager.
        
        Args:
            class_names: List of class names
            memory_frames: Frames to remember slot state
            iou_threshold: IoU threshold for spatial matching
        """
        self.class_names = class_names
        self.memory_frames = memory_frames
        self.iou_threshold = iou_threshold
        
        self.slots: Dict[int, ParkingSlot] = {}
        self.next_slot_id = 0
        self.frame_count = 0
        
        # Availability history for prediction
        self.availability_history = deque(maxlen=1000)  # Last 1000 frames
        self.timestamp_history = deque(maxlen=1000)
        
        logger.info(f"Slot manager initialized with {len(class_names)} classes")
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value [0, 1]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def update(self, detections: np.ndarray, classes: np.ndarray, confidences: np.ndarray):
        """
        Update slot states with new detections.
        
        Args:
            detections: (N, 4) array of [x1, y1, x2, y2]
            classes: (N,) array of class IDs
            confidences: (N,) array of confidence scores
        """
        self.frame_count += 1
        
        # Match detections to existing slots
        matched_slots = set()
        
        for det_idx, (bbox, cls, conf) in enumerate(zip(detections, classes, confidences)):
            # Find matching slot
            best_iou = 0.0
            best_slot_id = None
            
            for slot_id, slot in self.slots.items():
                iou = self._compute_iou(bbox, slot.spatial_anchor)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_slot_id = slot_id
            
            if best_slot_id is not None:
                # Update existing slot
                self.slots[best_slot_id].update(int(cls), float(conf))
                matched_slots.add(best_slot_id)
            else:
                # Create new slot
                new_slot = ParkingSlot(self.next_slot_id, bbox, self.class_names)
                new_slot.update(int(cls), float(conf))
                self.slots[self.next_slot_id] = new_slot
                matched_slots.add(self.next_slot_id)
                self.next_slot_id += 1
        
        # Update availability history
        available = sum(1 for slot in self.slots.values() if slot.is_available())
        total = len(self.slots)
        
        self.availability_history.append({
            'available': available,
            'total': total,
            'occupancy_rate': 1.0 - (available / total if total > 0 else 0.0)
        })
        self.timestamp_history.append(datetime.now())
    
    def get_availability(self) -> Dict:
        """
        Get current parking lot availability.
        
        Returns:
            Dictionary with availability metrics
        """
        total_slots = len(self.slots)
        free_slots = sum(1 for slot in self.slots.values() if slot.get_smoothed_state() == 0)
        occupied_slots = sum(1 for slot in self.slots.values() if slot.get_smoothed_state() == 1)
        partial_slots = sum(1 for slot in self.slots.values() if slot.get_smoothed_state() == 2)
        available_slots = free_slots + partial_slots
        
        return {
            'total': total_slots,
            'free': free_slots,
            'occupied': occupied_slots,
            'partially_free': partial_slots,
            'available': available_slots,
            'occupancy_rate': occupied_slots / total_slots if total_slots > 0 else 0.0,
            'availability_rate': available_slots / total_slots if total_slots > 0 else 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_slot_details(self) -> List[Dict]:
        """
        Get detailed information for all slots.
        
        Returns:
            List of slot dictionaries
        """
        slots_info = []
        
        for slot_id, slot in self.slots.items():
            smoothed_state = slot.get_smoothed_state()
            
            slots_info.append({
                'slot_id': slot_id,
                'state': self.class_names[smoothed_state],
                'state_id': smoothed_state,
                'confidence': slot.confidence,
                'bbox': slot.spatial_anchor.tolist(),
                'is_available': slot.is_available(),
                'occupancy_rate': slot.get_occupancy_rate(),
                'total_observations': slot.total_frames,
                'last_updated': slot.last_updated.isoformat() if slot.last_updated else None
            })
        
        return slots_info
    
    def predict_availability(self, horizon_minutes: int = 15) -> Dict:
        """
        Predict future availability using exponential moving average.
        
        Args:
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Prediction dictionary
        """
        if len(self.availability_history) < 10:
            # Not enough data
            current = self.get_availability()
            return {
                'horizon_minutes': horizon_minutes,
                'predicted_available': current['available'],
                'predicted_occupancy_rate': current['occupancy_rate'],
                'confidence': 'low',
                'note': 'Insufficient historical data'
            }
        
        # Simple exponential moving average
        recent_occupancy = [h['occupancy_rate'] for h in list(self.availability_history)[-50:]]
        weights = np.exp(np.linspace(-1, 0, len(recent_occupancy)))
        weights /= weights.sum()
        
        predicted_occupancy = np.average(recent_occupancy, weights=weights)
        total_slots = len(self.slots)
        predicted_occupied = int(predicted_occupancy * total_slots)
        predicted_available = total_slots - predicted_occupied
        
        return {
            'horizon_minutes': horizon_minutes,
            'predicted_available': predicted_available,
            'predicted_occupied': predicted_occupied,
            'predicted_occupancy_rate': predicted_occupancy,
            'predicted_availability_rate': 1.0 - predicted_occupancy,
            'confidence': 'medium' if len(self.availability_history) > 50 else 'low',
            'based_on_frames': len(self.availability_history)
        }
    
    def reset(self):
        """Reset slot manager state."""
        self.slots = {}
        self.next_slot_id = 0
        self.frame_count = 0
        self.availability_history.clear()
        self.timestamp_history.clear()
        logger.info("Slot manager reset")


if __name__ == "__main__":
    # Test slot manager
    logging.basicConfig(level=logging.INFO)
    
    class_names = ['free_parking_space', 'not_free_parking_space', 'partially_free_parking_space']
    manager = SlotManager(class_names)
    
    # Simulate detections
    for frame in range(20):
        detections = np.array([
            [10, 10, 50, 50],
            [60, 10, 100, 50],
            [110, 10, 150, 50]
        ])
        classes = np.array([0, 1, 0])  # free, occupied, free
        confidences = np.array([0.9, 0.85, 0.88])
        
        manager.update(detections, classes, confidences)
    
    # Get availability
    availability = manager.get_availability()
    print(f"Availability: {availability}")
    
    # Get prediction
    prediction = manager.predict_availability(15)
    print(f"Prediction: {prediction}")
    
    print("✓ Slot manager test completed")
