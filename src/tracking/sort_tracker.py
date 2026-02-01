"""
SORT Tracker - Simple Online Realtime Tracking
==============================================
Lightweight tracking algorithm for maintaining slot identity across frames.
Based on Kalman filtering and Hungarian algorithm for data association.

Reference: Bewley et al., "Simple Online and Realtime Tracking", 2016
Author: Research Team
Date: February 2026
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import logging

logger = logging.getLogger(__name__)


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes in image space.
    State: [x, y, s, r, dx, dy, ds], where (x,y) is center, s is scale, r is aspect ratio
    """
    
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize tracker with initial bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] format
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """
        Update tracker with observed bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] format
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        
    def predict(self):
        """
        Advance state and return predicted bounding box estimate.
        
        Returns:
            Predicted bbox in [x1, y1, x2, y2] format
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        """
        Return current bounding box estimate.
        
        Returns:
            Current bbox in [x1, y1, x2, y2] format
        """
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox):
        """
        Convert [x1, y1, x2, y2] to [x, y, s, r] format.
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            [x, y, s, r] where x,y is center, s is scale, r is aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h  # scale is area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        """
        Convert [x, y, s, r] to [x1, y1, x2, y2] format.
        
        Args:
            x: [x, y, s, r, ...]
            score: Optional confidence score
            
        Returns:
            [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([
                x[0] - w / 2.0,
                x[1] - h / 2.0,
                x[0] + w / 2.0,
                x[1] + h / 2.0
            ]).reshape((1, 4))
        else:
            return np.array([
                x[0] - w / 2.0,
                x[1] - h / 2.0,
                x[0] + w / 2.0,
                x[1] + h / 2.0,
                score
            ]).reshape((1, 5))


def iou_batch(bb_test, bb_gt):
    """
    Compute IoU between two sets of bounding boxes.
    
    Args:
        bb_test: (N, 4) array of [x1, y1, x2, y2]
        bb_gt: (M, 4) array of [x1, y1, x2, y2]
        
    Returns:
        (N, M) array of IoU values
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


class SORTTracker:
    """
    SORT: Simple Online Realtime Tracker
    Manages multiple object trackers using Kalman filters and Hungarian algorithm.
    """
    
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
        logger.info(f"SORT tracker initialized: max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")
        
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: (N, 5) array of [x1, y1, x2, y2, score] or (N, 6) with class_id
            
        Returns:
            (M, 5) array of [x1, y1, x2, y2, track_id] for confirmed tracks
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        if len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                detections, trks, self.iou_threshold
            )
            
            # Update matched trackers
            for m in matched:
                self.trackers[m[1]].update(detections[m[0], :4])
            
            # Create new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(detections[i, :4])
                self.trackers.append(trk)
        
        # Return confirmed tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            
            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assign detections to tracked objects using Hungarian algorithm.
        
        Args:
            detections: (N, 5) array
            trackers: (M, 5) array
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_trackers)
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Use Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack([row_ind, col_ind], axis=1)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matched with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def reset(self):
        """Reset tracker state."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
        logger.info("Tracker reset")


if __name__ == "__main__":
    # Test SORT tracker
    logging.basicConfig(level=logging.INFO)
    
    tracker = SORTTracker(max_age=3, min_hits=3, iou_threshold=0.3)
    
    # Simulate detections over 5 frames
    frame1_dets = np.array([[10, 10, 50, 50, 0.9], [100, 100, 150, 150, 0.85]])
    frame2_dets = np.array([[12, 12, 52, 52, 0.88], [102, 102, 152, 152, 0.87]])
    frame3_dets = np.array([[14, 14, 54, 54, 0.91], [104, 104, 154, 154, 0.86]])
    
    tracks1 = tracker.update(frame1_dets)
    print(f"Frame 1 tracks: {len(tracks1)}")
    
    tracks2 = tracker.update(frame2_dets)
    print(f"Frame 2 tracks: {len(tracks2)}")
    
    tracks3 = tracker.update(frame3_dets)
    print(f"Frame 3 tracks: {len(tracks3)}")
    
    print("✓ SORT tracker test completed")
