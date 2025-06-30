#!/usr/bin/env python3
"""
wrapper.py - Complete Vision Pipeline with Optimized OpenCV Detection for 180×128
Raw Frames → OpenCV → CNN ↘
                           Vision Transformer → Predictions
Health/Score Data → Momentum Tracker ↗
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from collections import deque
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Optional
import math

# Constants
MAX_HEALTH = 176
CRITICAL_HEALTH_THRESHOLD = MAX_HEALTH * 0.3


class EnhancedOpenCVDetector:
    """Enhanced OpenCV detector optimized for 180×128 resolution"""

    def __init__(self):
        # OPTIMIZED: Color ranges for fire knife (big, orange/red projectile)
        self.fire_knife_lower = np.array([5, 120, 120])  # Orange/red in HSV
        self.fire_knife_upper = np.array([25, 255, 255])

        # OPTIMIZED: Color ranges for small bombs (small, bright objects)
        self.bomb_lower = np.array([0, 0, 180])  # Very bright objects
        self.bomb_upper = np.array([180, 80, 255])

        # OPTIMIZED: Size constraints for 180×128 resolution
        self.fire_knife_min_area = 150  # ~12×12 pixels - big projectiles
        self.fire_knife_max_area = 4000  # ~63×63 pixels - avoid player detection
        self.bomb_min_area = 25  # ~5×5 pixels - small objects
        self.bomb_max_area = 350  # ~19×19 pixels - small bombs

        # Floor detection for 128px height
        self.floor_threshold = 128 * 0.6  # Bottom 40% of screen (y > 77)

        # Motion tracking
        self.prev_frame_gray = None

        print(f"🔍 OpenCV detector optimized for 180×128:")
        print(
            f"   Fire knife: {self.fire_knife_min_area}-{self.fire_knife_max_area}px²"
        )
        print(f"   Bombs: {self.bomb_min_area}-{self.bomb_max_area}px²")
        print(f"   Floor zone: y > {self.floor_threshold}")

    def detect_threats(self, frame: np.ndarray) -> Dict:
        """
        Main detection function optimized for 180×128 frames
        Input: frame [128, 180, 3] - RGB frame (H×W×C)
        Output: threat detection dictionary
        """
        if frame is None or frame.size == 0:
            return self._empty_detection()

        try:
            # Convert RGB to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect fire knife (big projectile)
            fire_knives = self._detect_fire_knife(hsv, frame)

            # Detect bombs (small floor objects)
            bombs = self._detect_bombs(hsv, frame, gray)

            # Detect motion for validation
            motion_info = self._detect_motion(gray)

            # Combine results
            result = {
                "fire_knives": fire_knives,
                "bombs": bombs,
                "motion_detected": motion_info["has_motion"],
                "total_threats": len(fire_knives) + len(bombs),
                "threat_level": self._calculate_threat_level(fire_knives, bombs),
            }

            # Debug logging for significant detections
            if result["total_threats"] > 0:
                print(
                    f"   🎯 Detected: {len(fire_knives)} fire knives, {len(bombs)} bombs"
                )

            return result

        except Exception as e:
            print(f"   ⚠️ OpenCV detection error: {e}")
            return self._empty_detection()

    def _detect_fire_knife(self, hsv: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """Detect big fire knife projectiles"""
        try:
            # Create mask for fire colors
            fire_mask = cv2.inRange(hsv, self.fire_knife_lower, self.fire_knife_upper)

            # Morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            fire_knives = []
            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by size - fire knife should be big
                if self.fire_knife_min_area < area < self.fire_knife_max_area:
                    # Get properties
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w // 2, y + h // 2

                    # Calculate aspect ratio (fire knife might be elongated)
                    aspect_ratio = w / h if h > 0 else 1.0

                    # Get average color in the region for validation
                    roi_hsv = hsv[y : y + h, x : x + w]
                    avg_hue = np.mean(roi_hsv[:, :, 0]) if roi_hsv.size > 0 else 0
                    avg_sat = np.mean(roi_hsv[:, :, 1]) if roi_hsv.size > 0 else 0

                    fire_knife = {
                        "type": "fire_knife",
                        "position": (center_x, center_y),
                        "bbox": (x, y, w, h),
                        "area": area,
                        "aspect_ratio": aspect_ratio,
                        "avg_hue": avg_hue,
                        "avg_saturation": avg_sat,
                        "confidence": self._calculate_fire_knife_confidence(
                            area, aspect_ratio, avg_hue, avg_sat
                        ),
                    }

                    fire_knives.append(fire_knife)

            # Sort by confidence and return top detections
            fire_knives.sort(key=lambda x: x["confidence"], reverse=True)
            return fire_knives[:3]  # Return top 3 most confident detections

        except Exception as e:
            print(f"   ⚠️ Fire knife detection error: {e}")
            return []

    def _detect_bombs(
        self, hsv: np.ndarray, frame: np.ndarray, gray: np.ndarray
    ) -> List[Dict]:
        """Detect small bombs on the floor"""
        try:
            # Create mask for bright objects (bombs are typically bright)
            bomb_mask = cv2.inRange(hsv, self.bomb_lower, self.bomb_upper)

            # Additional brightness filter in RGB space
            bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

            # Combine masks
            combined_mask = cv2.bitwise_and(bomb_mask, bright_mask)

            # Morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            bombs = []

            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by size - bombs should be small
                if self.bomb_min_area < area < self.bomb_max_area:
                    # Get properties
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w // 2, y + h // 2

                    # Check if it's on the floor (bottom part of screen)
                    is_on_floor = center_y > self.floor_threshold

                    # Calculate circularity (bombs are typically round)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = (
                        4 * np.pi * area / (perimeter * perimeter)
                        if perimeter > 0
                        else 0
                    )

                    # Get brightness
                    roi_gray = gray[y : y + h, x : x + w]
                    avg_brightness = np.mean(roi_gray) if roi_gray.size > 0 else 0

                    bomb = {
                        "type": "bomb",
                        "position": (center_x, center_y),
                        "bbox": (x, y, w, h),
                        "area": area,
                        "circularity": circularity,
                        "is_on_floor": is_on_floor,
                        "avg_brightness": avg_brightness,
                        "confidence": self._calculate_bomb_confidence(
                            area, circularity, is_on_floor, avg_brightness
                        ),
                    }

                    bombs.append(bomb)

            # Sort by confidence and return top detections
            bombs.sort(key=lambda x: x["confidence"], reverse=True)
            return bombs[:5]  # Return top 5 most confident bomb detections

        except Exception as e:
            print(f"   ⚠️ Bomb detection error: {e}")
            return []

    def _detect_motion(self, gray: np.ndarray) -> Dict:
        """Detect motion for validation of moving projectiles"""
        try:
            if self.prev_frame_gray is None:
                self.prev_frame_gray = gray.copy()
                return {"has_motion": False, "motion_areas": []}

            # Calculate frame difference
            frame_diff = cv2.absdiff(self.prev_frame_gray, gray)

            # Threshold to get motion areas
            _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

            # Find motion contours
            contours, _ = cv2.findContours(
                motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            motion_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter small motions
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_areas.append((x, y, w, h))

            # Update previous frame
            self.prev_frame_gray = gray.copy()

            return {"has_motion": len(motion_areas) > 0, "motion_areas": motion_areas}

        except Exception as e:
            print(f"   ⚠️ Motion detection error: {e}")
            return {"has_motion": False, "motion_areas": []}

    def _calculate_fire_knife_confidence(
        self, area: float, aspect_ratio: float, avg_hue: float, avg_sat: float
    ) -> float:
        """Calculate confidence score for fire knife detection"""
        confidence = 0.0

        # Size confidence (bigger is better for fire knife, but not too big)
        if 500 < area < 2500:
            confidence += 0.4
        elif 150 < area < 4000:
            confidence += 0.2

        # Color confidence (orange/red hues)
        if 8 < avg_hue < 22 and avg_sat > 100:  # Good orange/red
            confidence += 0.3
        elif 5 < avg_hue < 25:  # Acceptable range
            confidence += 0.15

        # Shape confidence (somewhat elongated is good)
        if 0.5 < aspect_ratio < 3.0:
            confidence += 0.2

        # Saturation confidence (fire should be saturated)
        if avg_sat > 150:
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_bomb_confidence(
        self, area: float, circularity: float, is_on_floor: bool, avg_brightness: float
    ) -> float:
        """Calculate confidence score for bomb detection"""
        confidence = 0.0

        # Size confidence (small objects)
        if 50 < area < 200:
            confidence += 0.3
        elif 25 < area < 350:
            confidence += 0.15

        # Shape confidence (round objects)
        if circularity > 0.7:
            confidence += 0.25
        elif circularity > 0.5:
            confidence += 0.1

        # Position confidence (on floor)
        if is_on_floor:
            confidence += 0.25

        # Brightness confidence (bombs are bright)
        if avg_brightness > 200:
            confidence += 0.2
        elif avg_brightness > 150:
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_threat_level(
        self, fire_knives: List[Dict], bombs: List[Dict]
    ) -> float:
        """Calculate overall threat level from detections"""
        threat_level = 0.0

        # Fire knife threats (immediate danger)
        for fk in fire_knives:
            threat_level += fk["confidence"] * 0.8  # High weight for fire knives

        # Bomb threats (delayed danger)
        for bomb in bombs:
            threat_level += bomb["confidence"] * 0.5  # Medium weight for bombs

        return min(1.0, threat_level)

    def _empty_detection(self) -> Dict:
        """Return empty detection result"""
        return {
            "fire_knives": [],
            "bombs": [],
            "motion_detected": False,
            "total_threats": 0,
            "threat_level": 0.0,
        }


class CNNFeatureExtractor(nn.Module):
    """CNN to extract features from 8-frame stack (180×128 resolution)"""

    def __init__(self, input_channels=24, feature_dim=256):  # 8 frames * 3 channels
        super().__init__()
        self.feature_dim = feature_dim

        # Optimized CNN architecture for 180×128 input
        self.cnn = nn.Sequential(
            # First conv block - reduce spatial size quickly
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Project to desired feature dimension
        self.projection = nn.Linear(256, feature_dim)

    def forward(self, frame_stack: torch.Tensor) -> torch.Tensor:
        try:
            # Apply CNN layers
            cnn_output = self.cnn(frame_stack)

            # Project to final feature dimension
            features = self.projection(cnn_output)

            return features

        except Exception as e:
            print(f"CNN error: {e}")
            batch_size = frame_stack.shape[0] if len(frame_stack.shape) > 0 else 1
            return torch.zeros(batch_size, self.feature_dim, device=frame_stack.device)


class MomentumTracker:
    """Tracks health and score momentum over time"""

    def __init__(self, history_length=8):
        self.history_length = history_length
        self.health_history = deque(maxlen=history_length)
        self.score_history = deque(maxlen=history_length)

    def update(self, health: float, score: float) -> np.ndarray:
        self.health_history.append(health)
        self.score_history.append(score)

        # Calculate momentum features
        health_momentum = self._calculate_momentum(self.health_history)
        score_momentum = self._calculate_momentum(self.score_history)

        # Create 8-dimensional feature vector
        features = np.array(
            [
                health_momentum,  # Health change rate
                score_momentum,  # Score change rate
                health / 200.0,  # Normalized current health
                score / 5000.0,  # Normalized current score
                1.0 if health_momentum < -5 else 0.0,  # Rapid health loss flag
                1.0 if score_momentum > 50 else 0.0,  # Rapid score gain flag
                len(self.health_history) / self.history_length,  # History completeness
                0.0,  # Reserved for future use
            ],
            dtype=np.float32,
        )

        return features

    def _calculate_momentum(self, history):
        """Calculate momentum (rate of change)"""
        if len(history) < 2:
            return 0.0

        values = list(history)
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]

        # Use recent changes (last 3) if available, otherwise all changes
        recent_changes = changes[-3:] if len(changes) >= 3 else changes
        return np.mean(recent_changes)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class VisionTransformer(nn.Module):
    """Vision Transformer for combining visual, OpenCV, and momentum features"""

    def __init__(self, visual_dim=256, opencv_dim=10, momentum_dim=8, seq_length=8):
        super().__init__()
        self.seq_length = seq_length

        # Combined input dimension: 256 + 10 + 8 = 274
        combined_dim = visual_dim + opencv_dim + momentum_dim

        # Project to transformer dimension
        d_model = 256
        self.input_projection = nn.Linear(combined_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Prediction heads
        self.health_predictor = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

        self.score_predictor = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

        self.action_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),  # Attack, Defend, Wait
        )

    def forward(self, combined_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            # Project input features
            projected = self.input_projection(combined_sequence)
            projected = self.pos_encoding(projected)

            # Apply transformer
            transformer_out = self.transformer(projected)
            final_features = transformer_out[:, -1, :]  # Use last timestep

            # Generate predictions
            health_pred = self.health_predictor(final_features)
            score_pred = self.score_predictor(final_features)
            action_logits = self.action_predictor(final_features)

            return {
                "predicted_health_change": health_pred,
                "predicted_score_change": score_pred,
                "action_probabilities": torch.softmax(action_logits, dim=-1),
            }

        except Exception as e:
            print(f"Transformer error: {e}")
            batch_size = combined_sequence.shape[0]
            device = combined_sequence.device
            return {
                "predicted_health_change": torch.zeros(batch_size, 1, device=device),
                "predicted_score_change": torch.zeros(batch_size, 1, device=device),
                "action_probabilities": torch.ones(batch_size, 3, device=device) / 3,
            }


class VisionPipelineWrapper(gym.Wrapper):
    """Main wrapper integrating OpenCV + CNN + Vision Transformer pipeline"""

    def __init__(self, env, frame_stack=8, enable_vision_transformer=True):
        super().__init__(env)

        self.frame_stack = frame_stack
        self.enable_vision_transformer = enable_vision_transformer
        self.target_size = (128, 180)  # H, W - optimized for 180×128

        # Setup observation space: [channels, height, width]
        obs_shape = (3 * frame_stack, self.target_size[0], self.target_size[1])
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        # Initialize pipeline components
        self.opencv_detector = EnhancedOpenCVDetector()
        self.momentum_tracker = MomentumTracker()

        # Frame and feature buffers
        self.frame_buffer = deque(maxlen=frame_stack)
        self.visual_features_history = deque(maxlen=frame_stack)
        self.opencv_features_history = deque(maxlen=frame_stack)
        self.momentum_features_history = deque(maxlen=frame_stack)

        # Vision transformer components (initialized when feature extractor is injected)
        self.cnn_extractor = None
        self.vision_transformer = None
        self.vision_ready = False

        # Game state tracking
        self.prev_health = MAX_HEALTH
        self.prev_score = 0

        # Statistics
        self.stats = {
            "fire_knives_detected": 0,
            "bombs_detected": 0,
            "predictions_made": 0,
            "vision_transformer_ready": False,
        }

        print(f"🎮 Vision Pipeline Wrapper initialized:")
        print(f"   Resolution: {self.target_size[1]}×{self.target_size[0]}")
        print(f"   Frame stack: {frame_stack}")
        print(
            f"   Vision Transformer: {'Enabled' if enable_vision_transformer else 'Disabled'}"
        )

    def inject_feature_extractor(self, feature_extractor):
        """Inject CNN feature extractor and initialize vision transformer"""
        if not self.enable_vision_transformer:
            print("   🔧 Vision Transformer disabled")
            return

        try:
            self.cnn_extractor = feature_extractor

            # Initialize vision transformer
            device = next(feature_extractor.parameters()).device
            self.vision_transformer = VisionTransformer().to(device)
            self.vision_ready = True
            self.stats["vision_transformer_ready"] = True

            print("   ✅ Vision Transformer initialized and ready!")

        except Exception as e:
            print(f"   ❌ Vision Transformer injection failed: {e}")
            self.vision_ready = False

    def reset(self, **kwargs):
        """Reset environment and initialize buffers"""
        obs, info = self.env.reset(**kwargs)

        # Reset game state
        self.prev_health = info.get("health", MAX_HEALTH)
        self.prev_score = info.get("score", 0)

        # Initialize frame buffer with preprocessed frames
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        # Clear feature history buffers
        self.visual_features_history.clear()
        self.opencv_features_history.clear()
        self.momentum_features_history.clear()

        return self._get_stacked_observation(), info

    def step(self, action):
        """Execute action and process through vision pipeline"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Process new frame
        processed_frame = self._preprocess_frame(obs)
        self.frame_buffer.append(processed_frame)

        # Get current game state
        current_health = info.get("health", self.prev_health)
        current_score = info.get("score", self.prev_score)

        # Process through vision pipeline
        stacked_obs = self._get_stacked_observation()
        self._process_vision_pipeline(stacked_obs, current_health, current_score)

        # Update game state
        self.prev_health = current_health
        self.prev_score = current_score

        # Add stats to info
        info.update(self.stats)

        return stacked_obs, reward, terminated, truncated, info

    def _preprocess_frame(self, frame):
        """Preprocess frame to target size (128, 180)"""
        try:
            if frame is None:
                return np.zeros((*self.target_size, 3), dtype=np.uint8)

            # Resize to target size: height=128, width=180
            resized = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
            return resized

        except Exception as e:
            print(f"Frame preprocessing error: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.uint8)

    def _get_stacked_observation(self):
        """Get stacked observation in CHW format"""
        try:
            if len(self.frame_buffer) == 0:
                return np.zeros(self.observation_space.shape, dtype=np.uint8)

            # Stack frames: [frame1, frame2, ..., frame8] each [H, W, 3]
            # Convert to [3*8, H, W] format
            stacked = np.concatenate(list(self.frame_buffer), axis=2)  # [H, W, 24]
            return stacked.transpose(2, 0, 1)  # [24, H, W]

        except Exception as e:
            print(f"Frame stacking error: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def _process_vision_pipeline(self, stacked_obs, health, score):
        """Process through complete vision pipeline"""
        try:
            # Step 1: OpenCV detection on latest frame
            latest_frame = self._extract_latest_frame_rgb(stacked_obs)
            opencv_detections = self.opencv_detector.detect_threats(latest_frame)
            opencv_features = self._opencv_to_features(opencv_detections)

            # Update detection statistics
            self.stats["fire_knives_detected"] += len(
                opencv_detections.get("fire_knives", [])
            )
            self.stats["bombs_detected"] += len(opencv_detections.get("bombs", []))

            # Step 2: Momentum tracking
            momentum_features = self.momentum_tracker.update(health, score)

            # Step 3: CNN feature extraction (if available)
            if self.cnn_extractor is not None:
                with torch.no_grad():
                    device = next(self.cnn_extractor.parameters()).device
                    obs_tensor = (
                        torch.from_numpy(stacked_obs).float().unsqueeze(0).to(device)
                    )
                    visual_features = (
                        self.cnn_extractor(obs_tensor).squeeze(0).cpu().numpy()
                    )
            else:
                visual_features = np.zeros(256, dtype=np.float32)

            # Store in history buffers
            self.visual_features_history.append(visual_features)
            self.opencv_features_history.append(opencv_features)
            self.momentum_features_history.append(momentum_features)

            # Step 4: Vision transformer prediction (if ready and enough history)
            if (
                self.vision_ready
                and len(self.visual_features_history) == self.frame_stack
            ):
                prediction = self._make_vision_prediction()
                if prediction:
                    self.stats["predictions_made"] += 1

                    # Log significant predictions
                    health_pred = prediction["predicted_health_change"]
                    score_pred = prediction["predicted_score_change"]

                    if abs(health_pred) > 5:
                        print(f"   🔮 Health prediction: {health_pred:.1f}")
                    if abs(score_pred) > 100:
                        print(f"   💰 Score prediction: {score_pred:.1f}")

        except Exception as e:
            print(f"Vision pipeline processing error: {e}")

    def _extract_latest_frame_rgb(self, stacked_obs):
        """Extract latest frame in RGB format for OpenCV"""
        try:
            # stacked_obs shape: [24, 128, 180] = [8*3, H, W]
            # Latest frame is last 3 channels: [21:24, :, :]
            latest_frame_chw = stacked_obs[-3:]  # [3, 128, 180]
            latest_frame_hwc = latest_frame_chw.transpose(1, 2, 0)  # [128, 180, 3]
            return latest_frame_hwc

        except Exception as e:
            print(f"Frame extraction error: {e}")
            return np.zeros((128, 180, 3), dtype=np.uint8)

    def _opencv_to_features(self, opencv_detections):
        """Convert OpenCV detections to fixed-size feature vector (10 dims)"""
        try:
            features = np.zeros(10, dtype=np.float32)

            # Fire knife features (4 features: count, x, y, confidence)
            fire_knives = opencv_detections.get("fire_knives", [])
            features[0] = min(len(fire_knives), 5.0)  # Clamp count to reasonable range

            if fire_knives:
                fk = fire_knives[0]  # Most confident detection
                features[1] = fk["position"][0] / 180.0  # Normalized x position
                features[2] = fk["position"][1] / 128.0  # Normalized y position
                features[3] = fk["confidence"]  # Confidence score

            # Bomb features (4 features: count, x, y, confidence)
            bombs = opencv_detections.get("bombs", [])
            features[4] = min(len(bombs), 5.0)  # Clamp count to reasonable range

            if bombs:
                bomb = bombs[0]  # Most confident detection
                features[5] = bomb["position"][0] / 180.0  # Normalized x position
                features[6] = bomb["position"][1] / 128.0  # Normalized y position
                features[7] = bomb["confidence"]  # Confidence score

            # General features (2 features: threat level, motion)
            features[8] = opencv_detections.get("threat_level", 0.0)
            features[9] = (
                1.0 if opencv_detections.get("motion_detected", False) else 0.0
            )

            return features

        except Exception as e:
            print(f"OpenCV feature conversion error: {e}")
            return np.zeros(10, dtype=np.float32)

    def _make_vision_prediction(self):
        """Make prediction using vision transformer"""
        try:
            if (
                not self.vision_ready
                or len(self.visual_features_history) < self.frame_stack
            ):
                return None

            # Stack sequences
            visual_seq = np.stack(list(self.visual_features_history))  # [8, 256]
            opencv_seq = np.stack(list(self.opencv_features_history))  # [8, 10]
            momentum_seq = np.stack(list(self.momentum_features_history))  # [8, 8]

            # Combine features at each timestep
            combined_seq = np.concatenate(
                [visual_seq, opencv_seq, momentum_seq], axis=1
            )  # [8, 274]

            # Convert to tensor and add batch dimension
            device = next(self.vision_transformer.parameters()).device
            combined_tensor = (
                torch.from_numpy(combined_seq).float().unsqueeze(0).to(device)
            )  # [1, 8, 274]

            # Get prediction from vision transformer
            with torch.no_grad():
                predictions = self.vision_transformer(combined_tensor)

            return {
                "predicted_health_change": predictions["predicted_health_change"]
                .cpu()
                .item(),
                "predicted_score_change": predictions["predicted_score_change"]
                .cpu()
                .item(),
                "action_probabilities": predictions["action_probabilities"]
                .cpu()
                .squeeze()
                .tolist(),
            }

        except Exception as e:
            print(f"Vision prediction error: {e}")
            return None


# Enhanced CNN for stable-baselines3 compatibility
class JEPAEnhancedCNN(BaseFeaturesExtractor):
    """Enhanced CNN feature extractor compatible with stable-baselines3"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = CNNFeatureExtractor(
            input_channels=n_input_channels, feature_dim=features_dim
        )

        print(f"🏗️ JEPA Enhanced CNN initialized:")
        print(
            f"   Input: {n_input_channels} channels → Output: {features_dim} features"
        )
        print(f"   Expected input shape: {observation_space.shape}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize observations to [0, 1] range
        normalized_obs = observations.float() / 255.0
        return self.cnn(normalized_obs)
