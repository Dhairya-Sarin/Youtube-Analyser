import cv2
import numpy as np
import requests
from PIL import Image, ImageStat
import io
from typing import Dict, Any, List, Optional
from .base_extractor import BaseFeatureExtractor
from config.constants import COLOR_RANGES, THUMBNAIL_THRESHOLDS


class ThumbnailVisionExtractor(BaseFeatureExtractor):
    """Extract computer vision features from video thumbnails"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_names = [
            # Basic image properties
            'thumbnail_width', 'thumbnail_height', 'thumbnail_aspect_ratio',
            'thumbnail_file_size', 'thumbnail_resolution_score',

            # Color analysis
            'thumbnail_brightness', 'thumbnail_contrast', 'thumbnail_saturation',
            'thumbnail_red_mean', 'thumbnail_green_mean', 'thumbnail_blue_mean',
            'thumbnail_dominant_color_hue', 'thumbnail_color_diversity',

            # Visual complexity
            'thumbnail_edge_density', 'thumbnail_sharpness', 'thumbnail_blur_score',
            'thumbnail_visual_complexity', 'thumbnail_symmetry_score',

            # Face and object detection
            'thumbnail_face_count', 'thumbnail_largest_face_area', 'thumbnail_face_center_score',
            'thumbnail_eye_contact_score', 'thumbnail_smile_detection',

            # Text detection
            'thumbnail_text_detected', 'thumbnail_text_area_ratio', 'thumbnail_text_position_score',
            'thumbnail_text_contrast_score', 'thumbnail_caps_text_ratio',

            # Clickbait visual patterns
            'thumbnail_has_arrows', 'thumbnail_has_circles', 'thumbnail_has_red_elements',
            'thumbnail_has_shocked_face', 'thumbnail_clickbait_visual_score',

            # Aesthetic scores
            'thumbnail_rule_of_thirds_score', 'thumbnail_golden_ratio_score',
            'thumbnail_color_harmony_score', 'thumbnail_aesthetic_appeal'
        ]

        # Initialize face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        except:
            self.face_cascade = None
            self.eye_cascade = None

    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract thumbnail vision features"""
        thumbnail_url = data.get('thumbnail_url', '')
        if not thumbnail_url:
            return {name: 0 for name in self.feature_names}

        try:
            # Download thumbnail
            response = requests.get(thumbnail_url, timeout=10)
            if response.status_code != 200:
                return {name: 0 for name in self.feature_names}

            # Load image
            img_array = np.frombuffer(response.content, np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_pil = Image.open(io.BytesIO(response.content))

            if img_cv is None or img_pil is None:
                return {name: 0 for name in self.feature_names}

            features = {}

            # Basic properties
            features.update(self._extract_basic_properties(img_pil, len(response.content)))

            # Color analysis
            features.update(self._extract_color_features(img_cv, img_pil))

            # Visual complexity
            features.update(self._extract_complexity_features(img_cv))

            # Face detection
            features.update(self._extract_face_features(img_cv))

            # Text detection
            features.update(self._extract_text_features(img_cv))

            # Clickbait patterns
            features.update(self._extract_clickbait_features(img_cv))

            # Aesthetic analysis
            features.update(self._extract_aesthetic_features(img_cv))

            return features

        except Exception as e:
            print(f"Error processing thumbnail: {e}")
            return {name: 0 for name in self.feature_names}

    def _extract_basic_properties(self, img_pil: Image.Image, file_size: int) -> Dict[str, Any]:
        """Extract basic image properties"""
        width, height = img_pil.size

        return {
            'thumbnail_width': width,
            'thumbnail_height': height,
            'thumbnail_aspect_ratio': width / max(height, 1),
            'thumbnail_file_size': file_size,
            'thumbnail_resolution_score': (width * height) / (1920 * 1080)  # Normalized to 1080p
        }

    def _extract_color_features(self, img_cv: np.ndarray, img_pil: Image.Image) -> Dict[str, Any]:
        """Extract color-based features"""
        features = {}

        # Convert to different color spaces
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Basic color statistics
        b, g, r = cv2.split(img_cv)
        features['thumbnail_red_mean'] = np.mean(r)
        features['thumbnail_green_mean'] = np.mean(g)
        features['thumbnail_blue_mean'] = np.mean(b)

        # Brightness and contrast
        features['thumbnail_brightness'] = np.mean(gray)
        features['thumbnail_contrast'] = np.std(gray)

        # Saturation
        features['thumbnail_saturation'] = np.mean(hsv[:, :, 1])

        # Dominant color (hue)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        features['thumbnail_dominant_color_hue'] = np.argmax(hue_hist)

        # Color diversity (number of distinct colors)
        features['thumbnail_color_diversity'] = len(np.unique(img_cv.reshape(-1, img_cv.shape[-1]), axis=0))

        return features

    def _extract_complexity_features(self, img_cv: np.ndarray) -> Dict[str, Any]:
        """Extract visual complexity features"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Blur detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Visual complexity (combination of edges and texture)
        complexity = edge_density * np.std(gray)

        # Symmetry (simplified)
        height, width = gray.shape
        left_half = gray[:, :width // 2]
        right_half = cv2.flip(gray[:, width // 2:], 1)
        if left_half.shape == right_half.shape:
            symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
        else:
            symmetry = 0.0

        return {
            'thumbnail_edge_density': edge_density,
            'thumbnail_sharpness': sharpness,
            'thumbnail_blur_score': 1.0 / (1.0 + blur_score),  # Inverse: higher = more blurry
            'thumbnail_visual_complexity': complexity,
            'thumbnail_symmetry_score': symmetry
        }

    def _extract_face_features(self, img_cv: np.ndarray) -> Dict[str, Any]:
        """Extract face-related features"""
        features = {
            'thumbnail_face_count': 0,
            'thumbnail_largest_face_area': 0,
            'thumbnail_face_center_score': 0,
            'thumbnail_eye_contact_score': 0,
            'thumbnail_smile_detection': 0
        }

        if self.face_cascade is None:
            return features

        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            features['thumbnail_face_count'] = len(faces)

            if len(faces) > 0:
                # Largest face
                face_areas = [(w * h, (x, y, w, h)) for x, y, w, h in faces]
                largest_area, largest_face = max(face_areas)

                features['thumbnail_largest_face_area'] = largest_area / (img_cv.shape[0] * img_cv.shape[1])

                # Face center score (how centered the largest face is)
                x, y, w, h = largest_face
                face_center_x = x + w / 2
                face_center_y = y + h / 2
                img_center_x = img_cv.shape[1] / 2
                img_center_y = img_cv.shape[0] / 2

                center_distance = np.sqrt((face_center_x - img_center_x) ** 2 + (face_center_y - img_center_y) ** 2)
                max_distance = np.sqrt(img_center_x ** 2 + img_center_y ** 2)
                features['thumbnail_face_center_score'] = 1.0 - (center_distance / max_distance)

                # Eye detection (simplified)
                if self.eye_cascade is not None:
                    face_roi = gray[y:y + h, x:x + w]
                    eyes = self.eye_cascade.detectMultiScale(face_roi)
                    features['thumbnail_eye_contact_score'] = min(len(eyes) / 2.0, 1.0)  # Normalize to 0-1

        except:
            pass

        return features

    def _extract_text_features(self, img_cv: np.ndarray) -> Dict[str, Any]:
        """Extract text-related features (simplified)"""
        # Note: For production, consider using OCR libraries like pytesseract
        features = {
            'thumbnail_text_detected': 0,
            'thumbnail_text_area_ratio': 0,
            'thumbnail_text_position_score': 0,
            'thumbnail_text_contrast_score': 0,
            'thumbnail_caps_text_ratio': 0
        }

        # Simplified text detection using edge analysis
        # Look for rectangular regions that might contain text
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours that might be text
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_like_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / max(h, 1)
            area = w * h

            # Heuristic: text-like shapes
            if 0.5 < aspect_ratio < 8 and 100 < area < 50000:
                text_like_contours.append((x, y, w, h))

        if text_like_contours:
            features['thumbnail_text_detected'] = 1
            total_text_area = sum(w * h for x, y, w, h in text_like_contours)
            features['thumbnail_text_area_ratio'] = total_text_area / (img_cv.shape[0] * img_cv.shape[1])

        return features

    def _extract_clickbait_features(self, img_cv: np.ndarray) -> Dict[str, Any]:
        """Extract clickbait visual pattern features"""
        features = {
            'thumbnail_has_arrows': 0,
            'thumbnail_has_circles': 0,
            'thumbnail_has_red_elements': 0,
            'thumbnail_has_shocked_face': 0,
            'thumbnail_clickbait_visual_score': 0
        }

        # Red elements detection
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = red_mask1 + red_mask2

        red_ratio = np.sum(red_mask > 0) / (red_mask.shape[0] * red_mask.shape[1])
        features['thumbnail_has_red_elements'] = 1 if red_ratio > 0.05 else 0

        # Circle detection (for highlighting)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=10, maxRadius=100)

        features['thumbnail_has_circles'] = 1 if circles is not None else 0

        # Simple arrow detection (look for triangular shapes)
        contours, _ = cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arrow_like_shapes = 0
        for contour in contours:
            # Simplified: look for triangular approximations
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 3:  # Triangle
                arrow_like_shapes += 1

        features['thumbnail_has_arrows'] = 1 if arrow_like_shapes > 0 else 0

        # Clickbait visual score
        score = 0
        score += features['thumbnail_has_red_elements'] * 0.3
        score += features['thumbnail_has_circles'] * 0.2
        score += features['thumbnail_has_arrows'] * 0.3
        score += min(red_ratio * 5, 0.2)  # High red usage

        features['thumbnail_clickbait_visual_score'] = score

        return features

    def _extract_aesthetic_features(self, img_cv: np.ndarray) -> Dict[str, Any]:
        """Extract aesthetic composition features"""
        height, width = img_cv.shape[:2]

        # Rule of thirds
        third_x = width / 3
        third_y = height / 3

        # Simplified: check if main subject (brightest/most contrasted area) follows rule of thirds
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Find the most prominent region
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rule_of_thirds_score = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Check proximity to rule of thirds lines
                thirds_points = [(third_x, third_y), (2 * third_x, third_y),
                                 (third_x, 2 * third_y), (2 * third_x, 2 * third_y)]

                min_distance = min(np.sqrt((cx - px) ** 2 + (cy - py) ** 2) for px, py in thirds_points)
                max_distance = np.sqrt(width ** 2 + height ** 2)
                rule_of_thirds_score = 1.0 - (min_distance / max_distance)

        # Golden ratio (simplified)
        golden_ratio = 1.618
        actual_ratio = width / height
        golden_ratio_score = 1.0 - abs(actual_ratio - golden_ratio) / golden_ratio
        golden_ratio_score = max(0, golden_ratio_score)

        # Color harmony (simplified)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180])  # 12 color bins
        hue_hist = hue_hist.flatten() / np.sum(hue_hist)

        # Measure how well distributed colors are (harmony)
        color_harmony_score = 1.0 - np.std(hue_hist)

        # Overall aesthetic appeal (combination)
        aesthetic_appeal = (rule_of_thirds_score * 0.4 +
                            golden_ratio_score * 0.3 +
                            color_harmony_score * 0.3)

        return {
            'thumbnail_rule_of_thirds_score': rule_of_thirds_score,
            'thumbnail_golden_ratio_score': golden_ratio_score,
            'thumbnail_color_harmony_score': color_harmony_score,
            'thumbnail_aesthetic_appeal': aesthetic_appeal
        }

    def get_feature_names(self) -> List[str]:
        return self.feature_names