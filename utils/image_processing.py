import cv2
import numpy as np
from PIL import Image
import io
from typing import Dict, List, Tuple, Optional


class ImageProcessor:
    """Utility class for image processing operations"""

    def __init__(self):
        # Initialize face detection cascades
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
        except:
            self.face_cascade = None
            self.eye_cascade = None
            self.smile_cascade = None

    def analyze_color_distribution(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze color distribution in image"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        # Normalize histograms
        h_hist = h_hist.flatten() / np.sum(h_hist)
        s_hist = s_hist.flatten() / np.sum(s_hist)
        v_hist = v_hist.flatten() / np.sum(v_hist)

        # Calculate color diversity (entropy)
        color_entropy = -np.sum(h_hist * np.log2(h_hist + 1e-10))

        # Dominant hue
        dominant_hue = np.argmax(h_hist)

        # Color temperature (warm vs cool)
        warm_hues = np.sum(h_hist[0:30]) + np.sum(h_hist[150:180])  # Reds and oranges
        cool_hues = np.sum(h_hist[90:150])  # Blues and greens
        color_temperature = warm_hues / (warm_hues + cool_hues + 1e-10)

        return {
            'color_diversity': color_entropy / 7.49,  # Normalize to 0-1
            'dominant_hue': dominant_hue,
            'color_temperature': color_temperature,
            'saturation_mean': np.mean(hsv[:, :, 1]) / 255,
            'brightness_mean': np.mean(hsv[:, :, 2]) / 255
        }

    def detect_faces_advanced(self, image: np.ndarray) -> Dict[str, any]:
        """Advanced face detection with additional features"""
        if self.face_cascade is None:
            return {'face_count': 0, 'faces': []}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        face_data = []
        for (x, y, w, h) in faces:
            face_info = {
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': w * h,
                'center_x': x + w / 2,
                'center_y': y + h / 2,
                'area_ratio': (w * h) / (image.shape[0] * image.shape[1])
            }

            # Detect eyes in face region
            if self.eye_cascade is not None:
                face_roi = gray[y:y + h, x:x + w]
                eyes = self.eye_cascade.detectMultiScale(face_roi)
                face_info['eye_count'] = len(eyes)

            # Detect smile
            if self.smile_cascade is not None:
                face_roi = gray[y:y + h, x:x + w]
                smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
                face_info['has_smile'] = len(smiles) > 0

            face_data.append(face_info)

        # Calculate face positioning metrics
        face_center_score = 0
        if face_data:
            img_center_x = image.shape[1] / 2
            img_center_y = image.shape[0] / 2

            # Find the largest face and calculate its centrality
            largest_face = max(face_data, key=lambda f: f['area'])
            distance_from_center = np.sqrt(
                (largest_face['center_x'] - img_center_x) ** 2 +
                (largest_face['center_y'] - img_center_y) ** 2
            )
            max_distance = np.sqrt(img_center_x ** 2 + img_center_y ** 2)
            face_center_score = 1.0 - (distance_from_center / max_distance)

        return {
            'face_count': len(faces),
            'faces': face_data,
            'largest_face_area': max([f['area_ratio'] for f in face_data], default=0),
            'face_center_score': face_center_score,
            'has_multiple_faces': len(faces) > 1,
            'total_face_area': sum(f['area_ratio'] for f in face_data)
        }

    def analyze_composition(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze image composition using photography rules"""
        height, width = image.shape[:2]

        # Rule of thirds analysis
        third_x = width / 3
        third_y = height / 3

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find edges (important visual elements)
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge density at rule of thirds intersections
        intersections = [
            (int(third_x), int(third_y)),
            (int(2 * third_x), int(third_y)),
            (int(third_x), int(2 * third_y)),
            (int(2 * third_x), int(2 * third_y))
        ]

        intersection_scores = []
        window_size = 20

        for x, y in intersections:
            x1, y1 = max(0, x - window_size), max(0, y - window_size)
            x2, y2 = min(width, x + window_size), min(height, y + window_size)

            region = edges[y1:y2, x1:x2]
            score = np.sum(region) / (region.shape[0] * region.shape[1] * 255)
            intersection_scores.append(score)

        rule_of_thirds_score = np.mean(intersection_scores)

        # Symmetry analysis
        left_half = gray[:, :width // 2]
        right_half = cv2.flip(gray[:, width // 2:], 1)

        if left_half.shape == right_half.shape:
            symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
        else:
            symmetry_score = 0.0

        # Balance analysis (distribution of visual weight)
        # Calculate center of mass for the image
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            center_of_mass_x = moments['m10'] / moments['m00']
            center_of_mass_y = moments['m01'] / moments['m00']

            # How far is center of mass from image center?
            distance_from_center = np.sqrt(
                (center_of_mass_x - width / 2) ** 2 +
                (center_of_mass_y - height / 2) ** 2
            )
            max_distance = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
            balance_score = 1.0 - (distance_from_center / max_distance)
        else:
            balance_score = 0.0

        return {
            'rule_of_thirds_score': rule_of_thirds_score,
            'symmetry_score': symmetry_score,
            'balance_score': balance_score,
            'composition_score': (rule_of_thirds_score + symmetry_score + balance_score) / 3
        }

    def calculate_visual_complexity(self, image: np.ndarray) -> float:
        """Calculate visual complexity score"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Texture analysis using Local Binary Pattern approximation
        # Simple texture measure: standard deviation of local regions
        texture_scores = []
        window_size = 16

        for y in range(0, gray.shape[0] - window_size, window_size):
            for x in range(0, gray.shape[1] - window_size, window_size):
                region = gray[y:y + window_size, x:x + window_size]
                texture_scores.append(np.std(region))

        texture_complexity = np.mean(texture_scores) / 255 if texture_scores else 0

        # Color complexity
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        color_hist = color_hist.flatten() / np.sum(color_hist)
        color_entropy = -np.sum(color_hist * np.log2(color_hist + 1e-10))
        color_complexity = color_entropy / np.log2(50)  # Normalize

        # Combine measures
        complexity_score = (edge_density * 0.4 +
                            texture_complexity * 0.3 +
                            color_complexity * 0.3)

        return min(complexity_score, 1.0)

    def extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = image.reshape((-1, 3))

        # Apply K-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get the dominant colors
        colors = kmeans.cluster_centers_.astype(int)

        # Convert BGR to RGB
        colors = [(int(c[2]), int(c[1]), int(c[0])) for c in colors]

        return colors

    def resize_image(self, image: np.ndarray, max_size: int = 800) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]

        if max(height, width) <= max_size:
            return image

        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)