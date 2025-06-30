"""
Copy-move detection functions
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.preprocessing import normalize as sk_normalize
from feature_detection import match_sift_features, match_orb_features, match_akaze_features
from config import *

def detect_copy_move_advanced(feature_sets, image_shape,
                            ratio_thresh=RATIO_THRESH, min_distance=MIN_DISTANCE,
                            ransac_thresh=RANSAC_THRESH, min_inliers=MIN_INLIERS):
    """Advanced copy-move detection dengan multiple features"""
    all_matches = []
    best_inliers = 0
    best_transform = None
    
    for detector_name, (keypoints, descriptors) in feature_sets.items():
        if descriptors is None or len(descriptors) < 10:
            continue
        
        print(f"  - Analyzing {detector_name.upper()} features: {len(keypoints)} keypoints")
        
        # Feature matching
        if detector_name == 'sift':
            matches, inliers, transform = match_sift_features(
                keypoints, descriptors, ratio_thresh, min_distance, ransac_thresh, min_inliers)
        elif detector_name == 'orb':
            matches, inliers, transform = match_orb_features(
                keypoints, descriptors, min_distance, ransac_thresh, min_inliers)
        else:  # akaze
            matches, inliers, transform = match_akaze_features(
                keypoints, descriptors, min_distance, ransac_thresh, min_inliers)
        
        all_matches.extend(matches)
        if inliers > best_inliers:
            best_inliers = inliers
            best_transform = transform
    
    return all_matches, best_inliers, best_transform

def detect_copy_move_blocks(image_pil, block_size=BLOCK_SIZE, threshold=0.95):
    """Enhanced block-based copy-move detection"""
    print("  - Block-based copy-move detection...")
    
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    image_array = np.array(image_pil)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    blocks = {}
    matches = []
    
    # Extract blocks with sliding window
    for y in range(0, h - block_size, block_size // 2):
        for x in range(0, w - block_size, block_size // 2):
            block = gray[y:y+block_size, x:x+block_size]
            
            # Calculate block hash/signature
            block_hash = cv2.resize(block, (8, 8)).flatten()
            block_normalized = block_hash / (np.linalg.norm(block_hash) + 1e-10)
            
            # Store block info
            block_key = tuple(block_normalized.round(3))
            if block_key not in blocks:
                blocks[block_key] = []
            blocks[block_key].append((x, y, block))
    
    # Find matching blocks
    for block_positions in blocks.values():
        if len(block_positions) > 1:
            for i in range(len(block_positions)):
                for j in range(i + 1, len(block_positions)):
                    x1, y1, block1 = block_positions[i]
                    x2, y2, block2 = block_positions[j]
                    
                    # Check spatial distance
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < block_size * 2:
                        continue
                    
                    # Calculate correlation
                    correlation = cv2.matchTemplate(block1, block2, cv2.TM_CCOEFF_NORMED)[0][0]
                    if correlation > threshold:
                        matches.append({
                            'block1': (x1, y1),
                            'block2': (x2, y2),
                            'correlation': correlation,
                            'distance': distance
                        })
    
    # Remove duplicate matches
    unique_matches = []
    for match in matches:
        is_duplicate = False
        for existing in unique_matches:
            if (abs(match['block1'][0] - existing['block1'][0]) < block_size and
                abs(match['block1'][1] - existing['block1'][1]) < block_size):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_matches.append(match)
    
    return unique_matches

def kmeans_tampering_localization(image_pil, ela_image, n_clusters=3):
    """K-means clustering untuk localization tampering - OPTIMIZED VERSION"""
    print("🔍 Performing K-means tampering localization...")
    
    # Konversi ke array
    image_array = np.array(image_pil.convert('RGB'))
    ela_array = np.array(ela_image)
    h, w = ela_array.shape
    
    # Adaptive block size and sampling based on image size
    total_pixels = h * w
    if total_pixels < 500000:  # Small image
        block_size = 8
        block_step = 4
    elif total_pixels < 2000000:  # Medium image
        block_size = 16
        block_step = 8
    else:  # Large image
        block_size = 32
        block_step = 16
    
    print(f"  - Using block_size={block_size}, step={block_step} for {h}x{w} image")
    
    # Ekstrak features untuk clustering
    features = []
    coordinates = []
    
    # Feature extraction per block dengan adaptive sampling
    for i in range(0, h-block_size, block_step):
        for j in range(0, w-block_size, block_step):
            # ELA features
            ela_block = ela_array[i:i+block_size, j:j+block_size]
            ela_mean = np.mean(ela_block)
            ela_std = np.std(ela_block)
            ela_max = np.max(ela_block)
            
            # Color features
            rgb_block = image_array[i:i+block_size, j:j+block_size]
            rgb_mean = np.mean(rgb_block, axis=(0,1))
            rgb_std = np.std(rgb_block, axis=(0,1))
            
            # Texture features (simple)
            gray_block = cv2.cvtColor(rgb_block, cv2.COLOR_RGB2GRAY)
            texture_var = np.var(gray_block)
            
            # Combine features
            feature_vector = [
                ela_mean, ela_std, ela_max,
                rgb_mean[0], rgb_mean[1], rgb_mean[2],
                rgb_std[0], rgb_std[1], rgb_std[2],
                texture_var
            ]
            
            features.append(feature_vector)
            coordinates.append((i, j))
    
    features = np.array(features)
    print(f"  - Total features for K-means: {len(features)}")
    
    # K-means clustering with error handling
    try:
        # Use mini-batch K-means for large datasets
        if len(features) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                   batch_size=100, n_init=3)
            print("  - Using MiniBatchKMeans for efficiency")
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        cluster_labels = kmeans.fit_predict(features)
    except MemoryError:
        print("  ⚠ Memory error in K-means, reducing clusters")
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        cluster_labels = kmeans.fit_predict(features)
    
    # Create localization map
    localization_map = np.zeros((h, w))
    
    # Fill the map based on clustering results
    for idx, (i, j) in enumerate(coordinates):
        cluster_id = cluster_labels[idx]
        # Fill the block area
        i_end = min(i + block_size, h)
        j_end = min(j + block_size, w)
        localization_map[i:i_end, j:j_end] = cluster_id
    
    # Identify tampering clusters (highest ELA response)
    cluster_ela_means = []
    for cluster_id in range(n_clusters):
        cluster_mask = (localization_map == cluster_id)
        if np.sum(cluster_mask) > 0:
            cluster_ela_mean = np.mean(ela_array[cluster_mask])
            cluster_ela_means.append(cluster_ela_mean)
        else:
            cluster_ela_means.append(0)
    
    # Cluster dengan ELA tertinggi dianggap sebagai tampering
    tampering_cluster = np.argmax(cluster_ela_means)
    tampering_mask = (localization_map == tampering_cluster)
    
    return {
        'localization_map': localization_map,
        'tampering_mask': tampering_mask,
        'cluster_labels': cluster_labels,
        'cluster_centers': kmeans.cluster_centers_,
        'tampering_cluster_id': tampering_cluster,
        'cluster_ela_means': cluster_ela_means
    }
