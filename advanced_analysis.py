"""
Advanced Analysis Module for Forensic Image Analysis System
Contains functions for noise, frequency, texture, edge, illumination, and statistical analysis
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import entropy
import warnings

# Conditional imports dengan error handling
try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.filters import sobel, prewitt, roberts
    from skimage.measure import shannon_entropy
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Some features will be limited.")
    SKIMAGE_AVAILABLE = False

# Import utilities dengan error handling
try:
    from utils import detect_outliers_iqr
except ImportError:
    print("Warning: utils module not found. Using fallback functions.")
    def detect_outliers_iqr(data, factor=1.5):
        """Fallback outlier detection"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return np.where((data < lower_bound) | (data > upper_bound))[0]

warnings.filterwarnings('ignore')

# ======================= Helper Functions =======================

def calculate_skewness(data):
    """Calculate skewness"""
    try:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    except Exception:
        return 0.0

def calculate_kurtosis(data):
    """Calculate kurtosis"""
    try:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    except Exception:
        return 0.0

def safe_entropy(data):
    """Safe entropy calculation with fallback"""
    try:
        if SKIMAGE_AVAILABLE:
            return shannon_entropy(data)
        else:
            # Fallback entropy calculation
            hist, _ = np.histogram(data.flatten(), bins=256, range=(0, 255))
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist + 1e-10))
    except Exception:
        return 0.0

# ======================= Noise Analysis =======================

def analyze_noise_consistency(image_pil, block_size=32):
    """Advanced noise consistency analysis"""
    print("  - Advanced noise consistency analysis...")
    
    try:
        image_array = np.array(image_pil.convert('RGB'))
        
        # Convert to different color spaces for comprehensive analysis
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        
        h, w, c = image_array.shape
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        noise_characteristics = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                # Safe block extraction
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                rgb_block = image_array[y_start:y_end, x_start:x_end]
                lab_block = lab[y_start:y_end, x_start:x_end]
                
                # Noise estimation using Laplacian variance
                gray_block = cv2.cvtColor(rgb_block, cv2.COLOR_RGB2GRAY)
                laplacian_var = cv2.Laplacian(gray_block, cv2.CV_64F).var()
                
                # High frequency content analysis with safe indexing
                try:
                    f_transform = np.fft.fft2(gray_block)
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
                    
                    # Safe frequency range calculation
                    h_block, w_block = magnitude_spectrum.shape
                    quarter_h, quarter_w = max(1, h_block//4), max(1, w_block//4)
                    three_quarter_h = min(h_block, 3*h_block//4)
                    three_quarter_w = min(w_block, 3*w_block//4)
                    
                    if three_quarter_h > quarter_h and three_quarter_w > quarter_w:
                        high_freq_energy = np.sum(magnitude_spectrum[quarter_h:three_quarter_h, 
                                                                   quarter_w:three_quarter_w])
                    else:
                        high_freq_energy = np.sum(magnitude_spectrum)
                except Exception:
                    high_freq_energy = 0.0
                
                # Color noise analysis
                rgb_std = np.std(rgb_block, axis=(0, 1))
                lab_std = np.std(lab_block, axis=(0, 1))
                
                # Statistical moments
                mean_intensity = np.mean(gray_block)
                std_intensity = np.std(gray_block)
                skewness = calculate_skewness(gray_block.flatten())
                kurtosis = calculate_kurtosis(gray_block.flatten())
                
                noise_characteristics.append({
                    'position': (i, j),
                    'laplacian_var': float(laplacian_var),
                    'high_freq_energy': float(high_freq_energy),
                    'rgb_std': rgb_std.tolist(),
                    'lab_std': lab_std.tolist(),
                    'mean_intensity': float(mean_intensity),
                    'std_intensity': float(std_intensity),
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis)
                })
        
        # Analyze consistency across blocks
        if noise_characteristics:
            laplacian_vars = [block['laplacian_var'] for block in noise_characteristics]
            high_freq_energies = [block['high_freq_energy'] for block in noise_characteristics]
            std_intensities = [block['std_intensity'] for block in noise_characteristics]
            
            # Calculate consistency metrics
            laplacian_consistency = np.std(laplacian_vars) / (np.mean(laplacian_vars) + 1e-6)
            freq_consistency = np.std(high_freq_energies) / (np.mean(high_freq_energies) + 1e-6)
            intensity_consistency = np.std(std_intensities) / (np.mean(std_intensities) + 1e-6)
            
            # Overall inconsistency score
            overall_inconsistency = (laplacian_consistency + freq_consistency + intensity_consistency) / 3
            
            # Detect outlier blocks with error handling
            outliers = []
            try:
                outlier_indices = detect_outliers_iqr(np.array(laplacian_vars))
                for idx in outlier_indices:
                    if idx < len(noise_characteristics):
                        outliers.append(noise_characteristics[idx])
            except Exception:
                pass
        else:
            laplacian_consistency = 0.0
            freq_consistency = 0.0
            intensity_consistency = 0.0
            overall_inconsistency = 0.0
            outliers = []
        
        return {
            'noise_characteristics': noise_characteristics,
            'laplacian_consistency': float(laplacian_consistency),
            'frequency_consistency': float(freq_consistency),
            'intensity_consistency': float(intensity_consistency),
            'overall_inconsistency': float(overall_inconsistency),
            'outlier_blocks': outliers,
            'outlier_count': len(outliers)
        }
        
    except Exception as e:
        print(f"  Warning: Noise analysis failed: {e}")
        return {
            'noise_characteristics': [],
            'laplacian_consistency': 0.0,
            'frequency_consistency': 0.0,
            'intensity_consistency': 0.0,
            'overall_inconsistency': 0.0,
            'outlier_blocks': [],
            'outlier_count': 0
        }

# ======================= Frequency Domain Analysis =======================

def analyze_frequency_domain(image_pil):
    """Analyze DCT coefficients for manipulation detection"""
    try:
        image_array = np.array(image_pil.convert('L'))
        
        # DCT Analysis dengan multiple fallback methods
        dct_coeffs = None
        
        # Method 1: OpenCV DCT
        try:
            dct_coeffs = cv2.dct(image_array.astype(np.float32))
        except Exception:
            pass
        
        # Method 2: SciPy DCT fallback
        if dct_coeffs is None:
            try:
                from scipy.fft import dctn
                dct_coeffs = dctn(image_array, type=2, norm='ortho')
            except Exception:
                pass
        
        # Method 3: NumPy FFT fallback
        if dct_coeffs is None:
            try:
                dct_coeffs = np.abs(np.fft.fft2(image_array))
            except Exception:
                dct_coeffs = np.zeros_like(image_array, dtype=np.float32)
        
        h, w = dct_coeffs.shape
        
        # Safe region calculation
        low_h, low_w = min(16, h), min(16, w)
        
        dct_stats = {
            'low_freq_energy': float(np.sum(np.abs(dct_coeffs[:low_h, :low_w]))),
            'high_freq_energy': float(np.sum(np.abs(dct_coeffs[low_h:, low_w:]))),
            'mid_freq_energy': float(np.sum(np.abs(dct_coeffs[8:min(24,h), 8:min(24,w)]))),
        }
        
        dct_stats['freq_ratio'] = dct_stats['high_freq_energy'] / (dct_stats['low_freq_energy'] + 1e-6)
        
        # Block-wise DCT analysis
        block_size = 8
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        block_freq_variations = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                block = image_array[y_start:y_end, x_start:x_end]
                
                try:
                    block_dct = cv2.dct(block.astype(np.float32))
                    block_energy = np.sum(np.abs(block_dct))
                except Exception:
                    block_energy = np.sum(np.abs(block))
                
                block_freq_variations.append(float(block_energy))
        
        # Calculate frequency inconsistency
        if len(block_freq_variations) > 0:
            freq_inconsistency = np.std(block_freq_variations) / (np.mean(block_freq_variations) + 1e-6)
        else:
            freq_inconsistency = 0.0
        
        return {
            'dct_stats': dct_stats,
            'frequency_inconsistency': float(freq_inconsistency),
            'block_variations': float(np.var(block_freq_variations)) if block_freq_variations else 0.0
        }
        
    except Exception as e:
        print(f"  Warning: Frequency analysis failed: {e}")
        return {
            'dct_stats': {
                'low_freq_energy': 0.0,
                'high_freq_energy': 0.0,
                'mid_freq_energy': 0.0,
                'freq_ratio': 0.0
            },
            'frequency_inconsistency': 0.0,
            'block_variations': 0.0
        }

# ======================= Texture Analysis =======================

def analyze_texture_consistency(image_pil, block_size=64):
    """Analyze texture consistency using GLCM and LBP"""
    try:
        image_gray = np.array(image_pil.convert('L'))
        
        # Local Binary Pattern analysis dengan fallback
        if SKIMAGE_AVAILABLE:
            try:
                radius = 3
                n_points = 8 * radius
                lbp = local_binary_pattern(image_gray, n_points, radius, method='uniform')
            except Exception:
                lbp = np.zeros_like(image_gray)
        else:
            lbp = np.zeros_like(image_gray)
        
        # Block-wise texture analysis
        h, w = image_gray.shape
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        texture_features = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                block = image_gray[y_start:y_end, x_start:x_end]
                
                # GLCM analysis dengan fallback
                if SKIMAGE_AVAILABLE:
                    try:
                        glcm = graycomatrix(block, distances=[1], angles=[0, 45, 90, 135],
                                          levels=256, symmetric=True, normed=True)
                        
                        contrast = graycoprops(glcm, 'contrast')[0, 0]
                        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                        energy = graycoprops(glcm, 'energy')[0, 0]
                    except Exception:
                        # Fallback measures
                        contrast = float(np.var(block))
                        dissimilarity = float(np.std(block))
                        homogeneity = 1.0 / (1.0 + np.var(block))
                        energy = float(np.mean(block ** 2) / 255**2)
                else:
                    # Fallback measures
                    contrast = float(np.var(block))
                    dissimilarity = float(np.std(block))
                    homogeneity = 1.0 / (1.0 + np.var(block))
                    energy = float(np.mean(block ** 2) / 255**2)
                
                # LBP calculation
                lbp_uniformity = safe_entropy(block)
                
                texture_features.append([
                    float(contrast), 
                    float(dissimilarity), 
                    float(homogeneity), 
                    float(energy), 
                    float(lbp_uniformity)
                ])
        
        # Analyze consistency
        texture_consistency = {}
        feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'lbp_uniformity']
        
        if len(texture_features) > 0:
            texture_features = np.array(texture_features)
            for i, name in enumerate(feature_names):
                feature_values = texture_features[:, i]
                consistency = np.std(feature_values) / (np.mean(feature_values) + 1e-6)
                texture_consistency[f'{name}_consistency'] = float(consistency)
            
            overall_texture_inconsistency = np.mean(list(texture_consistency.values()))
        else:
            for name in feature_names:
                texture_consistency[f'{name}_consistency'] = 0.0
            overall_texture_inconsistency = 0.0
        
        return {
            'texture_consistency': texture_consistency,
            'overall_inconsistency': float(overall_texture_inconsistency),
            'texture_features': texture_features.tolist() if len(texture_features) > 0 else []
        }
        
    except Exception as e:
        print(f"  Warning: Texture analysis failed: {e}")
        feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'lbp_uniformity']
        texture_consistency = {f'{name}_consistency': 0.0 for name in feature_names}
        
        return {
            'texture_consistency': texture_consistency,
            'overall_inconsistency': 0.0,
            'texture_features': []
        }

# ======================= Edge Analysis =======================

def analyze_edge_consistency(image_pil):
    """Analyze edge density consistency"""
    try:
        image_gray = np.array(image_pil.convert('L'))
        
        # Multiple edge detectors dengan fallback
        if SKIMAGE_AVAILABLE:
            try:
                edges_sobel = sobel(image_gray)
                edges_prewitt = prewitt(image_gray)
                edges_roberts = roberts(image_gray)
                combined_edges = (edges_sobel + edges_prewitt + edges_roberts) / 3
            except Exception:
                # Fallback to OpenCV Sobel
                grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
                combined_edges = np.sqrt(grad_x**2 + grad_y**2)
        else:
            # Fallback to OpenCV Sobel
            grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
            combined_edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # Block-wise edge density
        block_size = 32
        h, w = image_gray.shape
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        edge_densities = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                block_edges = combined_edges[y_start:y_end, x_start:x_end]
                edge_density = np.mean(block_edges)
                edge_densities.append(float(edge_density))
        
        edge_densities = np.array(edge_densities)
        
        if len(edge_densities) > 0:
            edge_inconsistency = np.std(edge_densities) / (np.mean(edge_densities) + 1e-6)
            edge_variance = np.var(edge_densities)
        else:
            edge_inconsistency = 0.0
            edge_variance = 0.0
        
        return {
            'edge_inconsistency': float(edge_inconsistency),
            'edge_densities': edge_densities.tolist(),
            'edge_variance': float(edge_variance)
        }
        
    except Exception as e:
        print(f"  Warning: Edge analysis failed: {e}")
        return {
            'edge_inconsistency': 0.0,
            'edge_densities': [],
            'edge_variance': 0.0
        }

# ======================= Illumination Analysis =======================

def analyze_illumination_consistency(image_pil):
    """Advanced illumination consistency analysis"""
    try:
        image_array = np.array(image_pil)
        
        # Convert to different color spaces
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        
        # Illumination map (L channel in LAB)
        illumination = lab[:, :, 0]
        
        # Gradient analysis
        grad_x = cv2.Sobel(illumination, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(illumination, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Block-wise illumination analysis
        block_size = 64
        h, w = illumination.shape
        blocks_h = max(1, h // block_size)
        blocks_w = max(1, w // block_size)
        
        illumination_means = []
        illumination_stds = []
        gradient_means = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                y_start, y_end = i*block_size, min((i+1)*block_size, h)
                x_start, x_end = j*block_size, min((j+1)*block_size, w)
                
                block_illum = illumination[y_start:y_end, x_start:x_end]
                block_grad = gradient_magnitude[y_start:y_end, x_start:x_end]
                
                illumination_means.append(np.mean(block_illum))
                illumination_stds.append(np.std(block_illum))
                gradient_means.append(np.mean(block_grad))
        
        # Consistency metrics
        if len(illumination_means) > 0:
            illum_mean_consistency = np.std(illumination_means) / (np.mean(illumination_means) + 1e-6)
            illum_std_consistency = np.std(illumination_stds) / (np.mean(illumination_stds) + 1e-6)
            gradient_consistency = np.std(gradient_means) / (np.mean(gradient_means) + 1e-6)
        else:
            illum_mean_consistency = 0.0
            illum_std_consistency = 0.0
            gradient_consistency = 0.0
        
        return {
            'illumination_mean_consistency': float(illum_mean_consistency),
            'illumination_std_consistency': float(illum_std_consistency),
            'gradient_consistency': float(gradient_consistency),
            'overall_illumination_inconsistency': float((illum_mean_consistency + gradient_consistency) / 2)
        }
        
    except Exception as e:
        print(f"  Warning: Illumination analysis failed: {e}")
        return {
            'illumination_mean_consistency': 0.0,
            'illumination_std_consistency': 0.0,
            'gradient_consistency': 0.0,
            'overall_illumination_inconsistency': 0.0
        }

# ======================= Statistical Analysis =======================

def perform_statistical_analysis(image_pil):
    """Comprehensive statistical analysis"""
    try:
        image_array = np.array(image_pil)
        stats = {}
        
        # Per-channel statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image_array[:, :, i].flatten()
            stats[f'{channel}_mean'] = float(np.mean(channel_data))
            stats[f'{channel}_std'] = float(np.std(channel_data))
            stats[f'{channel}_skewness'] = float(calculate_skewness(channel_data))
            stats[f'{channel}_kurtosis'] = float(calculate_kurtosis(channel_data))
            stats[f'{channel}_entropy'] = float(safe_entropy(image_array[:, :, i]))
        
        # Cross-channel correlation
        r_channel = image_array[:, :, 0].flatten()
        g_channel = image_array[:, :, 1].flatten()
        b_channel = image_array[:, :, 2].flatten()
        
        stats['rg_correlation'] = float(np.corrcoef(r_channel, g_channel)[0, 1])
        stats['rb_correlation'] = float(np.corrcoef(r_channel, b_channel)[0, 1])
        stats['gb_correlation'] = float(np.corrcoef(g_channel, b_channel)[0, 1])
        
        # Overall statistics
        stats['overall_entropy'] = float(safe_entropy(image_array))
        
        return stats
        
    except Exception as e:
        print(f"  Warning: Statistical analysis failed: {e}")
        # Return safe defaults
        channels = ['R', 'G', 'B']
        stats = {}
        for ch in channels:
            stats[f'{ch}_mean'] = 0.0
            stats[f'{ch}_std'] = 0.0
            stats[f'{ch}_skewness'] = 0.0
            stats[f'{ch}_kurtosis'] = 0.0
            stats[f'{ch}_entropy'] = 0.0
        
        stats['rg_correlation'] = 0.0
        stats['rb_correlation'] = 0.0
        stats['gb_correlation'] = 0.0
        stats['overall_entropy'] = 0.0
        
        return stats
