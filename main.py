# --- START OF FILE main.py ---

#!/usr/bin/env python3
"""
Advanced Forensic Image Analysis System v2.0
Main execution file

Usage:
    python main.py <image_path> [options]

Example:
    python main.py test_image.jpg
    python main.py test_image.jpg --export-all
    python main.py test_image.jpg --output-dir ./results
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
from PIL import Image
from datetime import datetime # PERBAIKAN: Tambahkan import datetime

# PERBAIKAN: Import fungsi save_analysis_to_history dari utils
from utils import save_analysis_to_history

# Import semua modul
from validation import validate_image_file, extract_enhanced_metadata, advanced_preprocess_image
from ela_analysis import perform_multi_quality_ela
from feature_detection import extract_multi_detector_features
from copy_move_detection import detect_copy_move_advanced, detect_copy_move_blocks, kmeans_tampering_localization
from advanced_analysis import (analyze_noise_consistency, analyze_frequency_domain,
                              analyze_texture_consistency, analyze_edge_consistency,
                              analyze_illumination_consistency, perform_statistical_analysis)
from jpeg_analysis import advanced_jpeg_analysis, jpeg_ghost_analysis
from classification import classify_manipulation_advanced, prepare_feature_vector
from visualization import visualize_results_advanced, export_kmeans_visualization
from export_utils import export_complete_package


# ======================= FUNGSI BARU UNTUK MEMPERBAIKI LOKALISASI =======================
def advanced_tampering_localization(image_pil, analysis_results):
    """
    Advanced tampering localization menggunakan multiple methods.
    Fungsi ini menggabungkan beberapa hasil untuk membuat masker deteksi yang lebih andal.
    """
    print("  -> Combining multiple localization methods...")
    
    # Ambil data yang diperlukan dari hasil analisis
    ela_image = analysis_results.get('ela_image')
    if ela_image is None:
        return {'tampering_percentage': 0, 'combined_tampering_mask': np.zeros(image_pil.size, dtype=bool)}

    # 1. K-means based localization (hasil dari copy_move_detection)
    kmeans_result = kmeans_tampering_localization(image_pil, ela_image)
    kmeans_mask = kmeans_result.get('tampering_mask', np.zeros(ela_image.size, dtype=bool))

    # 2. Threshold-based localization from ELA
    ela_array = np.array(ela_image)
    ela_mean = analysis_results.get('ela_mean', 0)
    ela_std = analysis_results.get('ela_std', 1) # Gunakan 1 jika tidak ada untuk menghindari div by zero
    threshold = ela_mean + 2 * ela_std
    threshold_mask = ela_array > threshold

    # 3. Combined localization mask
    # Menggabungkan masker dari K-Means dan ELA thresholding
    combined_mask = np.logical_or(kmeans_mask, threshold_mask)

    # 4. Morphological operations untuk membersihkan masker
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # Dapatkan ukuran gambar dari PIL Image object
    w, h = image_pil.size

    return {
        'kmeans_localization': kmeans_result,
        'threshold_mask': threshold_mask,
        'combined_tampering_mask': cleaned_mask.astype(bool), # Ini adalah kunci yang hilang
        'tampering_percentage': np.sum(cleaned_mask) / (h * w) * 100
    }
# ======================= AKHIR FUNGSI BARU =======================


# Bagian yang perlu dimodifikasi di main.py untuk tracking status pipeline

def analyze_image_comprehensive_advanced(image_path, output_dir="./results"):
    """Advanced comprehensive image analysis pipeline with status tracking"""
    print(f"\n{'='*80}")
    print(f"ADVANCED FORENSIC IMAGE ANALYSIS SYSTEM v2.0")
    print(f"Enhanced Detection: Copy-Move, Splicing, Authentic Images")
    print(f"{'='*80}\n")

    start_time = time.time()
    
    # Initialize pipeline status tracking
    pipeline_status = {
        'total_stages': 17,
        'completed_stages': 0,
        'failed_stages': [],
        'stage_details': {}
    }

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Validation
    try:
        validate_image_file(image_path)
        print("✅ [1/17] File validation passed")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['validation'] = True
    except Exception as e:
        print(f"❌ [1/17] Validation error: {e}")
        pipeline_status['failed_stages'].append('validation')
        pipeline_status['stage_details']['validation'] = False
        return None

    # 2. Load image
    try:
        original_image = Image.open(image_path)
        print(f"✅ [2/17] Image loaded: {os.path.basename(image_path)}")
        print(f"  Size: {original_image.size}, Mode: {original_image.mode}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['image_loading'] = True
    except Exception as e:
        print(f"❌ [2/17] Error loading image: {e}")
        pipeline_status['failed_stages'].append('image_loading')
        pipeline_status['stage_details']['image_loading'] = False
        return None

    # 3. Enhanced metadata extraction
    print("🔍 [3/17] Extracting enhanced metadata...")
    try:
        metadata = extract_enhanced_metadata(image_path)
        print(f"  Authenticity Score: {metadata['Metadata_Authenticity_Score']}/100")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['metadata_extraction'] = True
    except Exception as e:
        print(f"❌ [3/17] Metadata extraction failed: {e}")
        metadata = {'Metadata_Authenticity_Score': 0, 'Filename': os.path.basename(image_path)}
        pipeline_status['failed_stages'].append('metadata_extraction')
        pipeline_status['stage_details']['metadata_extraction'] = False

    # 4. Advanced preprocessing
    print("🔧 [4/17] Advanced preprocessing...")
    try:
        preprocessed, original_preprocessed = advanced_preprocess_image(original_image.copy())
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['preprocessing'] = True
    except Exception as e:
        print(f"❌ [4/17] Preprocessing failed: {e}")
        preprocessed = original_image.copy()
        original_preprocessed = original_image.copy()
        pipeline_status['failed_stages'].append('preprocessing')
        pipeline_status['stage_details']['preprocessing'] = False

    # 5. Multi-quality ELA
    print("📊 [5/17] Multi-quality Error Level Analysis...")
    try:
        ela_image, ela_mean, ela_std, ela_regional, ela_quality_stats, ela_variance = perform_multi_quality_ela(preprocessed.copy())
        print(f"  ELA Stats: μ={ela_mean:.2f}, σ={ela_std:.2f}, Regions={ela_regional['outlier_regions']}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['ela_analysis'] = True
    except Exception as e:
        print(f"❌ [5/17] ELA analysis failed: {e}")
        ela_image = Image.new('L', preprocessed.size)
        ela_mean = ela_std = 0
        ela_regional = {'outlier_regions': 0, 'regional_inconsistency': 0, 'suspicious_regions': []}
        ela_quality_stats = []
        ela_variance = np.zeros(preprocessed.size)
        pipeline_status['failed_stages'].append('ela_analysis')
        pipeline_status['stage_details']['ela_analysis'] = False

    # 6. Multi-detector feature extraction
    print("🎯 [6/17] Multi-detector feature extraction...")
    try:
        feature_sets, roi_mask, gray_enhanced = extract_multi_detector_features(
            preprocessed.copy(), ela_image, ela_mean, ela_std)
        total_features = sum(len(kp) for kp, _ in feature_sets.values())
        print(f"  Total keypoints: {total_features}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['feature_extraction'] = True
    except Exception as e:
        print(f"❌ [6/17] Feature extraction failed: {e}")
        feature_sets = {'sift': ([], None), 'orb': ([], None), 'akaze': ([], None)}
        roi_mask = np.ones(preprocessed.size, dtype=np.uint8)
        gray_enhanced = np.array(preprocessed.convert('L'))
        pipeline_status['failed_stages'].append('feature_extraction')
        pipeline_status['stage_details']['feature_extraction'] = False

    # 7. Advanced copy-move detection
    print("🔄 [7/17] Advanced copy-move detection...")
    try:
        ransac_matches, ransac_inliers, transform = detect_copy_move_advanced(
            feature_sets, preprocessed.size)
        print(f"  RANSAC inliers: {ransac_inliers}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['copymove_detection'] = True
    except Exception as e:
        print(f"❌ [7/17] Copy-move detection failed: {e}")
        ransac_matches = []
        ransac_inliers = 0
        transform = None
        pipeline_status['failed_stages'].append('copymove_detection')
        pipeline_status['stage_details']['copymove_detection'] = False

    # 8. Enhanced block matching
    print("🧩 [8/17] Enhanced block-based detection...")
    try:
        block_matches = detect_copy_move_blocks(preprocessed)
        print(f"  Block matches: {len(block_matches)}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['block_matching'] = True
    except Exception as e:
        print(f"❌ [8/17] Block matching failed: {e}")
        block_matches = []
        pipeline_status['failed_stages'].append('block_matching')
        pipeline_status['stage_details']['block_matching'] = False

    # 9. Advanced noise analysis
    print("📡 [9/17] Advanced noise consistency analysis...")
    try:
        noise_analysis = analyze_noise_consistency(preprocessed)
        print(f"  Noise inconsistency: {noise_analysis['overall_inconsistency']:.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['noise_analysis'] = True
    except Exception as e:
        print(f"❌ [9/17] Noise analysis failed: {e}")
        noise_analysis = {'overall_inconsistency': 0, 'outlier_count': 0}
        pipeline_status['failed_stages'].append('noise_analysis')
        pipeline_status['stage_details']['noise_analysis'] = False

    # 10. Advanced JPEG analysis
    print("📷 [10/17] Advanced JPEG artifact analysis...")
    try:
        from jpeg_analysis import advanced_jpeg_analysis, jpeg_ghost_analysis
        jpeg_analysis = advanced_jpeg_analysis(preprocessed)

        # Robust handling untuk return values dari jpeg_ghost_analysis
        jpeg_ghost_result = jpeg_ghost_analysis(preprocessed)

        if len(jpeg_ghost_result) == 2:
            ghost_map, ghost_suspicious = jpeg_ghost_result
            ghost_analysis_details = {}
        elif len(jpeg_ghost_result) == 3:
            ghost_map, ghost_suspicious, ghost_analysis_details = jpeg_ghost_result
        else:
            raise ValueError(f"Unexpected return values from jpeg_ghost_analysis: {len(jpeg_ghost_result)}")

        ghost_ratio = np.sum(ghost_suspicious) / ghost_suspicious.size if ghost_suspicious.size > 0 else 0
        print(f"  JPEG anomalies: {ghost_ratio:.1%}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['jpeg_analysis'] = True

    except Exception as e:
        print(f"❌ [10/17] JPEG analysis failed: {e}")
        # Fallback values
        jpeg_analysis = {
            'quality_responses': [],
            'response_variance': 0.0,
            'double_compression_indicator': 0.0,
            'estimated_original_quality': 0,
            'compression_inconsistency': False
        }
        ghost_map = np.zeros((preprocessed.size[1], preprocessed.size[0]))
        ghost_suspicious = np.zeros((preprocessed.size[1], preprocessed.size[0]), dtype=bool)
        ghost_analysis_details = {}
        ghost_ratio = 0.0
        pipeline_status['failed_stages'].append('jpeg_analysis')
        pipeline_status['stage_details']['jpeg_analysis'] = False

    # 11. Frequency domain analysis
    print("🌊 [11/17] Frequency domain analysis...")
    try:
        frequency_analysis = analyze_frequency_domain(preprocessed)
        print(f"  Frequency inconsistency: {frequency_analysis['frequency_inconsistency']:.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['frequency_analysis'] = True
    except Exception as e:
        print(f"❌ [11/17] Frequency analysis failed: {e}")
        frequency_analysis = {'frequency_inconsistency': 0, 'dct_stats': {}}
        pipeline_status['failed_stages'].append('frequency_analysis')
        pipeline_status['stage_details']['frequency_analysis'] = False

    # 12. Texture consistency analysis
    print("🧵 [12/17] Texture consistency analysis...")
    try:
        texture_analysis = analyze_texture_consistency(preprocessed)
        print(f"  Texture inconsistency: {texture_analysis['overall_inconsistency']:.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['texture_analysis'] = True
    except Exception as e:
        print(f"❌ [12/17] Texture analysis failed: {e}")
        texture_analysis = {'overall_inconsistency': 0}
        pipeline_status['failed_stages'].append('texture_analysis')
        pipeline_status['stage_details']['texture_analysis'] = False

    # 13. Edge consistency analysis
    print("📐 [13/17] Edge density analysis...")
    try:
        edge_analysis = analyze_edge_consistency(preprocessed)
        print(f"  Edge inconsistency: {edge_analysis['edge_inconsistency']:.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['edge_analysis'] = True
    except Exception as e:
        print(f"❌ [13/17] Edge analysis failed: {e}")
        edge_analysis = {'edge_inconsistency': 0}
        pipeline_status['failed_stages'].append('edge_analysis')
        pipeline_status['stage_details']['edge_analysis'] = False

    # 14. Illumination analysis
    print("💡 [14/17] Illumination consistency analysis...")
    try:
        illumination_analysis = analyze_illumination_consistency(preprocessed)
        print(f"  Illumination inconsistency: {illumination_analysis['overall_illumination_inconsistency']:.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['illumination_analysis'] = True
    except Exception as e:
        print(f"❌ [14/17] Illumination analysis failed: {e}")
        illumination_analysis = {'overall_illumination_inconsistency': 0}
        pipeline_status['failed_stages'].append('illumination_analysis')
        pipeline_status['stage_details']['illumination_analysis'] = False

    # 15. Statistical analysis
    print("📈 [15/17] Statistical analysis...")
    try:
        statistical_analysis = perform_statistical_analysis(preprocessed)
        print(f"  Overall entropy: {statistical_analysis['overall_entropy']:.3f}")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['statistical_analysis'] = True
    except Exception as e:
        print(f"❌ [15/17] Statistical analysis failed: {e}")
        statistical_analysis = {'overall_entropy': 0, 'rg_correlation': 0}
        pipeline_status['failed_stages'].append('statistical_analysis')
        pipeline_status['stage_details']['statistical_analysis'] = False

    # Prepare preliminary results for localization
    analysis_results = {
        'metadata': metadata,
        'ela_image': ela_image,
        'ela_mean': ela_mean,
        'ela_std': ela_std,
        'ela_regional_stats': ela_regional,
        'ela_quality_stats': ela_quality_stats,
        'ela_variance': ela_variance,
        'feature_sets': feature_sets,
        'sift_keypoints': feature_sets['sift'][0],
        'sift_descriptors': feature_sets['sift'][1],
        'sift_matches': len(ransac_matches),
        'ransac_matches': ransac_matches,
        'ransac_inliers': ransac_inliers,
        'geometric_transform': transform,
        'block_matches': block_matches,
        'noise_analysis': noise_analysis,
        'noise_map': cv2.cvtColor(np.array(preprocessed), cv2.COLOR_RGB2GRAY),
        'jpeg_analysis': jpeg_analysis,
        'jpeg_ghost': ghost_map,
        'jpeg_ghost_suspicious_ratio': ghost_ratio,
        'frequency_analysis': frequency_analysis,
        'texture_analysis': texture_analysis,
        'edge_analysis': edge_analysis,
        'illumination_analysis': illumination_analysis,
        'statistical_analysis': statistical_analysis,
        'color_analysis': {'illumination_inconsistency': illumination_analysis['overall_illumination_inconsistency']},
        'roi_mask': roi_mask,
        'enhanced_gray': gray_enhanced,
        'pipeline_status': pipeline_status  # Add pipeline status to results
    }

    # 16. Advanced tampering localization
    print("🎯 [16/17] Advanced tampering localization...")
    try:
        localization_results = advanced_tampering_localization(preprocessed, analysis_results)
        analysis_results['localization_analysis'] = localization_results
        print(f"  Tampering area: {localization_results.get('tampering_percentage', 0):.1f}% of image")
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['localization_analysis'] = True
    except Exception as e:
        print(f"❌ [16/17] Localization analysis failed: {e}")
        analysis_results['localization_analysis'] = {'tampering_percentage': 0, 'combined_tampering_mask': np.zeros(preprocessed.size, dtype=bool)}
        pipeline_status['failed_stages'].append('localization_analysis')
        pipeline_status['stage_details']['localization_analysis'] = False
    
    # 17. Advanced classification
    print("🤖 [17/17] Advanced manipulation classification...")
    try:
        classification = classify_manipulation_advanced(analysis_results)
        analysis_results['classification'] = classification
        pipeline_status['completed_stages'] += 1
        pipeline_status['stage_details']['classification'] = True
    except Exception as e:
        print(f"❌ [17/17] Classification failed: {e}")
        classification = {
            'type': 'Analysis Error',
            'confidence': 'Error',
            'copy_move_score': 0,
            'splicing_score': 0,
            'details': [f"Classification error: {str(e)}"]
        }
        analysis_results['classification'] = classification
        pipeline_status['failed_stages'].append('classification')
        pipeline_status['stage_details']['classification'] = False

    # Update final pipeline status
    analysis_results['pipeline_status'] = pipeline_status
    
    processing_time = time.time() - start_time

    # Print pipeline summary
    print(f"\n{'='*80}")
    print(f"PIPELINE STATUS SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Total Stages: {pipeline_status['total_stages']}")
    print(f"📊 Completed Successfully: {pipeline_status['completed_stages']}")
    print(f"📊 Failed Stages: {len(pipeline_status['failed_stages'])}")
    print(f"📊 Success Rate: {(pipeline_status['completed_stages']/pipeline_status['total_stages']*100):.1f}%")
    
    if pipeline_status['failed_stages']:
        print(f"📊 Failed Components: {', '.join(pipeline_status['failed_stages'])}")

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE - Processing Time: {processing_time:.2f}s")
    print(f"{'='*80}")
    print(f"📊 FINAL RESULT: {classification['type']}")
    print(f"📊 CONFIDENCE: {classification['confidence']}")
    print(f"📊 Copy-Move Score: {classification['copy_move_score']}/100")
    print(f"📊 Splicing Score: {classification['splicing_score']}/100")
    print(f"{'='*80}\n")

    if classification['details']:
        print("📋 Detection Details:")
        for detail in classification['details']:
            print(f"  {detail}")
        print()

    # Save to history (existing code)
    try:
        image_filename = os.path.basename(image_path)
        analysis_summary_for_history = {
            'type': classification.get('type', 'N/A'),
            'confidence': classification.get('confidence', 'N/A'),
            'copy_move_score': classification.get('copy_move_score', 0),
            'splicing_score': classification.get('splicing_score', 0)
        }
        thumbnail_dir = "history_thumbnails"
        os.makedirs(thumbnail_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        thumbnail_filename = f"thumb_{timestamp_str}.jpg"
        thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
        with Image.open(image_path) as img:
            img.thumbnail((128, 128))
            img.convert("RGB").save(thumbnail_path, "JPEG", quality=85)
        save_analysis_to_history(
            image_filename, 
            analysis_summary_for_history, 
            f"{processing_time:.2f}s",
            thumbnail_path
        )
        print(f"💾 Analysis results and thumbnail saved to history ({thumbnail_path}).")
    except Exception as e:
        print(f"⚠️ Failed to save analysis to history: {e}")

    return analysis_results

def main():
    parser = argparse.ArgumentParser(description='Advanced Forensic Image Analysis System v2.0')
    parser.add_argument('image_path', help='Path to the image file to analyze')
    parser.add_argument('--output-dir', '-o', default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--export-all', '-e', action='store_true',
                       help='Export complete package (PNG, PDF, DOCX, etc.)')
    parser.add_argument('--export-vis', '-v', action='store_true',
                       help='Export only visualization')
    parser.add_argument('--export-report', '-r', action='store_true',
                       help='Export only DOCX report')

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file '{args.image_path}' not found!")
        sys.exit(1)

    try:
        analysis_results = analyze_image_comprehensive_advanced(args.image_path, args.output_dir)

        if analysis_results is None:
            print("❌ Analysis failed!")
            sys.exit(1)

        original_image = Image.open(args.image_path)
        base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
        base_path = os.path.join(args.output_dir, base_filename)

        if args.export_all:
            print("\n📦 Exporting complete package...")
            export_complete_package(original_image, analysis_results, base_path)
        elif args.export_vis:
            print("\n📊 Exporting visualization...")
            visualize_results_advanced(original_image, analysis_results, f"{base_path}_analysis.png")
        elif args.export_report:
            print("\n📄 Exporting DOCX report...")
            from export_utils import export_to_advanced_docx
            export_to_advanced_docx(original_image, analysis_results, f"{base_path}_report.docx")
        else:
            print("\n📊 Exporting basic visualization...")
            visualize_results_advanced(original_image, analysis_results, f"{base_path}_analysis.png")

        print("✅ Analysis completed successfully!")

    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()