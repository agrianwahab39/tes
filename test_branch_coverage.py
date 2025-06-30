# test_branch_coverage.py (VERSI PERBAIKAN)

import unittest
import os
import json
import shutil
from unittest.mock import MagicMock, patch
from PIL import Image # <-- SOLUSI MASALAH 3: Penambahan impor dependensi

# Impor fungsi-fungsi yang akan diuji
from utils import load_analysis_history, save_analysis_to_history, delete_all_history, delete_selected_history, get_history_count, clear_empty_thumbnail_folder
from classification import classify_manipulation_advanced
try:
    from app2 import ForensicValidator
    APP2_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Could not import app2.py: {e}. ForensicValidator tests will be skipped.")
    APP2_AVAILABLE = False
    class ForensicValidator: # Dummy class
        def validate_cross_algorithm(self, results): return [], 0.0, "", []


# Helper untuk membuat data mock
def create_mock_analysis_results(
    copy_move_score=0, splicing_score=0, ransac_inliers=0, block_matches=0,
    noise_inconsistency=0, jpeg_suspicious=0, ela_mean=0, metadata_score=100,
    tampering_percentage=0.0):
    """Membuat dictionary analysis_results palsu yang lengkap untuk pengujian."""
    # SOLUSI MASALAH 2: Melengkapi data uji
    return {
        'ela_mean': ela_mean, 'ela_std': 10, 'ela_regional_stats': {'mean_variance': 0.1, 'regional_inconsistency': 0.1, 'outlier_regions': 2, 'suspicious_regions': []},
        'sift_matches': 100, 'ransac_inliers': ransac_inliers, 'geometric_transform': ['affine'] if ransac_inliers > 5 else None,
        'block_matches': [{} for _ in range(block_matches)],
        'noise_analysis': {'overall_inconsistency': noise_inconsistency, 'outlier_count': 1},
        'jpeg_ghost_suspicious_ratio': jpeg_suspicious,
        'jpeg_analysis': {'response_variance': 0.1, 'double_compression_indicator': 0.1, 'compression_inconsistency': False},
        'frequency_analysis': {'frequency_inconsistency': 0.1, 'dct_stats': {'freq_ratio': 0.1}},
        'texture_analysis': {'overall_inconsistency': 0.1}, 'edge_analysis': {'edge_inconsistency': 0.1},
        'illumination_analysis': {'overall_illumination_inconsistency': 0.1},
        'statistical_analysis': {
            'R_entropy': 7, 'G_entropy': 7, 'B_entropy': 7,
            'rg_correlation': 0.95, 'rb_correlation': 0.92, 'gb_correlation': 0.96,
            'overall_entropy': 7.5
        },
        'metadata': {'Metadata_Authenticity_Score': metadata_score, 'Metadata_Inconsistency': []},
        'localization_analysis': {'tampering_percentage': tampering_percentage, 'kmeans_localization': {'cluster_ela_means': [5, 15]}}
    }

class TestForensicSystemBranchCoverage(unittest.TestCase):

    def setUp(self):
        self.history_file = 'analysis_history.json'
        self.thumbnail_dir = "history_thumbnails"
        if os.path.exists(self.history_file): os.remove(self.history_file)
        if os.path.exists(self.thumbnail_dir): shutil.rmtree(self.thumbnail_dir)
        os.makedirs(self.thumbnail_dir, exist_ok=True)
        self.thumb_path1 = os.path.join(self.thumbnail_dir, "thumb1.jpg")
        self.thumb_path2 = os.path.join(self.thumbnail_dir, "thumb2.jpg")
        with open(self.thumb_path1, 'w') as f: f.write('dummy')
        with open(self.thumb_path2, 'w') as f: f.write('dummy')

    def tearDown(self):
        if os.path.exists(self.history_file): os.remove(self.history_file)
        if os.path.exists(self.thumbnail_dir): shutil.rmtree(self.thumbnail_dir)

    # --- Pengujian untuk utils.py ---

    def test_load_history_no_file(self):
        self.assertEqual(load_analysis_history(), [])

    def test_load_history_empty_json(self):
        with open(self.history_file, 'w') as f: f.write('[]')
        self.assertEqual(load_analysis_history(), [])

    def test_load_history_invalid_json(self):
        with open(self.history_file, 'w') as f: f.write('not a json string')
        self.assertEqual(load_analysis_history(), [])

    def test_delete_all_history_works(self):
        save_analysis_to_history("img1.jpg", {}, "1s", self.thumb_path1)
        delete_all_history()
        self.assertFalse(os.path.exists(self.history_file))
        self.assertFalse(os.path.exists(self.thumbnail_dir))

    def test_delete_selected_history(self):
        save_analysis_to_history("img1.jpg", {}, "1s", self.thumb_path1)
        save_analysis_to_history("img2.jpg", {}, "2s", self.thumb_path2)
        delete_selected_history([0])
        history = load_analysis_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['image_name'], 'img2.jpg')
        self.assertFalse(os.path.exists(self.thumb_path1))
        self.assertTrue(os.path.exists(self.thumb_path2))

    def test_delete_selected_invalid_index(self):
        save_analysis_to_history("img1.jpg", {}, "1s", self.thumb_path1)
        delete_selected_history([100])
        history = load_analysis_history()
        self.assertEqual(len(history), 1)

    # --- Pengujian untuk classification.py ---

    def test_classify_no_manipulation(self):
        mock_results = create_mock_analysis_results()
        result = classify_manipulation_advanced(mock_results)
        self.assertEqual(result['type'], "Tidak Terdeteksi Manipulasi")
        self.assertLess(result['copy_move_score'], 45)
        self.assertLess(result['splicing_score'], 45)

    def test_classify_strong_copy_move(self):
        mock_results = create_mock_analysis_results(ransac_inliers=30, block_matches=50)
        result = classify_manipulation_advanced(mock_results)
        self.assertEqual(result['type'], "Copy-Move Forgery")
        self.assertGreater(result['copy_move_score'], result['splicing_score'])

    def test_classify_strong_splicing(self):
        mock_results = create_mock_analysis_results(noise_inconsistency=0.6, ela_mean=15)
        result = classify_manipulation_advanced(mock_results)
        self.assertEqual(result['type'], "Splicing Forgery")
        self.assertGreater(result['splicing_score'], result['copy_move_score'])

    def test_classify_complex_manipulation(self):
        mock_results = create_mock_analysis_results(ransac_inliers=20, noise_inconsistency=0.5, block_matches=25)
        result = classify_manipulation_advanced(mock_results)
        self.assertEqual(result['type'], "Manipulasi Kompleks (Copy-Move + Splicing)")
        self.assertGreater(result['copy_move_score'], 60)
        self.assertGreater(result['splicing_score'], 60)

    def test_classify_metadata_anomaly(self):
        mock_results = create_mock_analysis_results(metadata_score=30)
        mock_results['metadata']['Metadata_Inconsistency'] = ['Time difference', 'Editing software', 'Another issue']
        result = classify_manipulation_advanced(mock_results)
        self.assertGreater(result['splicing_score'], 10)


    # --- Pengujian untuk ForensicValidator di app2.py ---

    @unittest.skipIf(not APP2_AVAILABLE, "Skipping ForensicValidator tests as app2.py is not available.")
    def test_validator_with_very_poor_results(self):
        # Menguji ForensicValidator setelah PERBAIKAN BUG
        mock_results = {
            'localization_analysis': {'kmeans_localization': {'cluster_ela_means': [1, 1.1], 'tampering_cluster_id': 0}, 'tampering_percentage': 0.1, 'combined_tampering_mask': MagicMock()},
            'ela_image': MagicMock(spec=Image.Image), 'ela_mean': 1, 'ela_std': 1, 'noise_analysis': {'overall_inconsistency': 0},
            'ela_regional_stats': {'regional_inconsistency': 0.01, 'outlier_regions': 0}, 'ela_quality_stats': [{'mean': 1}],
            'ransac_inliers': 0, 'sift_matches': 1, 'geometric_transform': None, 'block_matches': []
        }
        validator = ForensicValidator()
        _, final_score, _, _ = validator.validate_cross_algorithm(mock_results)
        # Dengan kode yang sudah diperbaiki, skor tidak lagi dipaksa menjadi 80
        self.assertLess(final_score, 50.0)

    @unittest.skipIf(not APP2_AVAILABLE, "Skipping ForensicValidator tests as app2.py is not available.")
    def test_validator_with_perfect_results(self):
        mock_results = {
            'localization_analysis': {'kmeans_localization': {'cluster_ela_means': [2, 25], 'tampering_cluster_id': 1}, 'tampering_percentage': 20.0, 'combined_tampering_mask': MagicMock()},
            'ela_image': MagicMock(spec=Image.Image), 'ela_mean': 15, 'ela_std': 20, 'noise_analysis': {'overall_inconsistency': 0.5},
            'ela_regional_stats': {'regional_inconsistency': 0.7, 'outlier_regions': 5}, 'ela_quality_stats': [{'mean': 15}, {'mean': 5}],
            'ransac_inliers': 50, 'sift_matches': 200, 'geometric_transform': ['homography'], 'block_matches': [{} for _ in range(30)]
        }
        validator = ForensicValidator()
        _, final_score, _, failed_validations = validator.validate_cross_algorithm(mock_results)
        self.assertGreaterEqual(final_score, 95.0)
        self.assertEqual(len(failed_validations), 0)


if __name__ == '__main__':
    unittest.main()