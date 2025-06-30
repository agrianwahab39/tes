# --- START OF FILE visualization.py ---

"""
Visualization Module for Forensic Image Analysis System
Contains functions for creating comprehensive visualizations, plots, and visual reports
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from datetime import datetime
from skimage.filters import sobel
import os
import io
import warnings

# ======================= Imports =======================

warnings.filterwarnings('ignore')

# ======================= Main Visualization Function (DIPERBAIKI) =======================

def visualize_results_advanced(original_pil, analysis_results, output_filename="advanced_forensic_analysis.png"):
    """Advanced visualization with comprehensive forensic analysis results"""
    print("📊 Creating advanced forensic visualization...")
    
    # Figure dan grid untuk 4 baris analisis
    fig = plt.figure(figsize=(24, 16)) 
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.25)
    
    fig.suptitle(
        f"Laporan Visual Analisis Forensik Gambar\nFile: {analysis_results['metadata'].get('Filename', 'N/A')}",
        fontsize=20, fontweight='bold'
    )
    
    # ======================= PERBAIKAN: Buat subplot secara eksplisit =======================
    # Baris 1: Core Visuals
    ax1_1 = fig.add_subplot(gs[0, 0])
    ax1_2 = fig.add_subplot(gs[0, 1])
    ax1_3 = fig.add_subplot(gs[0, 2])
    ax1_4 = fig.add_subplot(gs[0, 3])
    create_core_visuals_grid(ax1_1, ax1_2, ax1_3, ax1_4, original_pil, analysis_results)
    
    # Baris 2: Advanced Analysis Visuals
    ax2_1 = fig.add_subplot(gs[1, 0])
    ax2_2 = fig.add_subplot(gs[1, 1])
    ax2_3 = fig.add_subplot(gs[1, 2])
    ax2_4 = fig.add_subplot(gs[1, 3])
    create_advanced_analysis_grid(ax2_1, ax2_2, ax2_3, ax2_4, original_pil, analysis_results)
    
    # Baris 3: Statistical & Metric Visuals
    ax3_1 = fig.add_subplot(gs[2, 0])
    ax3_2 = fig.add_subplot(gs[2, 1])
    ax3_3 = fig.add_subplot(gs[2, 2])
    ax3_4 = fig.add_subplot(gs[2, 3])
    create_statistical_grid(ax3_1, ax3_2, ax3_3, ax3_4, analysis_results)

    # Baris 4: Summary & Report
    ax_report = fig.add_subplot(gs[3, :])
    create_detailed_report(ax_report, analysis_results)
    # ======================= AKHIR PERBAIKAN =======================

    try:
        plt.savefig(output_filename, dpi=200, bbox_inches='tight')
        print(f"📊 Advanced forensic visualization saved as '{output_filename}'")
        plt.close(fig)
        return output_filename
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
        plt.close(fig)
        return None

# ======================= Grid Helper Functions (DIPERBAIKI) =======================

def create_core_visuals_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create core visuals grid with explicit axes"""
    # Subplot 1: Original Image
    ax1.imshow(original_pil)
    ax1.set_title("1. Gambar Asli", fontsize=11)
    ax1.axis('off')

    # Subplot 2: ELA
    ela_display = ax2.imshow(results['ela_image'], cmap='hot')
    ax2.set_title(f"2. ELA (μ={results['ela_mean']:.1f})", fontsize=11)
    ax2.axis('off')
    plt.colorbar(ela_display, ax=ax2, fraction=0.046, pad=0.04)

    # Subplot 3: Feature Matching
    create_feature_match_visualization(ax3, original_pil, results)

    # Subplot 4: Block Matching
    create_block_match_visualization(ax4, original_pil, results)

def create_advanced_analysis_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create advanced analysis grid with explicit axes"""
    # Subplot 1: Edge Analysis
    create_edge_visualization(ax1, original_pil, results)

    # Subplot 2: Illumination Analysis
    create_illumination_visualization(ax2, original_pil, results)
    
    # Subplot 3: JPEG Ghost
    ghost_display = ax3.imshow(results['jpeg_ghost'], cmap='hot')
    ax3.set_title(f"7. JPEG Ghost ({results['jpeg_ghost_suspicious_ratio']:.1%} susp.)", fontsize=11)
    ax3.axis('off')
    plt.colorbar(ghost_display, ax=ax3, fraction=0.046, pad=0.04)

    # Subplot 4: Combined Heatmap
    combined_heatmap = create_advanced_combined_heatmap(results, original_pil.size)
    ax4.imshow(original_pil, alpha=0.3)
    ax4.imshow(combined_heatmap, cmap='hot', alpha=0.7)
    ax4.set_title("8. Peta Kecurigaan Gabungan", fontsize=11)
    ax4.axis('off')

def create_statistical_grid(ax1, ax2, ax3, ax4, results):
    """Create statistical analysis grid with explicit axes"""
    # Subplot 1: Frequency Analysis
    create_frequency_visualization(ax1, results)
    
    # Subplot 2: Texture Analysis
    create_texture_visualization(ax2, results)

    # Subplot 3: Statistical Analysis
    create_statistical_visualization(ax3, results)

    # Subplot 4: Quality Response
    create_quality_response_plot(ax4, results)

# ======================= Individual Visualization Functions (Tidak Berubah) =======================

def create_feature_match_visualization(ax, original_pil, results):
    img_matches = np.array(original_pil.convert('RGB'))
    if results.get('sift_keypoints') and results.get('ransac_matches'):
        keypoints = results['sift_keypoints']
        matches = results['ransac_matches'][:20]
        for match in matches:
            pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
            pt2 = tuple(map(int, keypoints[match.trainIdx].pt))
            cv2.line(img_matches, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(img_matches, pt1, 4, (255, 0, 0), -1)
            cv2.circle(img_matches, pt2, 4, (0, 0, 255), -1)
    ax.imshow(img_matches)
    ax.set_title(f"3. Feature Matches ({results['ransac_inliers']} inliers)", fontsize=11)
    ax.axis('off')

def create_block_match_visualization(ax, original_pil, results):
    img_blocks = np.array(original_pil.convert('RGB'))
    if results.get('block_matches'):
        for i, match in enumerate(results['block_matches'][:15]):
            x1, y1 = match['block1']; x2, y2 = match['block2']
            color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
            cv2.rectangle(img_blocks, (x1, y1), (x1+16, y1+16), color, 2)
            cv2.rectangle(img_blocks, (x2, y2), (x2+16, y2+16), color, 2)
    ax.imshow(img_blocks)
    ax.set_title(f"4. Block Matches ({len(results['block_matches'])} found)", fontsize=11)
    ax.axis('off')
    
def create_localization_visualization(ax, original_pil, analysis_results):
    loc_analysis = analysis_results.get('localization_analysis', {})
    mask = loc_analysis.get('combined_tampering_mask') 
    tampering_pct = loc_analysis.get('tampering_percentage', 0)

    if mask is not None:
        img_overlay = np.array(original_pil.convert('RGB'))
        red_overlay = np.zeros_like(img_overlay)
        red_overlay[mask.astype(bool)] = [255, 0, 0]
        final_img = cv2.addWeighted(img_overlay, 0.7, red_overlay, 0.3, 0)
        ax.imshow(final_img)
        ax.set_title(f"K-means Localization ({tampering_pct:.1f}%)", fontsize=11)
    else:
        ax.imshow(original_pil)
        ax.set_title("Localization Data Not Found", fontsize=11)
    ax.axis('off')

def create_frequency_visualization(ax, results):
    freq_data = results.get('frequency_analysis', {}).get('dct_stats', {})
    values = [freq_data.get('low_freq_energy', 0), freq_data.get('mid_freq_energy', 0), freq_data.get('high_freq_energy', 0)]
    ax.bar(['Low', 'Mid', 'High'], values, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_title(f"9. Analisis Frekuensi", fontsize=11)
    ax.set_ylabel('Energi DCT')

def create_texture_visualization(ax, results):
    texture_data = results.get('texture_analysis', {}).get('texture_consistency', {})
    metrics = [k.replace('_consistency', '') for k in texture_data.keys()]
    values = list(texture_data.values())
    ax.barh(metrics, values, color='purple', alpha=0.7)
    ax.set_title(f"10. Konsistensi Tekstur", fontsize=11)
    ax.set_xlabel('Skor Inkonsistensi')

def create_edge_visualization(ax, original_pil, results):
    image_gray = np.array(original_pil.convert('L'))
    edges = sobel(image_gray)
    edge_inconsistency = results.get('edge_analysis', {}).get('edge_inconsistency', 0)
    ax.imshow(edges, cmap='gray')
    ax.set_title(f"5. Analisis Tepi (Incons: {edge_inconsistency:.2f})", fontsize=11)
    ax.axis('off')

def create_illumination_visualization(ax, original_pil, results):
    image_array = np.array(original_pil)
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    illumination = lab[:, :, 0]
    illum_inconsistency = results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0)
    ax.imshow(illumination, cmap='gray')
    ax.set_title(f"6. Peta Iluminasi (Incons: {illum_inconsistency:.2f})", fontsize=11)
    ax.axis('off')

def create_statistical_visualization(ax, results):
    stats = results.get('statistical_analysis', {})
    r_entropy = stats.get('R_entropy', stats.get('r_entropy', 0))
    g_entropy = stats.get('G_entropy', stats.get('g_entropy', 0))
    b_entropy = stats.get('B_entropy', stats.get('b_entropy', 0))
    ax.bar(['R', 'G', 'B'], [r_entropy, g_entropy, b_entropy], color=['red', 'green', 'blue'], alpha=0.7)
    ax.set_title(f"11. Entropi Kanal", fontsize=11)
    ax.set_ylabel('Entropi')

def create_quality_response_plot(ax, results):
    qr = results.get('jpeg_analysis', {}).get('quality_responses', [])
    if qr:
        ax.plot([r['quality'] for r in qr], [r['response_mean'] for r in qr], 'b-o', markersize=4)
    ax.set_title(f"12. Respons Kualitas JPEG", fontsize=11)
    ax.set_xlabel('Kualitas')
    ax.set_ylabel('Error Mean')
    ax.grid(True, alpha=0.3)

def create_technical_metrics_plot(ax, results):
    ax.axis('off')
    ax.text(0.5, 0.5, 'Metrics in Report', ha='center', va='center', fontsize=12, alpha=0.5)


def create_advanced_combined_heatmap(analysis_results, image_size):
    w, h = image_size
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    ela_image = analysis_results.get('ela_image')
    if ela_image is not None:
        ela_resized = cv2.resize(np.array(ela_image), (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap += (ela_resized / 255.0) * 0.3
    
    jpeg_ghost = analysis_results.get('jpeg_ghost')
    if jpeg_ghost is not None:
        ghost_resized = cv2.resize(jpeg_ghost, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap += ghost_resized * 0.25
    
    if analysis_results.get('sift_keypoints'):
        feature_map = np.zeros((h, w), dtype=np.float32)
        for kp in analysis_results['sift_keypoints']:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < h and 0 <= x < w:
                cv2.circle(feature_map, (x, y), 20, 1, -1)
        heatmap += cv2.GaussianBlur(feature_map, (31, 31), 0) * 0.2
    
    if analysis_results.get('block_matches'):
        block_map = np.zeros((h, w), dtype=np.float32)
        for match in analysis_results['block_matches']:
            x1, y1 = match['block1']; x2, y2 = match['block2']
            cv2.rectangle(block_map, (x1, y1), (x1+16, y1+16), 1, -1)
            cv2.rectangle(block_map, (x2, y2), (x2+16, y2+16), 1, -1)
        heatmap += block_map * 0.25
        
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
    return heatmap_norm

def create_summary_report(ax, analysis_results):
    ax.axis('off')
    classification = analysis_results.get('classification', {})
    summary_text = f"""RINGKASAN LAPORAN
{'='*25}
Hasil: {classification.get('type', 'N/A')}
Kepercayaan: {classification.get('confidence', 'N/A')}
Skor Copy-Move: {classification.get('copy_move_score', 0)}/100
Skor Splicing: {classification.get('splicing_score', 0)}/100

Temuan Kunci:
"""
    for detail in classification.get('details', [])[:4]:
        summary_text += f"\n• {detail}"
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))


def create_detailed_report(ax, analysis_results):
    ax.axis('off')
    classification = analysis_results.get('classification', {})
    
    result_type = classification.get('type', 'N/A')
    color = "darkred" if "Manipulasi" in result_type or "Splicing" in result_type or "Copy-Move" in result_type else "darkgreen"
    
    text_content = f"Hasil Akhir: {result_type} ({classification.get('confidence', 'N/A')})\n"
    text_content += f"{'-'*40}\n"
    text_content += f"Skor Copy-Move: {classification.get('copy_move_score', 0)}/100\n"
    text_content += f"Skor Splicing: {classification.get('splicing_score', 0)}/100\n\n"
    text_content += "Temuan Kunci:\n" + "\n".join([f" • {d}" for d in classification.get('details', [])])

    ax.text(0.01, 0.95, text_content, transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9), color=color)


def export_kmeans_visualization(original_pil, analysis_results, output_filename="kmeans_analysis.jpg"):
    if 'localization_analysis' not in analysis_results:
        print("❌ K-means analysis not available")
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('K-means Clustering Analysis', fontsize=16)
    loc = analysis_results.get('localization_analysis', {})
    km = loc.get('kmeans_localization', {})
    
    axes[0, 0].imshow(original_pil)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    if 'localization_map' in km:
        axes[0, 1].imshow(km['localization_map'], cmap='viridis')
    axes[0, 1].set_title('Cluster Map')
    axes[0, 1].axis('off')
    
    if 'tampering_mask' in km:
        axes[1, 0].imshow(km['tampering_mask'], cmap='gray')
    axes[1, 0].set_title('Tampering Mask')
    axes[1, 0].axis('off')

    if 'combined_tampering_mask' in loc:
        axes[1, 1].imshow(original_pil)
        axes[1, 1].imshow(loc['combined_tampering_mask'], cmap='Reds', alpha=0.5)
    axes[1, 1].set_title('Final Detection')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_filename
    except Exception as e:
        print(f"❌ K-means visualization export failed: {e}")
        plt.close(fig)
        return None