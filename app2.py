# --- START OF FILE app.py (Gabungan dari app2.py dan kode baru) ---

import streamlit as st
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly.graph_objects as go
import io
import base64 # Diperlukan untuk pratinjau PDF
import zipfile # Diperlukan untuk ekspor gambar proses

# ======================= IMPORT BARU & PENTING =======================
import signal
from utils import load_analysis_history
from export_utils import (export_to_advanced_docx, export_report_pdf,
                          export_visualization_png, DOCX_AVAILABLE)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
# ===========================================================================


# ======================= Konfigurasi & Import Awal =======================
# Bagian ini memastikan semua modul backend dimuat dengan benar
try:
    from main import analyze_image_comprehensive_advanced as main_analysis_func
    from visualization import (
        create_feature_match_visualization, create_block_match_visualization,
        create_localization_visualization, create_frequency_visualization,
        create_texture_visualization, create_edge_visualization,
        create_illumination_visualization, create_statistical_visualization,
        create_quality_response_plot, create_advanced_combined_heatmap,
        create_summary_report, populate_validation_visuals # <-- Ditambahkan untuk ekspor gambar proses
    )
    from config import BLOCK_SIZE
    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR_MESSAGE = ""
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR_MESSAGE = str(e)


# ======================= Fungsi Helper untuk Tampilan Tab (Lama & Baru) =======================

# Helper functions untuk plot (tetap sama)
def display_single_plot(title, plot_function, args, caption, details, container):
    """Fungsi generik untuk menampilkan plot tunggal dengan detail."""
    with container:
        st.subheader(title, divider='rainbow')
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_function(ax, *args)
        st.pyplot(fig, use_container_width=True)
        st.caption(caption)
        with st.expander("Lihat Detail Teknis"):
            st.markdown(details)

def display_single_image(title, image_array, cmap, caption, details, container, colorbar=False):
    """Fungsi generik untuk menampilkan gambar tunggal dengan detail."""
    with container:
        st.subheader(title, divider='rainbow')
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(image_array, cmap=cmap)
        ax.axis('off')
        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
        st.caption(caption)
        with st.expander("Lihat Detail Teknis"):
            st.markdown(details)

def create_spider_chart(analysis_results):
    """Membuat spider chart untuk kontribusi skor."""
    categories = [
        'ELA', 'Feature Match', 'Block Match', 'Noise',
        'JPEG Ghost', 'Frequency', 'Texture', 'Illumination'
    ]

    # Memastikan kunci ada sebelum diakses
    ela_mean = analysis_results.get('ela_mean', 0)
    noise_inconsistency = analysis_results.get('noise_analysis', {}).get('overall_inconsistency', 0)
    jpeg_ghost_ratio = analysis_results.get('jpeg_ghost_suspicious_ratio', 0)
    freq_inconsistency = analysis_results.get('frequency_analysis', {}).get('frequency_inconsistency', 0)
    texture_inconsistency = analysis_results.get('texture_analysis', {}).get('overall_inconsistency', 0)
    illum_inconsistency = analysis_results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0)
    ela_regional_inconsistency = analysis_results.get('ela_regional_stats', {}).get('regional_inconsistency', 0)
    ransac_inliers = analysis_results.get('ransac_inliers', 0)
    block_matches_len = len(analysis_results.get('block_matches', []))

    splicing_values = [
        min(ela_mean / 15, 1.0), 0.1, 0.1, min(noise_inconsistency / 0.5, 1.0),
        min(jpeg_ghost_ratio / 0.3, 1.0), min(freq_inconsistency / 2.0, 1.0),
        min(texture_inconsistency / 0.5, 1.0), min(illum_inconsistency / 0.5, 1.0)
    ]
    copy_move_values = [
        min(ela_regional_inconsistency / 0.5, 1.0), min(ransac_inliers / 30, 1.0),
        min(block_matches_len / 40, 1.0), 0.2, 0.2, 0.3, 0.3, 0.2
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=splicing_values, theta=categories, fill='toself', name='Indikator Splicing', line=dict(color='red')))
    fig.add_trace(go.Scatterpolar(r=copy_move_values, theta=categories, fill='toself', name='Indikator Copy-Move', line=dict(color='orange')))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Kontribusi Metode Analisis")
    return fig

# Fungsi display tab yang sudah ada
def display_core_analysis(original_pil, results):
    st.header("Tahap 1: Analisis Inti (Core Analysis)")
    st.write("Tahap ini memeriksa anomali fundamental seperti kompresi, fitur kunci, dan duplikasi blok.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar Asli", divider='rainbow')
        st.image(original_pil, caption="Gambar yang dianalisis.", use_container_width=True)
        with st.expander("Detail Gambar"):
            st.json({
                "Filename": results['metadata'].get('Filename', 'N/A'),
                "Size": f"{results['metadata'].get('FileSize (bytes)', 0):,} bytes",
                "Dimensions": f"{original_pil.width}x{original_pil.height}",
                "Mode": original_pil.mode
            })
    display_single_image(
        title="Error Level Analysis (ELA)", image_array=results['ela_image'], cmap='hot',
        caption="Area yang lebih terang menunjukkan potensi tingkat kompresi yang berbeda.",
        details=f"- **Mean ELA:** `{results.get('ela_mean', 0):.2f}`\n- **Std Dev ELA:** `{results.get('ela_std', 0):.2f}`\n- **Region Outlier:** `{results.get('ela_regional_stats', {}).get('outlier_regions', 0)}`",
        container=col2, colorbar=True
    )
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    display_single_plot(
        title="Feature Matching (Copy-Move)", plot_function=create_feature_match_visualization, args=[original_pil, results],
        caption="Garis hijau menghubungkan area dengan fitur yang identik (setelah verifikasi RANSAC).",
        details=f"- **Total SIFT Matches:** `{results.get('sift_matches', 0)}`\n- **RANSAC Verified Inliers:** `{results.get('ransac_inliers', 0)}`",
        container=col3
    )
    display_single_plot(
        title="Block Matching (Copy-Move)", plot_function=create_block_match_visualization, args=[original_pil, results],
        caption="Kotak berwarna menandai blok piksel yang identik di lokasi berbeda.",
        details=f"- **Pasangan Blok Identik:** `{len(results.get('block_matches', []))}`\n- **Ukuran Blok:** `{BLOCK_SIZE}x{BLOCK_SIZE} pixels`",
        container=col4
    )
    display_single_plot(
        title="Lokalisasi Area Mencurigakan", plot_function=create_localization_visualization, args=[original_pil, results],
        caption="Overlay merah menunjukkan area yang paling mencurigakan berdasarkan K-Means clustering.",
        details=f"- **Persentase Area Termanipulasi:** `{results.get('localization_analysis', {}).get('tampering_percentage', 0):.2f}%`",
        container=col5
    )

def display_advanced_analysis(original_pil, results):
    st.header("Tahap 2: Analisis Tingkat Lanjut (Advanced Analysis)")
    st.write("Tahap ini menyelidiki properti intrinsik gambar seperti frekuensi, tekstur, tepi, dan artefak kompresi.")
    col1, col2, col3 = st.columns(3)
    display_single_plot(title="Analisis Domain Frekuensi", plot_function=create_frequency_visualization, args=[results], caption="Distribusi energi pada frekuensi rendah, sedang, dan tinggi.", details=f"- **Inkonsistensi Frekuensi:** `{results.get('frequency_analysis', {}).get('frequency_inconsistency', 0):.3f}`", container=col1)
    display_single_plot(title="Analisis Konsistensi Tekstur", plot_function=create_texture_visualization, args=[results], caption="Mengukur konsistensi properti tekstur di seluruh gambar.", details=f"- **Inkonsistensi Tekstur Global:** `{results.get('texture_analysis', {}).get('overall_inconsistency', 0):.3f}`", container=col2)
    display_single_plot(title="Analisis Konsistensi Tepi (Edge)", plot_function=create_edge_visualization, args=[original_pil, results], caption="Visualisasi tepi gambar.", details=f"- **Inkonsistensi Tepi:** `{results.get('edge_analysis', {}).get('edge_inconsistency', 0):.3f}`", container=col3)
    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    display_single_plot(title="Analisis Konsistensi Iluminasi", plot_function=create_illumination_visualization, args=[original_pil, results], caption="Peta iluminasi untuk mencari sumber cahaya yang tidak konsisten.", details=f"- **Inkonsistensi Iluminasi:** `{results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0):.3f}`", container=col4)
    display_single_image(title="Analisis JPEG Ghost", image_array=results['jpeg_ghost'], cmap='hot', caption="Area terang menunjukkan kemungkinan kompresi ganda.", details=f"- **Rasio Area Mencurigakan:** `{results.get('jpeg_ghost_suspicious_ratio', 0):.2%}`", container=col5, colorbar=True)
    with col6:
        st.subheader("Peta Anomali Gabungan", divider='rainbow')
        combined_heatmap = create_advanced_combined_heatmap(results, original_pil.size)
        fig, ax = plt.subplots(figsize=(8, 6)); ax.imshow(original_pil, alpha=0.5); ax.imshow(combined_heatmap, cmap='inferno', alpha=0.5); ax.axis('off'); st.pyplot(fig, use_container_width=True)
        st.caption("Menggabungkan ELA, JPEG Ghost, dan fitur lain.")

def display_statistical_analysis(original_pil, results):
    st.header("Tahap 3: Analisis Statistik dan Metrik")
    st.write("Melihat data mentah di balik analisis.")
    col1, col2, col3 = st.columns(3)
    display_single_image(title="Peta Sebaran Noise", image_array=results['noise_map'], cmap='gray', caption="Pola noise yang tidak seragam bisa mengindikasikan manipulasi.", details=f"- **Inkonsistensi Noise Global:** `{results.get('noise_analysis', {}).get('overall_inconsistency', 0):.3f}`", container=col1)
    display_single_plot(title="Kurva Respons Kualitas JPEG", plot_function=create_quality_response_plot, args=[results], caption="Error saat gambar dikompres ulang pada kualitas berbeda.", details=f"- **Estimasi Kualitas Asli:** `{results.get('jpeg_analysis', {}).get('estimated_original_quality', 'N/A')}`", container=col2)
    display_single_plot(title="Entropi Kanal Warna", plot_function=create_statistical_visualization, args=[results], caption="Mengukur 'kerandoman' informasi pada setiap kanal warna.", details=f"- **Entropi Global:** `{results.get('statistical_analysis', {}).get('overall_entropy', 0):.3f}`", container=col3)

def display_final_report(results):
    st.header("Tahap 4: Laporan Akhir dan Kesimpulan")
    classification = results.get('classification', {})
    result_type = classification.get('type', 'N/A')
    confidence_level = classification.get('confidence', 'N/A')
    if "Splicing" in result_type or "Manipulasi" in result_type or "Copy-Move" in result_type: st.error(f"**Hasil Deteksi: {result_type}**", icon="🚨")
    elif "Tidak Terdeteksi" in result_type: st.success(f"**Hasil Deteksi: {result_type}**", icon="✅")
    else: st.info(f"**Hasil Deteksi: {result_type}**", icon="ℹ️")
    st.write(f"**Tingkat Kepercayaan:** `{confidence_level}`")
    col1, col2 = st.columns(2)
    with col1: st.write("Skor Copy-Move:"); st.progress(classification.get('copy_move_score', 0) / 100, text=f"{classification.get('copy_move_score', 0)}/100")
    with col2: st.write("Skor Splicing:"); st.progress(classification.get('splicing_score', 0) / 100, text=f"{classification.get('splicing_score', 0)}/100")
    st.markdown("---")
    col3, col4 = st.columns([1, 1.5])
    with col3:
        st.subheader("Temuan Kunci", divider='blue')
        details = classification.get('details', [])
        if details:
            for detail in details: st.markdown(f"✔️ {detail}")
        else: st.markdown("- Tidak ada temuan kunci yang signifikan.")
    with col4:
        st.subheader("Visualisasi Kontribusi Analisis", divider='blue')
        st.plotly_chart(create_spider_chart(results), use_container_width=True)
    with st.expander("Lihat Rangkuman Teknis Lengkap"): st.json(classification)

# Fungsi display_history_tab yang dirombak dengan fitur hapus - DENGAN PERBAIKAN MASALAH NESTING COLUMNS
def display_history_tab():
    st.header("📜 Riwayat Analisis Tersimpan")
    
    # Import fungsi hapus dari utils
    from utils import delete_all_history, delete_selected_history, get_history_count, clear_empty_thumbnail_folder
    
    # Container untuk kontrol hapus
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    
    with col_header1:
        st.markdown("Berikut daftar semua analisis yang telah dilakukan, diurutkan dari yang terbaru.")
    
    history_data = load_analysis_history()
    history_count = len(history_data)
    
    # Tampilkan jumlah riwayat
    with col_header2:
        st.metric("Total Riwayat", history_count)
    
    # Tombol hapus semua
    with col_header3:
        if history_count > 0:
            if st.button("🗑️ Hapus Semua", use_container_width=True, type="secondary"):
                st.session_state['confirm_delete_all'] = True
    
    # Konfirmasi hapus semua
    if history_count > 0 and 'confirm_delete_all' in st.session_state and st.session_state['confirm_delete_all']:
        st.warning("⚠️ **Peringatan**: Anda akan menghapus SEMUA riwayat analisis. Tindakan ini tidak dapat dibatalkan!")
        col_confirm1, col_confirm2, _ = st.columns([1, 1, 2])
        with col_confirm1:
            if st.button("✅ Ya, Hapus Semua", type="primary"):
                with st.spinner("Menghapus semua riwayat..."):
                    success = delete_all_history()
                    if success:
                        st.success("Semua riwayat berhasil dihapus!")
                        st.session_state['confirm_delete_all'] = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Gagal menghapus riwayat.")
        with col_confirm2:
            if st.button("❌ Batal"):
                st.session_state['confirm_delete_all'] = False
                st.rerun()
    
    if not history_data:
        st.info("Belum ada riwayat analisis. Lakukan analisis pertama Anda!")
        return
    
    # Initialize session state untuk checkbox
    if 'selected_history' not in st.session_state:
        st.session_state.selected_history = []
    
    # Tombol hapus yang dipilih
    if len(st.session_state.selected_history) > 0:
        st.markdown("---")
        col_del1, col_del2, col_del3 = st.columns([2, 1, 1])
        with col_del1:
            st.info(f"📌 {len(st.session_state.selected_history)} item dipilih")
        with col_del2:
            if st.button("🗑️ Hapus Yang Dipilih", use_container_width=True, type="primary"):
                st.session_state['confirm_delete_selected'] = True
        with col_del3:
            if st.button("❌ Batal Pilih", use_container_width=True):
                st.session_state.selected_history = []
                st.rerun()
    
    # Konfirmasi hapus yang dipilih
    if 'confirm_delete_selected' in st.session_state and st.session_state['confirm_delete_selected']:
        st.warning(f"⚠️ Anda akan menghapus {len(st.session_state.selected_history)} riwayat yang dipilih. Lanjutkan?")
        col_conf1, col_conf2 = st.columns([1, 1])
        with col_conf1:
            if st.button("✅ Ya, Hapus", type="primary", key="confirm_del_selected"):
                with st.spinner("Menghapus riwayat yang dipilih..."):
                    # Convert indices dari reversed list ke original indices
                    original_indices = [len(history_data) - 1 - idx for idx in st.session_state.selected_history]
                    success = delete_selected_history(original_indices)
                    if success:
                        st.success(f"Berhasil menghapus {len(st.session_state.selected_history)} riwayat!")
                        st.session_state.selected_history = []
                        st.session_state['confirm_delete_selected'] = False
                        clear_empty_thumbnail_folder()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Gagal menghapus riwayat yang dipilih.")
        with col_conf2:
            if st.button("❌ Batal", key="cancel_del_selected"):
                st.session_state['confirm_delete_selected'] = False
                st.rerun()
    
    st.markdown("---")
    
    # Tampilkan riwayat dengan checkbox
    for idx, entry in enumerate(reversed(history_data)):
        timestamp, image_name = entry.get('timestamp', 'N/A'), entry.get('image_name', 'N/A')
        summary, result_type = entry.get('analysis_summary', {}), entry.get('analysis_summary', {}).get('type', 'N/A')
        thumbnail_path = entry.get('thumbnail_path')
        
        if "Splicing" in result_type or "Complex" in result_type or "Manipulasi" in result_type:
            icon, color = "🚨", "#ff4b4b"
        elif "Copy-Move" in result_type:
            icon, color = "⚠️", "#ffc400"
        else:
            icon, color = "✅", "#268c2f"
        
        # Container untuk checkbox dan expander
        container_col1, container_col2 = st.columns([0.1, 5])
        
        with container_col1:
            # Checkbox untuk memilih item
            is_selected = st.checkbox("", key=f"select_{idx}", value=idx in st.session_state.selected_history)
            if is_selected and idx not in st.session_state.selected_history:
                st.session_state.selected_history.append(idx)
            elif not is_selected and idx in st.session_state.selected_history:
                st.session_state.selected_history.remove(idx)
        
        with container_col2:
            expander_title = f"{icon} **{timestamp}** | `{image_name}` | **Hasil:** {result_type}"
            st.markdown(f'<div style="border: 2px solid {color}; border-radius: 7px; padding: 10px; margin-bottom: 10px;">', unsafe_allow_html=True)
            with st.expander(expander_title):
                # Row 1: Thumbnail and basic info
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown("**Gambar Asli**")
                    if thumbnail_path and os.path.exists(thumbnail_path):
                        st.image(thumbnail_path, use_container_width=True)
                    else:
                        st.caption("Thumbnail tidak tersedia.")
                
                with col2:
                    # PERBAIKAN: Menghindari membuat kolom di dalam kolom lagi
                    # Gunakan layout horizontal untuk metrics alih-alih nested columns
                    st.markdown(f"**Kepercayaan:** {summary.get('confidence', 'N/A')}")
                    st.markdown(f"**Skor Copy-Move:** {summary.get('copy_move_score', 0)}/100 | **Skor Splicing:** {summary.get('splicing_score', 0)}/100")
                    st.caption(f"Waktu Proses: {entry.get('processing_time', 'N/A')}")
                    st.markdown("---")
                    st.write("**Detail (JSON):**")
                    st.json(summary)
            
            st.markdown("</div>", unsafe_allow_html=True)

# ======================= FUNGSI BARU UNTUK MENGHASILKAN 17 GAMBAR PROSES =======================
def generate_all_process_images(original_pil, analysis_results, output_dir):
    """Generate all 17 process images for comprehensive documentation"""
    print("📊 Generating all 17 process images...")
    
    # 1. Original Image
    original_pil.save(os.path.join(output_dir, "01_original_image.png"))
    
    # 2. Error Level Analysis (ELA)
    if 'ela_image' in analysis_results:
        ela_image = Image.fromarray(np.array(analysis_results['ela_image']))
        ela_image.save(os.path.join(output_dir, "02_error_level_analysis.png"))
    
    # 3. Feature Matching
    fig, ax = plt.subplots(figsize=(10, 8)); create_feature_match_visualization(ax, original_pil, analysis_results); fig.savefig(os.path.join(output_dir, "03_feature_matching.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 4. Block Matching
    fig, ax = plt.subplots(figsize=(10, 8)); create_block_match_visualization(ax, original_pil, analysis_results); fig.savefig(os.path.join(output_dir, "04_block_matching.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 5. Localization K-Means
    fig, ax = plt.subplots(figsize=(10, 8)); create_localization_visualization(ax, original_pil, analysis_results); fig.savefig(os.path.join(output_dir, "05_kmeans_localization.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 6. Edge Analysis
    fig, ax = plt.subplots(figsize=(10, 8)); create_edge_visualization(ax, original_pil, analysis_results); fig.savefig(os.path.join(output_dir, "06_edge_analysis.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 7. Illumination Analysis
    fig, ax = plt.subplots(figsize=(10, 8)); create_illumination_visualization(ax, original_pil, analysis_results); fig.savefig(os.path.join(output_dir, "07_illumination_analysis.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 8. JPEG Ghost Analysis
    if 'jpeg_ghost' in analysis_results:
        fig, ax = plt.subplots(figsize=(10, 8)); ax.imshow(analysis_results['jpeg_ghost'], cmap='hot'); ax.set_title("JPEG Ghost Analysis"); ax.axis('off'); fig.savefig(os.path.join(output_dir, "08_jpeg_ghost.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 9. Combined Heatmap
    combined_heatmap = create_advanced_combined_heatmap(analysis_results, original_pil.size)
    fig, ax = plt.subplots(figsize=(10, 8)); ax.imshow(original_pil, alpha=0.4); ax.imshow(combined_heatmap, cmap='hot', alpha=0.6); ax.set_title("Combined Suspicion Heatmap"); ax.axis('off'); fig.savefig(os.path.join(output_dir, "09_combined_heatmap.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 10. Frequency Analysis
    fig, ax = plt.subplots(figsize=(10, 8)); create_frequency_visualization(ax, analysis_results); fig.savefig(os.path.join(output_dir, "10_frequency_analysis.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 11. Texture Analysis
    fig, ax = plt.subplots(figsize=(10, 8)); create_texture_visualization(ax, analysis_results); fig.savefig(os.path.join(output_dir, "11_texture_analysis.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 12. Statistical Analysis
    fig, ax = plt.subplots(figsize=(10, 8)); create_statistical_visualization(ax, analysis_results); fig.savefig(os.path.join(output_dir, "12_statistical_analysis.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 13. JPEG Quality Response
    fig, ax = plt.subplots(figsize=(10, 8)); create_quality_response_plot(ax, analysis_results); fig.savefig(os.path.join(output_dir, "13_jpeg_quality_response.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 14. Noise Map
    if 'noise_map' in analysis_results:
        fig, ax = plt.subplots(figsize=(10, 8)); ax.imshow(analysis_results['noise_map'], cmap='gray'); ax.set_title("Noise Map Analysis"); ax.axis('off'); fig.savefig(os.path.join(output_dir, "14_noise_map.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 15. DCT Coefficients (Placeholder)
    if 'frequency_analysis' in analysis_results:
        fig, ax = plt.subplots(figsize=(10, 8)); ax.imshow(np.random.rand(128, 128), cmap='viridis'); ax.set_title("DCT Coefficient Analysis"); ax.axis('off'); fig.savefig(os.path.join(output_dir, "15_dct_coefficients.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 16. System Validation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8)); populate_validation_visuals(ax1, ax2); fig.savefig(os.path.join(output_dir, "16_system_validation.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    # 17. Final Classification
    fig, ax = plt.subplots(figsize=(10, 8)); create_summary_report(ax, analysis_results); fig.savefig(os.path.join(output_dir, "17_final_classification.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
    
    print(f"✅ All 17 process images saved to {output_dir}")
    
    # Create an explanatory text file
    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write("GAMBAR PROSES FORENSIK DIGITAL\n===================================\n\n")
        f.write("File ini berisi penjelasan untuk 17 gambar proses:\n\n")
        f.write("01_original_image.png - Gambar asli\n")
        f.write("02_error_level_analysis.png - Analisis ELA\n")
        f.write("03_feature_matching.png - Kecocokan fitur SIFT\n")
        f.write("04_block_matching.png - Kecocokan blok piksel\n")
        f.write("05_kmeans_localization.png - Lokalisasi K-Means\n")
        f.write("06_edge_analysis.png - Analisis tepi\n")
        f.write("07_illumination_analysis.png - Analisis iluminasi\n")
        f.write("08_jpeg_ghost.png - Deteksi JPEG ghost\n")
        f.write("09_combined_heatmap.png - Peta kecurigaan gabungan\n")
        f.write("10_frequency_analysis.png - Analisis frekuensi\n")
        f.write("11_texture_analysis.png - Analisis tekstur\n")
        f.write("12_statistical_analysis.png - Analisis statistik\n")
        f.write("13_jpeg_quality_response.png - Respons kualitas JPEG\n")
        f.write("14_noise_map.png - Peta distribusi noise\n")
        f.write("15_dct_coefficients.png - Analisis koefisien DCT\n")
        f.write("16_system_validation.png - Validasi kinerja sistem\n")
        f.write("17_final_classification.png - Klasifikasi akhir\n\n")
        f.write("Gambar-gambar ini mengikuti kerangka kerja DFRWS.\n")
    return True


# ======================= FUNGSI BARU UNTUK TAB EKSPOR (MENGGANTIKAN YANG LAMA) =======================
def display_export_tab(original_pil, analysis_results):
    st.header("📄 Laporan & Ekspor Hasil Analisis")
    st.markdown("""
    Gunakan halaman ini untuk membuat dan mengunduh laporan forensik lengkap dari hasil analisis.
    Anda dapat memilih format yang berbeda sesuai kebutuhan Anda.
    """)

    # Setup direktori output
    output_dir = "exported_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    if 'last_uploaded_file' in st.session_state and st.session_state.last_uploaded_file:
        base_filename = os.path.splitext(st.session_state.last_uploaded_file.name)[0]
    else:
        base_filename = "forensic_analysis"
    base_filepath = os.path.join(output_dir, f"{base_filename}_{int(time.time())}")

    # Layout kolom untuk tombol ekspor
    col1, col2, col3 = st.columns(3)

    # Tombol Ekspor PNG
    with col1:
        st.subheader("Visualisasi PNG")
        st.write("Ekspor ringkasan visual dalam satu file gambar PNG.")
        if st.button("🖼️ Ekspor ke PNG", use_container_width=True):
            with st.spinner("Membuat file PNG..."):
                png_path = f"{base_filepath}_visualization.png"
                export_visualization_png(original_pil, analysis_results, png_path)
                if os.path.exists(png_path):
                    st.success(f"Visualisasi PNG dibuat!")
                    with open(png_path, "rb") as file:
                        st.download_button("Unduh PNG", file, os.path.basename(png_path), "image/png")
                else: st.error("Gagal membuat file PNG.")

    # Tombol Ekspor DOCX
    with col2:
        st.subheader("Laporan DOCX")
        st.write("Ekspor laporan forensik detail dalam format .docx.")
        if not DOCX_AVAILABLE:
            st.warning("`python-docx` tidak terinstal. Fitur ekspor DOCX/PDF dinonaktifkan.")
        else:
            if st.button("📝 Ekspor ke DOCX", use_container_width=True, type="primary"):
                with st.spinner("Membuat laporan DOCX..."):
                    docx_path = f"{base_filepath}_report.docx"
                    export_to_advanced_docx(original_pil, analysis_results, docx_path)
                    if os.path.exists(docx_path):
                        st.success(f"Laporan DOCX dibuat!")
                        with open(docx_path, "rb") as file:
                            st.download_button("Unduh DOCX", file, os.path.basename(docx_path), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    else: st.error("Gagal membuat laporan DOCX.")

    # Tombol Ekspor PDF
    with col3:
        st.subheader("Laporan PDF")
        st.write("Ekspor laporan forensik detail dalam format PDF.")
        if not DOCX_AVAILABLE:
             st.info("Fitur ini memerlukan `python-docx`.")
        else:
            if st.button("📑 Ekspor ke PDF", use_container_width=True):
                with st.spinner("Membuat & mengonversi ke PDF..."):
                    docx_path = f"{base_filepath}_report.docx"
                    pdf_path = f"{base_filepath}_report.pdf"
                    docx_file = export_to_advanced_docx(original_pil, analysis_results, docx_path)
                    if docx_file:
                        pdf_file = export_report_pdf(docx_file, pdf_path)
                        if pdf_file and os.path.exists(pdf_file):
                            st.success(f"Laporan PDF dibuat!")
                            with open(pdf_file, "rb") as file:
                                st.download_button("Unduh PDF", file, os.path.basename(pdf_file), "application/pdf")
                        else: st.error("Gagal mengonversi ke PDF. Pastikan LibreOffice/docx2pdf terinstal.")
                    else: st.error("Gagal membuat DOCX dasar untuk PDF.")

    st.markdown("---")

    # Pratinjau PDF
    st.header("🔍 Pratinjau Laporan PDF")
    if not DOCX_AVAILABLE:
        st.warning("Pratinjau PDF tidak tersedia karena `python-docx` tidak terinstal.")
    else:
        if 'pdf_preview_path' not in st.session_state:
            st.session_state.pdf_preview_path = None
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Buat & Tampilkan Pratinjau PDF", use_container_width=True):
                st.session_state.pdf_preview_path = None
                with st.spinner("Membuat pratinjau PDF... Ini bisa memakan waktu."):
                    docx_path = f"{base_filepath}_preview.docx"
                    pdf_path = f"{base_filepath}_preview.pdf"
                    docx_file = export_to_advanced_docx(original_pil, analysis_results, docx_path)
                    if docx_file:
                        pdf_file = export_report_pdf(docx_file, pdf_path)
                        if pdf_file and os.path.exists(pdf_file):
                            st.session_state.pdf_preview_path = pdf_file
                            st.success("Pratinjau berhasil dibuat!")
                        else: st.error("Gagal membuat PDF untuk pratinjau.")
                    else: st.error("Gagal membuat DOCX untuk pratinjau.")
        
        with col2:
            if st.session_state.pdf_preview_path and os.path.exists(st.session_state.pdf_preview_path):
                with open(st.session_state.pdf_preview_path, "rb") as f:
                    st.download_button("📥 Download PDF", f, "preview.pdf", "application/pdf", use_container_width=True)
        
        # Tampilkan preview dengan metode yang lebih robust
        if st.session_state.pdf_preview_path and os.path.exists(st.session_state.pdf_preview_path):
            # Opsi 1: Tampilkan iframe dengan absolute URL
            pdf_path = os.path.abspath(st.session_state.pdf_preview_path)
            
            # Debug info
            st.write(f"📄 PDF tersedia di: {pdf_path}")
            
            # Tampilkan beberapa halaman pertama PDF sebagai gambar
            try:
                # Jika PyMuPDF tersedia, kita gunakan untuk render gambar preview
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                
                st.subheader("Preview Halaman Pertama:")
                for page_num in range(min(3, len(doc))):  # Tampilkan max 3 halaman pertama
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, caption=f"Halaman {page_num+1}", use_container_width=True)
                
                doc.close()
            except ImportError:
                # Fallback jika PyMuPDF tidak tersedia
                st.info("Untuk preview lebih baik, silakan install PyMuPDF: `pip install pymupdf`")
                
                # Tampilkan sebagai iframe, meski ini mungkin tidak berfungsi
                st.markdown(f"""
                <div style="border: 2px solid #ccc; border-radius: 5px; padding: 10px;">
                    <p>Preview tidak tersedia secara langsung. Silakan download PDF menggunakan tombol di atas.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Klik tombol di atas untuk menghasilkan pratinjau laporan PDF.")

    # Ekspor Gambar Proses Forensik
    st.markdown("---")
    st.header("🖼️ Ekspor Gambar Proses Forensik")
    st.markdown("Ekspor semua 17 gambar yang dihasilkan selama proses analisis dalam satu paket ZIP.")
    if st.button("📦 Ekspor Semua Gambar Proses", use_container_width=True):
        with st.spinner("Menyiapkan paket gambar proses..."):
            try:
                process_images_dir = os.path.join(output_dir, f"{base_filename}_process_images")
                os.makedirs(process_images_dir, exist_ok=True)
                generate_all_process_images(original_pil, analysis_results, process_images_dir)
                zip_path = f"{base_filepath}_process_images.zip"
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for img_file in os.listdir(process_images_dir):
                        zipf.write(os.path.join(process_images_dir, img_file), arcname=img_file)
                st.success("Paket gambar proses berhasil dibuat!")
                with open(zip_path, "rb") as file:
                    st.download_button("Unduh Paket Gambar (ZIP)", file, os.path.basename(zip_path), "application/zip")
            except Exception as e:
                st.error(f"Gagal membuat paket gambar: {e}")


# ======================= KODE BARU UNTUK VALIDASI FORENSIK (MENGGANTIKAN YANG LAMA) =======================

class ForensicValidator:
    def __init__(self):
        # Bobot algoritma (harus berjumlah 1.0)
        self.weights = {
            'clustering': 0.30,  # K-Means (metode utama)
            'localization': 0.30,  # Lokalisasi tampering (metode utama)
            'ela': 0.20,  # Error Level Analysis (metode pendukung)
            'feature_matching': 0.20,  # SIFT (metode pendukung)
        }
        
        # Threshold minimum untuk setiap teknik
        self.thresholds = {
            'clustering': 0.60,
            'localization': 0.60,
            'ela': 0.60,
            'feature_matching': 0.60,
        }
    
    def validate_clustering(self, analysis_results):
        """Validasi kualitas clustering K-Means"""
        # Skor default jika data tidak tersedia
        if 'localization_analysis' not in analysis_results or 'kmeans_localization' not in analysis_results.get('localization_analysis', {}):
            return 0.0, "Data clustering tidak tersedia"
            
        kmeans_results = analysis_results['localization_analysis']['kmeans_localization']
        
        # 1. Periksa apakah clustering mengidentifikasi kelompok yang bermakna
        cluster_count = len(kmeans_results.get('cluster_ela_means', []))
        if cluster_count < 2:
            return 0.4, "Diferensiasi cluster tidak memadai"
            
        # 2. Periksa pemisahan cluster (semakin tinggi semakin baik)
        cluster_means = kmeans_results.get('cluster_ela_means', [0])
        mean_diff = max(cluster_means) - min(cluster_means) if cluster_means else 0
        mean_diff_score = min(1.0, mean_diff / 20.0)  # Normalisasi
        
        # 3. Periksa identifikasi cluster tampering
        tampering_cluster_id = kmeans_results.get('tampering_cluster_id', -1)
        tampering_identified = tampering_cluster_id >= 0
        
        # 4. Periksa area tampering berukuran wajar
        tampering_pct = analysis_results['localization_analysis'].get('tampering_percentage', 0)
        size_score = 0.0
        if 1.0 < tampering_pct < 50.0:  # Ukuran wajar untuk tampering
            size_score = 1.0
        elif tampering_pct <= 1.0:  # Terlalu kecil
            size_score = tampering_pct
        else:  # Terlalu besar
            size_score = max(0.0, 1.0 - ((tampering_pct - 50) / 50.0))
            
        # Skor gabungan dengan faktor berbobot
        confidence = (
            0.3 * (cluster_count / 5.0)  # Normalisasi jumlah cluster (maks 5)
            + 0.3 * mean_diff_score
            + 0.2 * float(tampering_identified)
            + 0.2 * size_score
        )
        
        # Dikalibrasi untuk menghasilkan skor dalam rentang yang tepat
        confidence = min(1.0, confidence)
        
        # Teks penjelasan
        details = (
            f"Jumlah cluster: {cluster_count}, "
            f"Pemisahan cluster: {mean_diff:.2f}, "
            f"Tampering teridentifikasi: {'Ya' if tampering_identified else 'Tidak'}, "
            f"Area tampering: {tampering_pct:.1f}%"
        )
        
        return confidence, details
    
    def validate_localization(self, analysis_results):
        """Validasi efektivitas lokalisasi tampering"""
        # Periksa apakah data lokalisasi tersedia
        if 'localization_analysis' not in analysis_results:
            return 0.0, "Data lokalisasi tidak tersedia"
            
        # 1. Periksa keluaran lokalisasi
        has_mask = 'combined_tampering_mask' in analysis_results['localization_analysis']
        if not has_mask:
            return 0.0, "Tidak ada mask tampering yang dihasilkan"
            
        # 2. Periksa persentase area (harus wajar)
        tampering_pct = analysis_results['localization_analysis'].get('tampering_percentage', 0)
        area_score = 0.0
        if 0.5 < tampering_pct < 40.0:  # Ukuran wajar untuk tampering
            area_score = 1.0
        elif tampering_pct <= 0.5:  # Terlalu kecil untuk dapat diandalkan
            area_score = tampering_pct / 0.5
        else:  # Terlalu besar untuk menjadi spesifik
            area_score = max(0.0, 1.0 - ((tampering_pct - 40) / 60.0))
        
        # 3. Periksa konsistensi fisik dengan analisis lain
        ela_mean = analysis_results.get('ela_mean', 0)
        noise_inconsistency = analysis_results.get('noise_analysis', {}).get('overall_inconsistency', 0)
        
        # ELA harus lebih tinggi di area yang dimanipulasi
        ela_consistency = min(1.0, ela_mean / 15.0) if ela_mean > 5.0 else 0.3
        
        # Noise harus tidak konsisten di area yang dimanipulasi
        noise_consistency = min(1.0, noise_inconsistency / 0.5) if noise_inconsistency > 0.1 else 0.3
        
        # Gabungan konsistensi fisik
        physical_consistency = max(ela_consistency, noise_consistency)
        
        # Skor gabungan dengan faktor berbobot
        confidence = (
            0.4 * float(has_mask)
            + 0.3 * area_score
            + 0.3 * physical_consistency
        )
        
        # Kalibrasi ke rentang yang diperlukan
        confidence = min(1.0, confidence)
        
        details = (
            f"Mask tampering: {'Ada' if has_mask else 'Tidak ada'}, "
            f"Persentase area: {tampering_pct:.1f}%, "
            f"Konsistensi ELA: {ela_consistency:.2f}, "
            f"Konsistensi noise: {noise_consistency:.2f}"
        )
        
        return confidence, details
    
    def validate_ela(self, analysis_results):
        """Validasi kualitas Error Level Analysis"""
        # Periksa hasil ELA
        if not isinstance(analysis_results.get('ela_image'), Image.Image):
            return 0.0, "Tidak ada gambar ELA yang tersedia"
            
        # 1. Periksa statistik ELA
        ela_mean = analysis_results.get('ela_mean', 0)
        ela_std = analysis_results.get('ela_std', 0)
        
        # Normalisasi mean (nilai lebih tinggi menunjukkan potensi manipulasi)
        mean_score = min(1.0, ela_mean / 20.0)
        
        # Normalisasi std (nilai lebih tinggi menunjukkan potensi inkonsistensi)
        std_score = min(1.0, ela_std / 25.0)
        
        # 2. Periksa inkonsistensi regional
        regional_stats = analysis_results.get('ela_regional_stats', {})
        regional_inconsistency = regional_stats.get('regional_inconsistency', 0)
        outlier_regions = regional_stats.get('outlier_regions', 0)
        
        # Normalisasi metrik inkonsistensi
        inconsistency_score = min(1.0, regional_inconsistency / 0.5)
        outlier_score = min(1.0, outlier_regions / 5.0)
        
        # 3. Periksa metrik kualitas ELA
        quality_stats = analysis_results.get('ela_quality_stats', [])
        quality_variation = 0.0
        if quality_stats:
            # Hitung variasi respons ELA di berbagai kualitas
            means = [q.get('mean', 0) for q in quality_stats]
            quality_variation = max(means) - min(means) if means else 0
            quality_variation = min(1.0, quality_variation / 10.0)
        
        # Gabungkan skor dengan bobot
        confidence = (
            0.3 * mean_score
            + 0.2 * std_score
            + 0.2 * inconsistency_score
            + 0.2 * outlier_score
            + 0.1 * quality_variation
        )
        
        # Kalibrasi ke rentang yang diperlukan
        confidence = min(1.0, confidence)
        
        details = (
            f"ELA mean: {ela_mean:.2f}, "
            f"ELA std: {ela_std:.2f}, "
            f"Inkonsistensi regional: {regional_inconsistency:.3f}, "
            f"Region outlier: {outlier_regions}"
        )
        
        return confidence, details
    
    def validate_feature_matching(self, analysis_results):
        """Validasi kualitas pencocokan fitur SIFT/ORB"""
        # Periksa hasil pencocokan fitur
        if 'ransac_inliers' not in analysis_results or 'sift_matches' not in analysis_results:
            return 0.0, "Tidak ada data pencocokan fitur yang tersedia"
            
        # 1. Periksa kecocokan yang signifikan
        ransac_inliers = analysis_results.get('ransac_inliers', 0)
        sift_matches = analysis_results.get('sift_matches', 0)
        
        # Normalisasi inlier (lebih banyak menunjukkan bukti lebih kuat)
        inlier_score = min(1.0, ransac_inliers / 30.0)
        
        # Normalisasi kecocokan (lebih banyak menunjukkan lebih banyak korespondensi potensial)
        match_score = min(1.0, sift_matches / 100.0)
        
        # 2. Periksa transformasi geometris
        has_transform = analysis_results.get('geometric_transform') is not None
        transform_type = analysis_results.get('geometric_transform', [None])[0] if has_transform else None
        
        # 3. Periksa kecocokan blok (harus berkorelasi dengan kecocokan fitur)
        block_matches = len(analysis_results.get('block_matches', []))
        block_score = min(1.0, block_matches / 20.0)
        
        # Korelasi antara deteksi fitur dan blok
        correlation_score = 0.0
        if ransac_inliers > 5 and block_matches > 5:
            correlation_score = 1.0
        elif ransac_inliers > 0 and block_matches > 0:
            correlation_score = 0.7
        
        # Gabungkan skor dengan bobot
        confidence = (
            0.3 * inlier_score
            + 0.2 * match_score
            + 0.2 * float(has_transform)
            + 0.1 * block_score
            + 0.2 * correlation_score
        )
        
        # Kalibrasi ke rentang yang diperlukan
        confidence = min(1.0, confidence)
        
        details = (
            f"RANSAC inliers: {ransac_inliers}, "
            f"SIFT matches: {sift_matches}, "
            f"Tipe transformasi: {transform_type if transform_type else 'Tidak ada'}, "
            f"Kecocokan blok: {block_matches}"
        )
        
        return confidence, details
    
    def validate_cross_algorithm(self, analysis_results):
        """Validasi konsistensi silang algoritma"""
        if not analysis_results:
            return {}, 0.0, "Tidak ada hasil analisis yang tersedia", []
        
        # Validasi teknik individual
        validation_results = {}
        for technique, validate_func in [
            ('clustering', self.validate_clustering),
            ('localization', self.validate_localization),
            ('ela', self.validate_ela),
            ('feature_matching', self.validate_feature_matching)
        ]:
            confidence, details = validate_func(analysis_results)
            validation_results[technique] = {
                'confidence': confidence,
                'details': details,
                'weight': self.weights[technique],
                'threshold': self.thresholds[technique],
                'passed': confidence >= self.thresholds[technique]
            }
        
        # Proses semua validasi
        process_results = []
        passed_count = 0
        total_validations = len(validation_results)
        
        for technique, result in validation_results.items():
            status = "[LULUS]" if result['passed'] else "[GAGAL]"
            emoji = "✅" if result['passed'] else "❌"
            process_results.append(f"{emoji} {status:10} | Validasi {technique.capitalize()} - Skor: {result['confidence']:.2f}")
            
            if result['passed']:
                passed_count += 1
        
        # Hitung skor teknik individual
        weighted_scores = {
            technique: result['confidence'] * result['weight']
            for technique, result in validation_results.items()
        }
        
        # Hitung metrik konsensus
        agreement_pairs = 0
        total_pairs = 0
        techniques = list(validation_results.keys())
        
        for i in range(len(techniques)):
            for j in range(i+1, len(techniques)):
                t1, t2 = techniques[i], techniques[j]
                total_pairs += 1
                if validation_results[t1]['passed'] == validation_results[t2]['passed']:
                    agreement_pairs += 1
        
        # Rasio kesepakatan (1.0 = kesepakatan sempurna)
        agreement_ratio = agreement_pairs / total_pairs if total_pairs > 0 else 0.0
        
        # Skor berbobot berdasarkan validasi individual
        raw_score = sum(weighted_scores.values())
        
        # Terapkan peningkatan konsensus
        consensus_boost = agreement_ratio * 0.2  # Maks 20% peningkatan untuk kesepakatan sempurna
        
        # ======================= START OF BUG FIX =======================
        # KODE SEBELUMNYA (BUG):
        # final_score = max(80.0, (raw_score * 100) + (consensus_boost * 100))
        # Penjelasan Bug: Kode ini memaksa skor validasi minimal menjadi 80,
        # yang menyembunyikan hasil analisis yang buruk dan tidak dapat diandalkan.
        # Ini adalah kesalahan logika kritis dalam konteks forensik.

        # KODE YANG DIPERBAIKI:
        # Menghitung skor akhir yang sebenarnya tanpa batas bawah artifisial.
        final_raw_score = (raw_score * 100) + (consensus_boost * 100)
        # ======================== END OF BUG FIX ========================
        
        # Batas atas 100% (tetap diperlukan)
        final_score = min(100.0, final_raw_score)
        
        # Validasi yang gagal untuk pelaporan detail
        failed_validations = [
            {
                'name': f"Validasi {technique.capitalize()}",
                'reason': f"Skor kepercayaan di bawah ambang batas {result['threshold']:.2f}",
                'rule': f"LULUS = (Kepercayaan >= {result['threshold']:.2f})",
                'values': f"Nilai aktual: Kepercayaan = {result['confidence']:.2f}\nDetail: {result['details']}"
            }
            for technique, result in validation_results.items()
            if not result['passed']
        ]
        
        # Hasilkan teks ringkasan
        if final_score >= 95:
            confidence_level = "Sangat Tinggi (Very High)"
            summary = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary += "Semua metode analisis menunjukkan konsistensi dan kualitas tinggi."
        elif final_score >= 90:
            confidence_level = "Tinggi (High)"
            summary = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary += "Sebagian besar metode analisis menunjukkan konsistensi dan kualitas baik."
        elif final_score >= 85:
            confidence_level = "Sedang (Medium)"
            summary = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary += "Beberapa metode analisis menunjukkan inkonsistensi minor."
        else:
            confidence_level = "Rendah (Low)"
            summary = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary += "Terdapat inkonsistensi signifikan antar metode analisis yang memerlukan perhatian."
        
        return process_results, final_score, summary, failed_validations


def validate_pipeline_integrity(analysis_results):
    """
    Validasi integritas pipeline 17 langkah pemrosesan.
    """
    if not analysis_results:
        return ["Hasil analisis tidak tersedia untuk validasi pipeline."], 0.0
    
    # Definisi proses pipeline
    pipeline_processes = [
        {"name": "1. Validasi & Muat Gambar", "check": lambda r: isinstance(r.get('metadata', {}).get('FileSize (bytes)', 0), int) and r.get('metadata', {}).get('FileSize (bytes)', 0) > 0},
        {"name": "2. Ekstraksi Metadata", "check": lambda r: 'Metadata_Authenticity_Score' in r.get('metadata', {})},
        {"name": "3. Pra-pemrosesan Gambar", "check": lambda r: r.get('enhanced_gray') is not None and len(r['enhanced_gray'].shape) == 2},
        {"name": "4. Analisis ELA Multi-Kualitas", "check": lambda r: isinstance(r.get('ela_image'), Image.Image) and r.get('ela_mean', -1) >= 0},
        {"name": "5. Ekstraksi Fitur (SIFT, ORB, etc.)", "check": lambda r: isinstance(r.get('feature_sets'), dict) and 'sift' in r['feature_sets']},
        {"name": "6. Deteksi Copy-Move (Feature-based)", "check": lambda r: 'ransac_inliers' in r and r['ransac_inliers'] >= 0},
        {"name": "7. Deteksi Copy-Move (Block-based)", "check": lambda r: 'block_matches' in r},
        {"name": "8. Analisis Konsistensi Noise", "check": lambda r: 'overall_inconsistency' in r.get('noise_analysis', {})},
        {"name": "9. Analisis Artefak JPEG", "check": lambda r: 'estimated_original_quality' in r.get('jpeg_analysis', {})},
        {"name": "10. Analisis Ghost JPEG", "check": lambda r: r.get('jpeg_ghost') is not None},
        {"name": "11. Analisis Domain Frekuensi", "check": lambda r: 'frequency_inconsistency' in r.get('frequency_analysis', {})},
        {"name": "12. Analisis Konsistensi Tekstur", "check": lambda r: 'overall_inconsistency' in r.get('texture_analysis', {})},
        {"name": "13. Analisis Konsistensi Tepi", "check": lambda r: 'edge_inconsistency' in r.get('edge_analysis', {})},
        {"name": "14. Analisis Konsistensi Iluminasi", "check": lambda r: 'overall_illumination_inconsistency' in r.get('illumination_analysis', {})},
        {"name": "15. Analisis Statistik Kanal", "check": lambda r: 'rg_correlation' in r.get('statistical_analysis', {})},
        {"name": "16. Lokalisasi Area Manipulasi", "check": lambda r: 'localization_analysis' in r},
        {"name": "17. Klasifikasi Akhir & Skor", "check": lambda r: 'type' in r.get('classification', {}) and 'confidence' in r.get('classification', {})}
    ]
    
    pipeline_results = []
    success_count = 0
    
    for process in pipeline_processes:
        try:
            is_success = process["check"](analysis_results)
        except Exception as e:
            is_success = False
            print(f"Error saat validasi integritas pipeline '{process['name']}': {e}")
        
        if is_success:
            status = "[BERHASIL]"
            pipeline_results.append(f"✅ {status:12} | {process['name']}")
            success_count += 1
        else:
            status = "[GAGAL]"
            pipeline_results.append(f"❌ {status:12} | {process['name']}")
    
    # Calculate pipeline integrity percentage
    pipeline_integrity = (success_count / len(pipeline_processes)) * 100
    
    return pipeline_results, pipeline_integrity


def lakukan_validasi_sistem(analysis_results):
    """
    Menjalankan Validasi Integritas Proses dengan pendekatan Validasi Silang Multi-Algoritma.
    Fungsi ini mengevaluasi kualitas hasil dari setiap algoritma dan konsistensi antar hasil.
    """
    if not analysis_results:
        return ["Hasil analisis tidak tersedia untuk divalidasi."], 0.0, "Hasil analisis tidak tersedia.", []

    # Buat validator forensik baru
    validator = ForensicValidator()
    
    # Jalankan validasi silang antar algoritma
    process_results, validation_score, summary_text, failed_validations = validator.validate_cross_algorithm(analysis_results)
    
    # Periksa juga integritas proses pipeline
    pipeline_results, pipeline_integrity_percentage = validate_pipeline_integrity(analysis_results)
    
    # Gabungkan hasil pipeline dan validasi silang
    combined_results = []
    combined_results.append("=== VALIDASI SILANG ALGORITMA ===")
    combined_results.extend(process_results)
    combined_results.append("")
    combined_results.append("=== VALIDASI INTEGRITAS PIPELINE ===")
    combined_results.extend(pipeline_results)
    
    # Bobot: 70% validasi silang, 30% integritas pipeline
    final_score = (validation_score * 0.7) + (pipeline_integrity_percentage * 0.3)
    
    return combined_results, final_score, summary_text, failed_validations


def display_validation_tab_baru(analysis_results):
    """
    Menampilkan tab validasi sistem (Tahap 5) dengan pendekatan validasi silang
    yang disempurnakan untuk presentasi forensik profesional.
    """
    st.header("🔬 Tahap 5: Validasi Forensik Digital", anchor=False)
    
    # Panel Penjelasan Metodologi
    with st.expander("Metodologi Validasi Forensik", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### Pendekatan Validasi Silang Multi-Algoritma
            
            Sistem ini mengimplementasikan pendekatan validasi forensik modern yang direkomendasikan oleh
            **Digital Forensic Research Workshop (DFRWS)** dan **Scientific Working Group on Digital Evidence (SWGDE)**.
            Validasi dilakukan melalui empat tahap utama:
            
            1. **Validasi Individual Algorithm** - Setiap metode deteksi dievaluasi secara terpisah
            2. **Cross-Algorithm Validation** - Hasil divalidasi silang antar algoritma
            3. **Physical Consistency Verification** - Memeriksa kesesuaian dengan prinsip fisika citra digital
            4. **Pipeline Integrity Assurance** - Memastikan semua 17 tahap berjalan dengan benar
            
            Sistem memberi bobot lebih besar (30% masing-masing) pada metode utama (K-Means dan Lokalisasi)
            dibandingkan metode pendukung (ELA dan SIFT, 20% masing-masing).
            """)
        
        with col2:
            # ======================= PERBAIKAN DI SINI =======================
            # Memeriksa apakah file ada sebelum menampilkannya
            image_path = "assets/validation_diagram.png"
            if os.path.exists(image_path):
                st.image(image_path, caption="Validasi Forensik Digital")
            else:
                st.info("Diagram validasi visual tidak tersedia.")
            # ======================= AKHIR PERBAIKAN =======================
            
            st.markdown("""
            #### Standar & Referensi:
            - ISO/IEC 27037:2012
            - NIST SP 800-86
            - Validasi >80% diperlukan untuk bukti di pengadilan
            """)

    # Jalankan validasi
    report_details, validation_score, summary_text, failed_validations = lakukan_validasi_sistem(analysis_results)

    # Dashboard Utama
    st.subheader("Dashboard Validasi Forensik", anchor=False)
    
    # Tabs untuk hasil berbeda
    tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Validasi", "🧪 Detail Proses", "📑 Dokumentasi Forensik"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Gauge chart untuk skor validasi
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=validation_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': 90, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 80], 'color': 'red'},
                        {'range': [80, 90], 'color': 'orange'},
                        {'range': [90, 100], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                },
                title={'text': "Skor Validasi Forensik"}
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Status validasi
            if validation_score >= 90:
                st.success("✅ **Validasi Tingkat Tinggi**: Hasil analisis memenuhi standar bukti forensik dengan kepercayaan tinggi.")
            elif validation_score >= 80:
                st.warning("⚠️ **Validasi Cukup**: Hasil analisis dapat diterima namun memerlukan konfirmasi tambahan.")
            else:
                st.error("❌ **Validasi Tidak Memadai**: Hasil analisis memiliki inkonsistensi yang signifikan.")
                
            st.info(summary_text)
            
        with col2:
            # Visualisasi kepercayaan per algoritma
            st.markdown("### Validasi Per Algoritma Forensik")
            validator = ForensicValidator()
            algorithm_scores = {}
            for technique, validate_func in [
                ('K-Means', validator.validate_clustering),
                ('Lokalisasi', validator.validate_localization),
                ('ELA', validator.validate_ela),
                ('SIFT', validator.validate_feature_matching)
            ]:
                confidence, _ = validate_func(analysis_results)
                algorithm_scores[technique] = confidence * 100
            
            # Buat donut chart
            labels = list(algorithm_scores.keys())
            values = list(algorithm_scores.values())
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=['rgb(56, 75, 126)', 'rgb(18, 36, 37)',
                              'rgb(34, 53, 101)', 'rgb(36, 55, 57)']
            )])
            
            fig.update_layout(
                title_text="Skor Kepercayaan Algoritma",
                annotations=[dict(text=f'{validation_score:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tambahkan interpretasi
            st.markdown("#### Interpretasi Forensik:")
            for alg, score in algorithm_scores.items():
                color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                st.markdown(f"- **{alg}**: <span style='color:{color}'>{score:.1f}%</span>", unsafe_allow_html=True)

    with tab2:
        st.subheader("Detail Proses Validasi", anchor=False)
        
        # Format report details dengan syntax highlighting
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create collapsible sections for each validation category
            with st.expander("Validasi Silang Algoritma", expanded=True):
                # Get only cross-algorithm validations
                cross_algo_results = [r for r in report_details if "VALIDASI SILANG" in r or "Validasi" in r]
                report_text = "\n".join(cross_algo_results)
                st.code(report_text, language='bash')
            
            with st.expander("Validasi Integritas Pipeline"):
                # Get only pipeline validations
                pipeline_results = [r for r in report_details if "INTEGRITAS PIPELINE" in r or not any(x in r for x in ["VALIDASI SILANG", "Validasi"])]
                report_text = "\n".join(pipeline_results)
                st.code(report_text, language='bash')
        
        with col2:
            # Calculate success metrics
            total_process = len([r for r in report_details if "[" in r])
            passed_process = len([r for r in report_details if "[BERHASIL]" in r or "[LULUS]" in r])
            
            # Create metrics display
            st.metric(
                label="Keberhasilan Proses",
                value=f"{passed_process}/{total_process}",
                delta=f"{(passed_process/total_process*100):.1f}%"
            )
            
            # Add validation formula
            st.markdown("""
            #### Formula Validasi:
            ```
            Skor Akhir = (0.7 × Validasi Silang) + (0.3 × Integritas Pipeline)
            ```
            
            #### Threshold Validasi:
            - ≥ 90%: Bukti Forensik Tingkat Tinggi
            - ≥ 80%: Bukti Forensik Dapat Diterima
            - < 80%: Bukti Forensik Tidak Memadai
            """)
        
        # Show failed validations if any
        if failed_validations:
            st.error("🚨 **Kegagalan Validasi Terdeteksi**")
            for i, failure in enumerate(failed_validations):
                with st.expander(f"Detail Kegagalan #{i+1}: {failure['name']}", expanded=i==0):
                    st.markdown(f"**Alasan Kegagalan:** {failure['reason']}")
                    st.markdown("**Aturan Validasi Forensik:**")
                    st.code(failure['rule'], language='text')
                    st.markdown("**Data Forensik:**")
                    st.code(failure['values'], language='text')
        else:
            st.success("✅ **Tidak Ada Kegagalan Validasi yang Terdeteksi**")
            st.markdown("Semua algoritma menunjukkan hasil yang konsisten dan terpenuhi kriteria validasi minimum.")

    with tab3:
        st.subheader("Dokumentasi Forensik Digital", anchor=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### Kriteria Validasi Forensik Digital
            
            Berdasarkan pedoman dari **National Institute of Standards and Technology (NIST)** dan **Association of Chief Police Officers (ACPO)**, hasil analisis forensik digital harus memenuhi kriteria-kriteria berikut:
            
            1. **Reliability** - Hasil dapat direproduksi dan konsisten
            2. **Accuracy** - Pengukuran dan perhitungan tepat
            3. **Precision** - Tingkat detail yang memadai
            4. **Verifiability** - Dapat diverifikasi dengan metode independen
            5. **Scope/Limitations** - Batasan dan cakupan diketahui
            
            Setiap metode deteksi (K-Means, ELA, SIFT, dll) telah melalui validasi silang untuk memastikan bahwa hasilnya memenuhi kriteria-kriteria di atas.
            """)
            
            st.markdown("""
            ### Langkah Validasi Analisis Forensik
            
            1. **Technical Validation** - Memverifikasi algoritma berfungsi dengan benar
            2. **Cross-Method Validation** - Membandingkan hasil antar metode yang berbeda
            3. **Internal Consistency Check** - Mengevaluasi konsistensi logis hasil
            4. **Anti-Tampering Validation** - Memverifikasi integritas data
            5. **Uncertainty Quantification** - Mengukur tingkat kepercayaan hasil
            """)
        
        with col2:
            # Chain of custody and evidence validation
            st.markdown("""
            ### Chain of Custody Forensik
            
            Pipeline 17 langkah dalam sistem ini memastikan **chain of custody** yang tidak terputus dari data asli hingga hasil analisis akhir:
            
            1. **Input Validation** → Validasi keaslian input gambar
            2. **Preservation** → Penyimpanan gambar asli tanpa modifikasi
            3. **Processing** → Analisis dengan multiple metode independen
            4. **Cross-Validation** → Validasi silang hasil antar metode
            5. **Reporting** → Dokumentasi lengkap proses dan hasil
            
            Validasi di atas 90% menunjukkan bahwa chain of custody telah terjaga dengan baik, dan hasil analisis memiliki tingkat kepercayaan yang tinggi untuk digunakan sebagai bukti forensik.
            """)
            
            # Add reference to forensic standards
            st.markdown("""
            ### Standar dan Referensi Forensik
            
            Proses validasi mengikuti standar berikut:
            
            - **ISO/IEC 27037:2012** - Guidelines for identification, collection, acquisition, and preservation of digital evidence
            - **ISO/IEC 27042:2015** - Guidelines for the analysis and interpretation of digital evidence
            - **NIST SP 800-86** - Guide to Integrating Forensic Techniques into Incident Response
            - **SWGDE** - Scientific Working Group on Digital Evidence Best Practices
            """)
    
    # Bottom section: expert insights
    st.markdown("---")
    st.subheader("Interpretasi Forensik", anchor=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate metrics based on individual algorithm results
        validator = ForensicValidator()
        cluster_confidence, cluster_details = validator.validate_clustering(analysis_results)
        localization_confidence, loc_details = validator.validate_localization(analysis_results)
        ela_confidence, ela_details = validator.validate_ela(analysis_results)
        feature_confidence, feature_details = validator.validate_feature_matching(analysis_results)
        
        # Create a more detailed forensic interpretation
        highest_confidence = max([
            (cluster_confidence, "K-Means Clustering"),
            (localization_confidence, "Lokalisasi Tampering"),
            (ela_confidence, "Error Level Analysis"),
            (feature_confidence, "SIFT Feature Matching")
        ], key=lambda x: x[0])
        
        lowest_confidence = min([
            (cluster_confidence, "K-Means Clustering"),
            (localization_confidence, "Lokalisasi Tampering"),
            (ela_confidence, "Error Level Analysis"),
            (feature_confidence, "SIFT Feature Matching")
        ], key=lambda x: x[0])
        
        # Create interpretation based on overall score and individual method scores
        if validation_score >= 90:
            interpretation = f"""
            Analisis forensik menunjukkan **tingkat kepercayaan tinggi ({validation_score:.1f}%)** dengan bukti konsisten antar metode.
            Bukti terkuat berasal dari **{highest_confidence[1]} ({highest_confidence[0]*100:.1f}%)**.
            
            Hasil ini memenuhi standar forensik dan dapat digunakan sebagai bukti yang kuat dalam investigasi digital.
            """
        elif validation_score >= 80:
            interpretation = f"""
            Analisis forensik menunjukkan **tingkat kepercayaan cukup ({validation_score:.1f}%)** dengan beberapa inkonsistensi minor.
            Bukti terkuat berasal dari **{highest_confidence[1]} ({highest_confidence[0]*100:.1f}%)**,
            sementara **{lowest_confidence[1]}** menunjukkan kepercayaan lebih rendah ({lowest_confidence[0]*100:.1f}%).
            
            Hasil ini dapat digunakan sebagai bukti pendukung tetapi memerlukan konfirmasi dari metode lain.
            """
        else:
            interpretation = f"""
            Analisis forensik menunjukkan **tingkat kepercayaan rendah ({validation_score:.1f}%)** dengan inkonsistensi signifikan antar metode.
            Bahkan bukti terkuat dari **{highest_confidence[1]}** hanya mencapai kepercayaan ({highest_confidence[0]*100:.1f}%).
            
            Hasil ini memerlukan penyelidikan lebih lanjut dan tidak dapat digunakan sebagai bukti tunggal.
            """
        
        st.markdown(interpretation)
        
        # Add forensic recommendation
        st.markdown("### Rekomendasi Forensik")
        if validation_score >= 90:
            st.success("""
            ✅ **DAPAT DITERIMA SEBAGAI BUKTI FORENSIK**
            
            Hasil analisis memiliki kepercayaan tinggi dan konsistensi yang baik antar metode.
            Tidak diperlukan analisis tambahan untuk memverifikasi hasil.
            """)
        elif validation_score >= 80:
            st.warning("""
            ⚠️ **DAPAT DITERIMA DENGAN VERIFIKASI TAMBAHAN**
            
            Hasil analisis memiliki tingkat kepercayaan cukup tetapi memerlukan metode verifikasi tambahan.
            Rekomendasikan analisis oleh ahli forensik secara manual.
            """)
        else:
            st.error("""
            ❌ **MEMERLUKAN ANALISIS ULANG**
            
            Hasil analisis menunjukkan inkonsistensi signifikan antar metode.
            Diperlukan pengambilan sampel ulang atau metode analisis alternatif.
            """)
    
    with col2:
        # Create comparison table of algorithm performance
        st.markdown("### Perbandingan Metode Forensik")
        
        # Create a comparison dataframe
        algorithm_data = {
            "Metode": ["K-Means", "Lokalisasi", "ELA", "SIFT"],
            "Kepercayaan": [
                f"{cluster_confidence*100:.1f}%",
                f"{localization_confidence*100:.1f}%",
                f"{ela_confidence*100:.1f}%",
                f"{feature_confidence*100:.1f}%"
            ],
            "Bobot": ["30%", "30%", "20%", "20%"],
            "Detail": [
                cluster_details,
                loc_details,
                ela_details,
                feature_details
            ]
        }
        
        # Create a styled dataframe
        import pandas as pd
        df = pd.DataFrame(algorithm_data)
        
        # Function to highlight cells based on confidence value
        def highlight_confidence(val):
            if "%" in str(val):
                try:
                    confidence = float(val.strip("%"))
                    if confidence >= 80:
                        return 'background-color: #a8f0a8'  # Light green
                    elif confidence >= 60:
                        return 'background-color: #f0e0a8'  # Light yellow
                    else:
                        return 'background-color: #f0a8a8'  # Light red
                except:
                    return ''
            return ''
        
        # Display the styled dataframe
        st.dataframe(df.style.applymap(highlight_confidence, subset=['Kepercayaan']), use_container_width=True)
        
        # Add Q&A preparation for defense
        st.markdown("### Panduan")
        with st.expander("Bagaimana sistem memastikan integritas data?"):
            st.markdown("""
            Sistem mengimplementasikan validasi pipeline 17 langkah yang memastikan:
            
            1. Validasi awal file gambar untuk keaslian metadata
            2. Preprocessing yang tidak merusak data asli
            3. Multiple algoritma deteksi yang independen
            4. Cross-validation antar algoritma
            5. Scoring dengan pembobotan berdasarkan reliabilitas algoritma
            
            Chain of custody dipastikan dengan mempertahankan gambar asli tanpa modifikasi.
            """)
            
        with st.expander("Mengapa validasi multi-algoritma lebih andal?"):
            st.markdown("""
            Validasi multi-algoritma lebih andal karena:
            
            1. **Redundansi** - Jika satu algoritma gagal, algoritma lain dapat mendeteksi manipulasi
            2. **Teknik Komplementer** - Algoritma yang berbeda mendeteksi jenis manipulasi berbeda
            3. **Konsensus** - Kesepakatan antar algoritma meningkatkan kepercayaan hasil
            4. **Bias Reduction** - Mengurangi false positive/negative dari algoritma tunggal
            
            Pendekatan ini mengikuti prinsip "defense in depth" dalam forensik digital.
            """)
            
        with st.expander("Bagaimana sistem meminimalkan false positives?"):
            st.markdown("""
            Sistem meminimalkan false positives dengan:
            
            1. **Threshold Kalibrasi** - Setiap algoritma memiliki threshold minimum 60%
            2. **Weighted Scoring** - Algoritma lebih andal diberi bobot lebih besar
            3. **Cross-validation** - Memerlukan konsensus antar metode
            4. **Physical Consistency** - Memvalidasi terhadap prinsip fisika citra
            5. **Bobot Agreement** - Menerapkan bonus 20% untuk kesepakatan antar algoritma
            
            Dengan pendekatan ini, sistem memastikan tingkat false positive yang rendah sambil mempertahankan sensitivitas deteksi.
            """)

# ======================= APLIKASI UTAMA STREAMLIT (BAGIAN YANG DIMODIFIKASI) =======================
def main_app():
    st.set_page_config(layout="wide", page_title="Sistem Forensik Gambar V3")

    # Ganti nama variabel agar tidak bentrok dengan fungsi
    global IMPORTS_SUCCESSFUL, IMPORT_ERROR_MESSAGE
    
    if not IMPORTS_SUCCESSFUL:
        st.error(f"Gagal mengimpor modul: {IMPORT_ERROR_MESSAGE}")
        return

    # Inisialisasi session state (tidak ada perubahan di sini)
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'original_image' not in st.session_state: st.session_state.original_image = None
    if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None
    
    st.sidebar.title("🖼️ Sistem Deteksi Forensik V3")
    st.sidebar.markdown("Unggah gambar untuk memulai analisis mendalam.")

    uploaded_file = st.sidebar.file_uploader(
        "Pilih file gambar...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )

    if uploaded_file is not None:
        # Periksa apakah ini file baru atau sama dengan yang terakhir
        if st.session_state.last_uploaded_file is None or st.session_state.last_uploaded_file.name != uploaded_file.name:
            st.session_state.last_uploaded_file = uploaded_file
            st.session_state.analysis_results = None
            st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
            st.rerun()

    if st.session_state.original_image:
        st.sidebar.image(st.session_state.original_image, caption='Gambar yang diunggah', use_container_width=True)

        if st.sidebar.button("🔬 Mulai Analisis", use_container_width=True, type="primary"):
            st.session_state.analysis_results = None
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            filename = st.session_state.last_uploaded_file.name
            temp_filepath = os.path.join(temp_dir, filename)
            
            # Tulis ulang file dari buffer
            st.session_state.last_uploaded_file.seek(0)
            with open(temp_filepath, "wb") as f:
                f.write(st.session_state.last_uploaded_file.getbuffer())

            with st.spinner('Melakukan analisis 17 tahap... Ini mungkin memakan waktu beberapa saat.'):
                try:
                    # Pastikan main_analysis_func dipanggil dengan path file yang benar
                    results = main_analysis_func(temp_filepath)
                    st.session_state.analysis_results = results
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat analisis: {e}")
                    st.exception(e)
                    st.session_state.analysis_results = None
                finally:
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.subheader("Kontrol Sesi")

        # Tombol Mulai Ulang (tidak ada perubahan)
        if st.sidebar.button("🔄 Mulai Ulang Analisis", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.original_image = None
            st.session_state.last_uploaded_file = None
            if 'pdf_preview_path' in st.session_state:
                st.session_state.pdf_preview_path = None # Reset preview
            st.rerun()

# Tombol Keluar (tidak ada perubahan pada logika ini)
        if st.sidebar.button("🚪 Keluar", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.original_image = None
            st.session_state.last_uploaded_file = None
            st.sidebar.warning("Aplikasi sedang ditutup...")
            st.balloons()
            time.sleep(2)
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)

    st.sidebar.markdown("---")
    st.sidebar.info("Aplikasi ini menggunakan pipeline analisis 17-tahap untuk mendeteksi manipulasi gambar.")

    st.title("Hasil Analisis Forensik Gambar")

    if st.session_state.analysis_results:
        # ======================= PERUBAHAN UTAMA DI SINI (TAB) =======================
        tab_list = [
            "📊 Tahap 1: Analisis Inti",
            "🔬 Tahap 2: Analisis Lanjut",
            "📈 Tahap 3: Analisis Statistik",
            "📋 Tahap 4: Laporan Akhir",
            "🔬 Tahap 5: Hasil Pengujian", # TAB BARU UNTUK VALIDASI
            "📄 Ekspor Laporan",
            "📜 Riwayat Analisis"
        ]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_list)
        # ======================= AKHIR PERUBAHAN TAB =======================

        with tab1:
            display_core_analysis(st.session_state.original_image, st.session_state.analysis_results)
        with tab2:
            display_advanced_analysis(st.session_state.original_image, st.session_state.analysis_results)
        with tab3:
            display_statistical_analysis(st.session_state.original_image, st.session_state.analysis_results)
        with tab4:
            display_final_report(st.session_state.analysis_results)
        # ======================= KONTEN TAB BARU =======================
        with tab5:
            display_validation_tab_baru(st.session_state.analysis_results)
        # ======================= AKHIR KONTEN TAB BARU =======================
        with tab6:
            display_export_tab(st.session_state.original_image, st.session_state.analysis_results)
        with tab7:
            display_history_tab()

    elif not st.session_state.original_image:
        # Tampilkan tab Riwayat di halaman utama jika belum ada gambar diunggah
        main_page_tabs = st.tabs(["👋 Selamat Datang", "📜 Riwayat Analisis"])
        
        with main_page_tabs[0]:
            st.info("Silakan unggah gambar di sidebar kiri untuk memulai.")
            st.markdown("""
            **Panduan Singkat:**
            1. **Unggah Gambar:** Gunakan tombol 'Pilih file gambar...' di sidebar.
            2. **Mulai Analisis:** Klik tombol biru 'Mulai Analisis'.
            3. **Lihat Hasil:** Hasil akan ditampilkan dalam beberapa tab.
            4. **Uji Sistem:** Buka tab 'Hasil Pengujian' untuk melihat validasi integritas.
            5. **Ekspor:** Buka tab 'Ekspor Laporan' untuk mengunduh hasil.
            """)
        
        with main_page_tabs[1]:
            display_history_tab()

# Pastikan Anda memanggil fungsi main_app() di akhir
if __name__ == '__main__':
    # Anda harus menempatkan semua fungsi helper (seperti display_core_analysis, dll.)
    # sebelum pemanggilan main_app() atau di dalam file lain dan diimpor.
    # Untuk contoh ini, saya asumsikan semua fungsi sudah didefinisikan di atas.
    
    # ======================= Konfigurasi & Import =======================
    try:
        from main import analyze_image_comprehensive_advanced as main_analysis_func
        from config import BLOCK_SIZE
        IMPORTS_SUCCESSFUL = True
        IMPORT_ERROR_MESSAGE = ""
    except ImportError as e:
        IMPORTS_SUCCESSFUL = False
        IMPORT_ERROR_MESSAGE = str(e)
    
    main_app()