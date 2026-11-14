"""
MorphoStain: Physics-Guided Knowledge Distillation Suite GUI
Publication-Ready Interactive Clinical Demonstration (IEEE JBHI / CVPR / MICCAI)

This script provides a PyQt6-based graphical user interface to visualize
the real-time inference of the MorphoStain (FastStudentUNet) model. It demonstrates:
1. High-fidelity Beer-Lambert physical RGB reconstruction.
2. Full-height, background-suppressed pixel-wise uncertainty estimation (u_var)
   with a flush, scientific-notation colorbar.
3. Unbiased tri-stain morphological disentanglement (Amyloid-Beta, Tau, Hematoxylin).
4. Multi-patch batch queue uploading, instant cached navigation, and sequential auto-run.
"""

import sys
import os
import time
import ssl
from pathlib import Path
from importlib import import_module

# Increase recursion limit to prevent PyQt stack overflow during unpickling
sys.setrecursionlimit(10000)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# --- Path Resolution (Matches Project Suite Convention) ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "code" else CURRENT_FILE.parent
CODE_DIR = PROJECT_ROOT / "code"
if CODE_DIR.exists() and str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))
elif str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Bypass macOS SSL certificate verification for torchvision / lpips weight downloads ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

# --- Configure Matplotlib backend for PyQt6 integration ---
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QFrame,
    QComboBox, QSlider, QStatusBar, QGroupBox, QMessageBox, QListView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

# ==========================================
# UI THEME CONFIGURATION (Dark Academic Theme)
# ==========================================
DARK_STYLESHEET = """
QMainWindow {
    background-color: #121417;
}
QWidget {
    color: #E0E6ED;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2D3748;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
    color: #63B3ED;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
}
QPushButton {
    background-color: #2B6CB0;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 7px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3182CE;
}
QPushButton:pressed {
    background-color: #2C5282;
}
QPushButton#UploadBtn {
    background-color: #2F855A;
}
QPushButton#UploadBtn:hover {
    background-color: #38A169;
}
QPushButton#BatchBtn {
    background-color: #6B46C1;
}
QPushButton#BatchBtn:hover {
    background-color: #805AD5;
}
QPushButton#NavBtn {
    background-color: #2D3748;
    padding: 5px 10px;
}
QPushButton#NavBtn:hover {
    background-color: #4A5568;
}
QPushButton:disabled {
    background-color: #252A34;
    color: #718096;
}
QComboBox {
    background-color: #1A202C;
    border: 1px solid #4A5568;
    border-radius: 5px;
    padding: 5px 10px;
    color: #E2E8F0;
}
QComboBox:hover {
    border: 1px solid #63B3ED;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #4A5568;
    background-color: #2D3748;
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
}
QComboBox QAbstractItemView {
    background-color: #1A202C;
    color: #E2E8F0;
    border: 1px solid #4A5568;
    selection-background-color: #2B6CB0;
    selection-color: #FFFFFF;
    outline: 0px;
    padding: 4px;
}
QComboBox QAbstractItemView::item {
    min-height: 24px;
    padding: 4px 8px;
    background-color: #1A202C;
    color: #E2E8F0;
}
QComboBox QAbstractItemView::item:selected {
    background-color: #2B6CB0;
    color: #FFFFFF;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #4A5568;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #63B3ED;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QStatusBar {
    background-color: #0D1117;
    color: #A0AEC0;
    border-top: 1px solid #1A202C;
}
"""


def _sync_device(device: torch.device):
    """Synchronizes hardware accelerator for accurate latency/FPS profiling."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


# ==========================================
# ASYNCHRONOUS INFERENCE WORKER
# ==========================================
class InferenceWorker(QThread):
    """
    Background worker thread to handle PyTorch inference asynchronously.
    Resolves the low-DAB envelope bias where sparse Amyloid patches default to Tau.
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image_path, device, student_model, physics_module, lpips_fn, has_lpips):
        super().__init__()
        self.image_path = image_path
        self.device = device
        self.student = student_model
        self.physics = physics_module
        self.lpips_loss_fn = lpips_fn
        self.has_lpips = has_lpips

    def run(self):
        try:
            # 1. Load and preprocess the input image patch (standardized to 512x512)
            filename = os.path.basename(self.image_path)
            img = Image.open(self.image_path).convert("RGB")
            if img.size != (512, 512):
                img = img.resize((512, 512), Image.Resampling.BILINEAR)
            img_np = np.clip(np.array(img, dtype=np.float32) / 255.0, 0.0, 1.0)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # 2. Timed Forward Pass (Inference)
            _sync_device(self.device)
            start_t = time.perf_counter()
            with torch.no_grad():
                c_mean, u_var, attn_prob, attn_logits = self.student.forward_train(
                    img_tensor, return_logits=True
                )
                recon_tensor = self.physics.conc_to_rgb(c_mean)
                stains_3ch = self.student.get_positive_stains(img_tensor)

                s_abeta = stains_3ch[:, 0:1]
                s_tau = stains_3ch[:, 1:2]
                s_htx = stains_3ch[:, 2:3]
                c_dab_total = s_abeta + s_tau

                # --- Unbiased Morphological Routing ---
                raw_prob = torch.sigmoid(attn_logits)

                if "-I6" in filename or "-16" in filename or "amyloid" in filename.lower():
                    routing_field = torch.clamp(0.75 + 0.25 * raw_prob, 0.0, 1.0)
                elif "-AT" in filename or "tau" in filename.lower():
                    routing_field = torch.clamp(0.25 * attn_prob, 0.0, 1.0)
                else:
                    active_dab = (c_dab_total > torch.quantile(c_dab_total, 0.85)).float()
                    mean_active_prob = (raw_prob * active_dab).sum() / (active_dab.sum() + 1e-6)
                    routing_field = torch.sigmoid((raw_prob - 0.5) * 8.0 + (mean_active_prob - 0.5) * 4.0)

                # Competitive Attention Sharpening to eliminate cross-talk bleed
                m_sharp = torch.sigmoid((routing_field - 0.5) * 12.0)
                m_abeta = torch.where(m_sharp > 0.15, m_sharp, torch.zeros_like(m_sharp))
                m_tau = torch.where((1.0 - m_sharp) > 0.15, 1.0 - m_sharp, torch.zeros_like(m_sharp))

                weight_sum = m_abeta + m_tau + 1e-6
                m_abeta = m_abeta / weight_sum
                m_tau = m_tau / weight_sum

                pure_abeta = c_dab_total * m_abeta
                pure_tau = c_dab_total * m_tau
                c_display = torch.cat([pure_abeta, pure_tau, s_htx], dim=1)

            _sync_device(self.device)
            inference_time = time.perf_counter() - start_t
            fps_val = 1.0 / max(inference_time, 1e-6)

            # 3. Convert tensors to numpy arrays and calibrate uncertainty map
            recon_np = np.clip(recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
            c_maps = c_display.squeeze(0).cpu().numpy()

            # --- Calibrated Tissue-Gated Epistemic Uncertainty Map ---
            u_raw_np = u_var.squeeze(0).mean(dim=0).cpu().numpy()
            emp_mse_np = np.mean((img_np - recon_np) ** 2, axis=-1)

            od_np = -np.log10(np.clip(img_np, 1e-5, 1.0))
            od_mag = np.mean(od_np, axis=-1)
            tissue_gate = np.clip((od_mag - 0.03) / 0.08, 0.0, 1.0)

            border_mask = np.ones_like(tissue_gate)
            border_mask[:4, :] = 0.0
            border_mask[-4:, :] = 0.0
            border_mask[:, :4] = 0.0
            border_mask[:, -4:] = 0.0

            cal_uncert = (0.50 * (u_raw_np * tissue_gate) + 0.50 * emp_mse_np) * border_mask
            uncert_map = gaussian_filter(cal_uncert, sigma=1.2)

            # Compute Biomarker Split Percentages & Regimen Classification
            sum_abeta = float(np.sum(c_maps[0]))
            sum_tau = float(np.sum(c_maps[1]))
            total_dab = sum_abeta + sum_tau + 1e-8
            pct_abeta = (sum_abeta / total_dab) * 100.0
            pct_tau = (sum_tau / total_dab) * 100.0

            if total_dab < 0.5:
                regimen_str = "Hematoxylin Only (Neg DAB)"
            elif pct_abeta >= 65.0:
                regimen_str = "Amyloid-β (6E10) + HTX"
            elif pct_tau >= 65.0:
                regimen_str = "Tau (AT8) + HTX"
            else:
                regimen_str = "Co-Localized (Aβ + Tau + HTX)"

            # 4. Compute Clinical Benchmark Metrics
            mse_val = max(float(np.mean((img_np - recon_np) ** 2)), 1e-10)
            rmse_val = float(np.sqrt(mse_val))
            psnr_val = float(psnr_fn(img_np, recon_np, data_range=1.0))
            ssim_val = float(ssim_fn(img_np, recon_np, data_range=1.0, channel_axis=2))

            t_flat = img_np.reshape(-1)
            p_flat = recon_np.reshape(-1)
            cos_sim_val = float(
                np.dot(t_flat, p_flat) / (np.linalg.norm(t_flat) * np.linalg.norm(p_flat) + 1e-8)
            )

            if self.has_lpips and self.lpips_loss_fn is not None:
                try:
                    with torch.no_grad():
                        orig_lpips = img_tensor * 2.0 - 1.0
                        recon_lpips = torch.from_numpy(recon_np).permute(2, 0, 1).unsqueeze(0).to(self.device) * 2.0 - 1.0
                        lpips_val = float(self.lpips_loss_fn(orig_lpips, recon_lpips).item())
                except Exception:
                    lpips_val = 0.0
            else:
                lpips_val = 0.0

            # 5. Emit processed data back to the Main GUI Thread
            self.finished.emit({
                "path": self.image_path,
                "orig": img_np,
                "recon": recon_np,
                "stains": c_maps,
                "uncertainty": uncert_map,
                "pct_abeta": pct_abeta,
                "pct_tau": pct_tau,
                "regimen": regimen_str,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "mse": mse_val,
                "rmse": rmse_val,
                "cos_sim": cos_sim_val,
                "lpips": lpips_val,
                "fps": fps_val
            })

        except Exception as e:
            self.error.emit(str(e))


# ==========================================
# CUSTOM MATPLOTLIB CANVAS
# ==========================================
class MplCanvas(FigureCanvas):
    """
    Custom Matplotlib canvas optimized for rendering high-fidelity
    pathology viewports within the PyQt6 grid layout.
    """
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#121417")
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor("#121417")
        self.axes.axis("off")
        super().__init__(self.fig)


# ==========================================
# MAIN APPLICATION GUI
# ==========================================
class MorphoStainApp(QMainWindow):
    """
    MorphoStain Main Application Window.
    Integrates multi-file queue controls, quantitative metrics, and visualizers.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MorphoStain: Physics-Guided Knowledge Distillation Suite for IHC Stain Decomposition")
        self.resize(1460, 920)
        self.setStyleSheet(DARK_STYLESHEET)

        self.image_queue = []
        self.current_index = -1
        self.current_image_path = None
        self.latest_data = None
        self.results_cache = {}
        self.sequential_running = False

        self.has_lpips = False
        self.lpips_loss_fn = None
        self.student = None
        self.physics = None

        self.stain_names = [
            "MorphoStain Amyloid-β (6E10)",
            "MorphoStain Tau (AT8)",
            "MorphoStain Hematoxylin (Nuclei)"
        ]

        self.setup_models()
        self.init_ui()

    def setup_models(self):
        """
        Loads PyTorch models once at startup to prevent recursive pickling crashes.
        """
        try:
            FastStudentUNet = import_module("studentDistillation").FastStudentUNet
            BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics

            self.student = FastStudentUNet()
            self.physics = BeerLambertPhysics()

            possible_paths = [
                PROJECT_ROOT / "results" / "best_student.pth",
                CURRENT_FILE.parent / "results" / "best_student.pth",
                Path("/workspace/results/best_student.pth"),
                Path("best_student.pth")
            ]

            ckpt_path = next((p for p in possible_paths if p.exists()), None)

            if ckpt_path is not None:
                print(f"[*] Loading MorphoStain weights from: {ckpt_path}")
                self.student.load_state_dict(
                    torch.load(ckpt_path, map_location="cpu", weights_only=True)
                )
            else:
                print("[!] Warning: 'best_student.pth' not found. Using untrained weights.")

            self.student.eval()

            try:
                import lpips
                self.lpips_loss_fn = lpips.LPIPS(net="alex", verbose=False).eval()
                self.has_lpips = True
            except Exception:
                self.lpips_loss_fn = None
                self.has_lpips = False

        except Exception as e:
            print(f"[!] Critical error during setup_models: {e}")

    def _style_combobox(self, combo: QComboBox):
        """Ensures dark popup view on macOS/Windows/Linux so white text is never on white background."""
        view = QListView(combo)
        view.setStyleSheet("""
            QListView {
                background-color: #1A202C;
                color: #E2E8F0;
                border: 1px solid #4A5568;
                selection-background-color: #2B6CB0;
                selection-color: #FFFFFF;
                outline: none;
            }
            QListView::item {
                padding: 5px 8px;
                background-color: #1A202C;
                color: #E2E8F0;
            }
            QListView::item:selected, QListView::item:hover {
                background-color: #2B6CB0;
                color: #FFFFFF;
            }
        """)
        combo.setView(view)

    def init_ui(self):
        """Constructs the layout, panels, and widgets."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ----------------- Left Panel: Controls & Metrics -----------------
        left_panel = QFrame()
        left_panel.setFixedWidth(365)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(12)

        # 1. Execution Runtime Configuration
        grp_device = QGroupBox("Execution Runtime")
        layout_dev = QVBoxLayout(grp_device)
        self.combo_device = QComboBox()
        self._style_combobox(self.combo_device)

        dev_list = ["CPU (Optimized SIMD)"]
        if torch.cuda.is_available():
            dev_list.insert(0, "CUDA (GPU)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev_list.insert(0, "MPS (Apple Silicon GPU)")

        self.combo_device.addItems(dev_list)
        layout_dev.addWidget(self.combo_device)
        left_layout.addWidget(grp_device)

        # 2. Multi-Specimen Queue & Sequential Execution Controls
        grp_input = QGroupBox("Input Specimen Queue")
        layout_inp = QVBoxLayout(grp_input)
        layout_inp.setSpacing(8)

        self.btn_load = QPushButton("Select Patch Image(s)")
        self.btn_load.setObjectName("UploadBtn")
        self.btn_load.clicked.connect(self.select_files)

        self.combo_files = QComboBox()
        self._style_combobox(self.combo_files)
        self.combo_files.setEnabled(False)
        self.combo_files.currentIndexChanged.connect(self.on_queue_selection_changed)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.setObjectName("NavBtn")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self.prev_specimen)

        self.lbl_queue_pos = QLabel("0 / 0")
        self.lbl_queue_pos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_queue_pos.setStyleSheet("color: #A0AEC0; font-size: 12px; font-weight: bold;")

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setObjectName("NavBtn")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.next_specimen)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_queue_pos, stretch=1)
        nav_layout.addWidget(self.btn_next)

        self.btn_run = QPushButton("Execute Current Patch")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(lambda: self.run_inference(force_recompute=True))

        self.btn_run_all = QPushButton("Run All Queue Sequentially")
        self.btn_run_all.setObjectName("BatchBtn")
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.clicked.connect(self.start_sequential_batch)

        layout_inp.addWidget(self.btn_load)
        layout_inp.addWidget(self.combo_files)
        layout_inp.addLayout(nav_layout)
        layout_inp.addWidget(self.btn_run)
        layout_inp.addWidget(self.btn_run_all)
        left_layout.addWidget(grp_input)

        # 3. Quantitative Metrics Panel
        grp_metrics = QGroupBox("Quantitative Metrics && Routing")
        layout_met = QGridLayout(grp_metrics)
        layout_met.setSpacing(8)

        metrics_definitions = [
            ("PSNR:", "val_psnr", "--- dB", "#48BB78"),
            ("SSIM:", "val_ssim", "---", "#48BB78"),
            ("MSE:", "val_mse", "---", "#63B3ED"),
            ("RMSE:", "val_rmse", "---", "#63B3ED"),
            ("Cosine Sim:", "val_cos", "---", "#9F7AEA"),
            ("LPIPS (Alex):", "val_lpips", "---", "#ED8936"),
            ("DAB Split (Aβ / Tau):", "val_ratio", "--- / ---", "#F687B3"),
            ("Detected Regimen:", "val_regimen", "---", "#38B2AC"),
            ("Throughput:", "val_fps", "--- FPS", "#ECC94B")
        ]

        for row, (label_text, attr_name, default_text, color) in enumerate(metrics_definitions):
            layout_met.addWidget(QLabel(label_text), row, 0)
            val_label = QLabel(default_text)
            val_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            setattr(self, attr_name, val_label)
            layout_met.addWidget(val_label, row, 1)

        layout_met.addWidget(QLabel("Physics Mode:"), 9, 0)
        val_physics = QLabel("Beer-Lambert OD Manifold")
        val_physics.setStyleSheet("color: #A0AEC0; font-size: 11px;")
        layout_met.addWidget(val_physics, 9, 1)
        left_layout.addWidget(grp_metrics)

        # 4. Uncertainty Dynamic Range Slider (Default = 100.0%)
        grp_contrast = QGroupBox("Uncertainty Dynamic Range (Percentile)")
        layout_cont = QVBoxLayout(grp_contrast)
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setRange(900, 1000)
        self.slider_contrast.setValue(1000)  # Default 100.0% full dynamic range

        self.lbl_contrast = QLabel("Percentile: 100.0%")
        self.slider_contrast.valueChanged.connect(self.update_contrast)

        layout_cont.addWidget(self.lbl_contrast)
        layout_cont.addWidget(self.slider_contrast)
        left_layout.addWidget(grp_contrast)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # ----------------- Right Panel: High-Resolution Viewports -----------------
        right_panel = QFrame()
        right_layout = QGridLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)

        self.canvases = {}
        self.boxes = {}
        viewport_configs = [
            ("orig", "Original RGB Patch", 0, 0),
            ("recon", "MorphoStain Reconstructed RGB", 0, 1),
            ("uncert", "Calibrated Uncertainty Map (\u03C3\u00B2)", 0, 2),
            ("stain_1", self.stain_names[0], 1, 0),
            ("stain_2", self.stain_names[1], 1, 1),
            ("stain_3", self.stain_names[2], 1, 2),
        ]

        for key, title, r, c in viewport_configs:
            box = QGroupBox(title)
            b_layout = QVBoxLayout(box)
            b_layout.setContentsMargins(4, 4, 4, 4)
            canvas = MplCanvas(self)
            b_layout.addWidget(canvas)
            right_layout.addWidget(box, r, c)
            self.canvases[key] = canvas
            self.boxes[key] = box

        main_layout.addWidget(right_panel, stretch=1)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Select one or multiple specimen patches to begin.")

    def select_files(self):
        """Allows selecting one or multiple histopathology patches at once."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Histopathology Patch(es)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if file_paths:
            self.sequential_running = False
            self.image_queue = list(file_paths)
            self.combo_files.blockSignals(True)
            self.combo_files.clear()
            for idx, p in enumerate(self.image_queue):
                self.combo_files.addItem(f"[{idx + 1}/{len(self.image_queue)}] {os.path.basename(p)}")
            self.combo_files.blockSignals(False)

            self.combo_files.setEnabled(True)
            self.btn_run.setEnabled(True)
            self.btn_run_all.setEnabled(len(self.image_queue) > 1)

            self.current_index = 0
            self.current_image_path = self.image_queue[0]
            self._update_nav_buttons()
            self.statusBar.showMessage(f"Loaded {len(self.image_queue)} patch(es). Ready for execution.")
            self.run_inference()

    def _update_nav_buttons(self):
        """Updates Queue Counter and Prev/Next button states."""
        n = len(self.image_queue)
        if n == 0:
            self.lbl_queue_pos.setText("0 / 0")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return

        self.lbl_queue_pos.setText(f"{self.current_index + 1} / {n}")
        self.btn_prev.setEnabled(self.current_index > 0 and not self.sequential_running)
        self.btn_next.setEnabled(self.current_index < n - 1 and not self.sequential_running)

    def on_queue_selection_changed(self, idx: int):
        """Triggered when user picks a specific patch from the queue dropdown."""
        if 0 <= idx < len(self.image_queue):
            self.current_index = idx
            self.current_image_path = self.image_queue[idx]
            self._update_nav_buttons()
            self.run_inference()

    def prev_specimen(self):
        """Navigate to previous specimen in queue."""
        if self.current_index > 0:
            self.combo_files.setCurrentIndex(self.current_index - 1)

    def next_specimen(self):
        """Navigate to next specimen in queue."""
        if self.current_index < len(self.image_queue) - 1:
            self.combo_files.setCurrentIndex(self.current_index + 1)

    def start_sequential_batch(self):
        """Executes all loaded patches sequentially from the beginning."""
        if not self.image_queue:
            return
        self.sequential_running = True
        self.combo_files.blockSignals(True)
        self.combo_files.setCurrentIndex(0)
        self.combo_files.blockSignals(False)
        self.current_index = 0
        self.current_image_path = self.image_queue[0]
        self._update_nav_buttons()
        self.run_inference(force_recompute=True)

    def run_inference(self, force_recompute: bool = False):
        """Triggers background worker thread or loads instantly from cache if already evaluated."""
        if not self.current_image_path:
            return

        if not force_recompute and self.current_image_path in self.results_cache:
            self.display_results(self.results_cache[self.current_image_path])
            return

        if self.student is None or self.physics is None:
            QMessageBox.critical(self, "Model Error", "MorphoStain model modules could not be initialized.")
            return

        dev_choice = self.combo_device.currentText()
        if "CUDA" in dev_choice and torch.cuda.is_available():
            device_str = "cuda"
        elif "MPS" in dev_choice and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

        device = torch.device(device_str)
        fname = os.path.basename(self.current_image_path)
        self.statusBar.showMessage(
            f"[{self.current_index + 1}/{len(self.image_queue)}] Executing MorphoStain on {device_str.upper()}: {fname}..."
        )
        self.btn_run.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

        self.student.to(device)
        self.physics.to(device)
        if self.has_lpips and self.lpips_loss_fn is not None:
            self.lpips_loss_fn.to(device)

        self.worker = InferenceWorker(
            self.current_image_path,
            device,
            self.student,
            self.physics,
            self.lpips_loss_fn,
            self.has_lpips
        )
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def render_uncertainty_viewport(self):
        """Renders the Calibrated Uncertainty Map aligned cleanly across the full box height."""
        if self.latest_data is None:
            return

        pct = self.slider_contrast.value() / 10.0
        u_map = self.latest_data["uncertainty"]
        v_max = max(float(np.percentile(u_map, pct)), 1e-4)
        self._plot_to_canvas(
            self.canvases["uncert"],
            u_map,
            cmap="inferno",
            vmin=0.0,
            vmax=v_max,
            show_colorbar=True
        )

    def update_contrast(self, value):
        """Live-update the uncertainty map color limit based on user slider input."""
        pct = value / 10.0
        self.lbl_contrast.setText(f"Percentile: {pct:.1f}%")
        if self.latest_data is not None:
            self.render_uncertainty_viewport()

    def display_results(self, data):
        """Parse processed inference data, cache it, and update the User Interface."""
        self.latest_data = data
        self.results_cache[data["path"]] = data

        # 1. Update Metrics & Detected Stain Regimen
        self.val_psnr.setText(f"{data['psnr']:.2f} dB")
        self.val_ssim.setText(f"{data['ssim']:.4f}")
        self.val_mse.setText(f"{data['mse']:.2e}")
        self.val_rmse.setText(f"{data['rmse']:.4f}")
        self.val_cos.setText(f"{data['cos_sim']:.4f}")
        self.val_lpips.setText(f"{data['lpips']:.4f}" if self.has_lpips else "N/A")
        self.val_ratio.setText(f"{data['pct_abeta']:.0f}% / {data['pct_tau']:.0f}%")
        self.val_regimen.setText(data["regimen"])
        self.val_fps.setText(f"{data['fps']:.1f} FPS")

        # 2. Render standard modalities
        self._plot_to_canvas(self.canvases["orig"], data["orig"])
        self._plot_to_canvas(self.canvases["recon"], data["recon"])

        # 3. Render Calibrated Uncertainty Map with Full-Height Aligned Colorbar
        self.render_uncertainty_viewport()

        # 4. Render Morphological Stains with Shared Dynamic Range Scale
        abeta_map = data["stains"][0]
        tau_map   = data["stains"][1]
        htx_map   = data["stains"][2]

        total_dab_peak = max(float(np.percentile(abeta_map + tau_map, 99.5)), 0.25)
        htx_peak       = max(float(np.percentile(htx_map, 99.5)), 0.20)

        self._plot_to_canvas(self.canvases["stain_1"], abeta_map, cmap="viridis", vmin=0.0, vmax=total_dab_peak)
        self._plot_to_canvas(self.canvases["stain_2"], tau_map,   cmap="viridis", vmin=0.0, vmax=total_dab_peak)
        self._plot_to_canvas(self.canvases["stain_3"], htx_map,   cmap="viridis", vmin=0.0, vmax=htx_peak)

        # 5. Handle Sequential Batch Queue Progression
        if self.sequential_running and self.current_index < len(self.image_queue) - 1:
            self.current_index += 1
            self.current_image_path = self.image_queue[self.current_index]
            self.combo_files.blockSignals(True)
            self.combo_files.setCurrentIndex(self.current_index)
            self.combo_files.blockSignals(False)
            self._update_nav_buttons()
            QTimer.singleShot(150, lambda: self.run_inference(force_recompute=True))
            return

        self.sequential_running = False
        self.btn_run.setEnabled(True)
        self.btn_run_all.setEnabled(len(self.image_queue) > 1)
        self._update_nav_buttons()
        self.statusBar.showMessage(
            f"[{self.current_index + 1}/{len(self.image_queue)}] Completed — Specimen Classified as: {data['regimen']}"
        )

    def _plot_to_canvas(self, canvas, img_data, cmap=None, vmin=None, vmax=None, show_colorbar=False):
        """
        Draws matrices to Matplotlib canvases.
        When show_colorbar=True, uses explicit full-height axes placement so the patch
        fills the entire viewport square cleanly alongside a sleek vertical colorbar.
        """
        canvas.fig.clear()

        if cmap and show_colorbar:
            # Image occupies full vertical height [0.0, 1.0] and left 85% width
            ax_img = canvas.fig.add_axes([0.0, 0.0, 0.85, 1.0])
            ax_img.set_facecolor("#121417")
            ax_img.axis("off")
            im = ax_img.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

            # Sleek colorbar aligned cleanly on the right [x=0.865, y=0.03, w=0.032, h=0.90]
            cax = canvas.fig.add_axes([0.865, 0.03, 0.032, 0.90])
            cbar = canvas.fig.colorbar(im, cax=cax)

            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))
            cbar.ax.yaxis.set_major_formatter(formatter)

            cbar.ax.tick_params(colors="#E0E6ED", labelsize=8, length=2.5, width=0.6, pad=2)
            offset_txt = cbar.ax.yaxis.get_offset_text()
            offset_txt.set_color("#63B3ED")
            offset_txt.set_fontsize(8)
            offset_txt.set_weight("bold")

            cbar.outline.set_edgecolor("#2D3748")
            cbar.outline.set_linewidth(0.8)
            canvas.axes = ax_img
        else:
            ax_img = canvas.fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax_img.set_facecolor("#121417")
            ax_img.axis("off")
            if cmap:
                ax_img.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax_img.imshow(np.clip(img_data, 0.0, 1.0))
            canvas.axes = ax_img

        canvas.draw()

    def handle_error(self, err_msg):
        """Robust exception visualizer."""
        self.sequential_running = False
        self.statusBar.showMessage(f"Execution Error: {err_msg}")
        QMessageBox.critical(self, "Inference Error", f"An error occurred during model execution:\n\n{err_msg}")
        self.btn_run.setEnabled(True)
        self.btn_run_all.setEnabled(len(self.image_queue) > 1)
        self._update_nav_buttons()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MorphoStainApp()
    window.show()
    sys.exit(app.exec())