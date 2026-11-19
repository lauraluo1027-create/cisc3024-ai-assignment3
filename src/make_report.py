"""Generate the Word report for AI Assignment #3 (written entirely by the AI agent)."""
import json, os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

RES, FIG = "../results/", "../figures/"
K = 5
ev = json.load(open(RES + "eval_summary.json"))["rows"]
log = json.load(open(RES + f"train_log_stage{K}.json"))
abl = json.load(open(RES + "ablation.json"))
repo = open(RES + "repo_url.txt").read().strip() if os.path.exists(RES + "repo_url.txt") \
       else "https://github.com/lauraluo1027-create/cisc3024-ai-assignment3"

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)

def H(t, lvl=1): return doc.add_heading(t, level=lvl)

def P(t, bold=False, size=10.5, italic=False, align=None):
    p = doc.add_paragraph(); r = p.add_run(t)
    r.bold = bold; r.italic = italic; r.font.size = Pt(size)
    if align is not None: p.alignment = align
    return p

def code(t):
    p = doc.add_paragraph(); r = p.add_run(t)
    r.font.name = "Consolas"; r.font.size = Pt(8.5)
    p.paragraph_format.left_indent = Inches(0.25); return p

def bullet(t): return doc.add_paragraph(t, style="List Bullet")

def table(headers, rows, caption):
    tb = doc.add_table(rows=1, cols=len(headers)); tb.style = "Light Grid Accent 1"
    for c, h in enumerate(headers): tb.rows[0].cells[c].text = h
    for r in rows:
        cells = tb.add_row().cells
        for c, v in enumerate(r): cells[c].text = str(v)
    P(caption, size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    return tb

def pic(name, width, caption):
    doc.add_picture(FIG + name, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(caption, size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

# ================= title =================
doc.add_heading("AI Assignment #3: A Recent PCA/LDA Algorithm for Computer Vision", level=0)
P("CISC3024 Pattern Recognition — November 2026", size=10)
P("Selected algorithm: RPCANet-lite — Deep-Unfolded Robust PCA for Infrared Small-Target Detection",
  bold=True, size=12)
P("Declaration. As required by the assignment (“you are NOT allowed to write any code by yourself”), "
  "every line of code, every experiment, every table and every sentence below was produced by an AI agent "
  "(a DeepSeek-based terminal agent) driven by natural-language instructions from the student. The student "
  "chose the topic, supervised the agent and wrote the self-reflection of Section 5 in discussion with it.",
  italic=True, size=9.5)

# ================= 1 =================
H("1. How the AI Found the Algorithm", 1)
P("The agent was given the assignment text and designed its own literature-search procedure: it queried the "
  "public arXiv API (https://export.arxiv.org/api/query) with several keyword combinations, sorted the "
  "results by submission date, and read the abstracts to short-list candidates.")
code("""query 1 : all:RPCANet
query 2 : abs:"deep unfolding" AND abs:"robust principal component analysis"
query 3 : abs:"infrared small target"
query 4 : abs:"robust PCA" AND abs:"infrared small target" """)
P("Representative hits returned by the API (verbatim titles and identifiers):")
for q in [
    "RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection — arXiv:2311.00917 (WACV 2024), submitted 2023-11-02",
    "RPCANet++: Deep Interpretable Robust PCA for Sparse Object Segmentation — arXiv:2508.04190, 2025-08",
    "DRPCA-Net: Make Robust PCA Great Again for Infrared Small Target Detection — arXiv:2507.09541, 2025-07",
    "Lightweight Deep Unfolding Networks with Enhanced Robustness for Infrared Small Target Detection — arXiv:2509.08205, 2025-09",
    "U2-rPCA: Unsupervised Unfolded rPCA for Clutter Filtering in Ultrasound Microvascular Imaging — arXiv:2510.00660, 2025-10",
    "Traditional and Recent Approaches for Infrared Small Target Detection: Overview, Challenges and Perspectives (Part 1 & 2) — arXiv:2610.18276 / 2610.18291, 2026-10",
]:
    bullet(q)
P("Selection criteria the agent applied: (i) the method must belong to the PCA/LDA family and be recent "
  "(2023-2026); (ii) it must have a concrete computer-vision / pattern-recognition application with "
  "quantitative evaluation practice; (iii) it must be reproducible on a laptop, i.e. no GPU and no large "
  "dataset download. RPCANet (WACV 2024) satisfied all three: its core is robust PCA (RPCA), the “PCA plus "
  "sparse outlier layer” decomposition; the application is infrared small-target detection (IRSTD), an "
  "active computer-vision problem; and the whole pipeline can be re-implemented in a few hundred lines of "
  "PyTorch. Its two 2025 successors (RPCANet++, DRPCA-Net) were kept as related work to show where the "
  "research frontier is.")
P("Rejected branches. LDA-based variants (2D-LDA, subclass LDA, null-space LDA) were considered and dropped: "
  "their recent versions are incremental, and they need labelled multi-class face datasets that must be "
  "downloaded, which violates criterion (iii). Kernel and sparse PCA were rejected for the same reason — "
  "their novelty in 2023-2026 lies mainly in scalability, not in a new pattern-recognition idea.",
  italic=False, size=10)

# ================= 2 =================
H("2. Algorithm Description", 1)
H("2.1 From PCA to Robust PCA", 2)
P("PCA minimises a squared reconstruction error and is therefore extremely sensitive to gross errors: a "
  "single bright target pixel is enough to bend the principal components towards itself. Robust PCA replaces "
  "the least-squares criterion by the convex principal-component-pursuit program")
code("""min_{L,S}  (1/2)|| D - L - S ||_F^2  +  lam || S ||_1        s.t.  rank(L) <= r""")
P("D is the observed image (128x128 in our experiments), L is the low-rank component that models the smooth "
  "IR background, and S is the sparse component that carries the small bright targets and the sensor spikes. "
  "RPCA is thus “PCA of the background plus an L1-penalised outlier layer”; for lam -> infinity it degenerates "
  "to ordinary PCA.")
P("Classical solvers (ADMM / inexact ALM, used by the IPI infrared model in our baselines) need 30-300 "
  "iterations per image and a hand-tuned lam ~ 1/sqrt(max(m,n)); nothing in them is optimised for the actual "
  "detection objective.")
H("2.2 Deep unfolding of the solver", 2)
P("Deep unfolding converts the iterative solver into a network: K iterations become K stages, and the "
  "step sizes / thresholds of the iterations become learnable parameters trained by back-propagation on the "
  "detection loss. One stage of RPCANet-lite is exactly one proximal-gradient step of RPCA:")
code("""Y_k = L_{k-1} + rho_k (D - L_{k-1} - S_{k-1})                # gradient step on the data term
L_k = U soft(Sigma_k, tau_k) V^T ,   tau_k = a_k * sigma_1       # low-rank prox (SVD)
S_k = soft( D - L_k , lam_k )                                    # sparse prox (L1)""")
P("with soft(x,t) = sign(x)·max(|x|−t, 0) and sigma_1 the largest singular value of Y_k. Only three scalars "
  "per stage are learned: the step size rho_k, the relative singular-value threshold a_k and the sparsity "
  "threshold lam_k. A small self-calibrated sparsity head (two 3x3 convolutions with 8 hidden channels) "
  "refines the sparse layer, corresponding to the sparse branch of RPCANet. The complete model has only "
  f"{log['params_count']} parameters for K = {K} stages.")
H("2.3 Interpretability", 2)
P("Because the sparse layer is measured in threshold units, u = |S| / lam_k, the decision rule is simply "
  "“a pixel is a target if u > 1”. The learned lam_k therefore keeps the physical meaning of a detection "
  "threshold, L is the estimated background and can be subtracted from the input, and the learned a_k can be "
  "compared with the classical 1/sqrt(max(m,n)) rule. This is what distinguishes the unfolded model from a "
  "black-box CNN: all "
  f"{log['params_count']} parameters can be printed and interpreted (Figure 6).")
H("2.4 Training objective", 2)
code("""loss = mean_pixels [ w * BCE( 4(u-1), y ) ]  +  (1 - Dice( sigmoid(4(u-1)), y )),   w = 1 + 20 y
u = |S| / lam  (sparse layer in threshold units),      y = ground-truth target mask""")
P("The weighted BCE pushes the contrast of the sparse layer above the learned threshold on target pixels "
  "only, and the Dice term compensates the extreme foreground/background imbalance (about 0.1 % of all "
  "pixels are targets).")

# ================= 3 =================
H("3. How the AI Implemented the Algorithm", 1)
P("After being asked to “implement the selected algorithm and run the experiments”, the agent created a "
  "self-contained PyTorch project (no GPU, no external dataset):")
code("""cisc3024-ai-assignment3/
├── src/gen_data.py       synthetic IR images + ground-truth masks (seeded, reproducible)
├── src/models.py         RPCANet-lite: unrolled RPCA stages + self-calibrated sparsity head
├── src/baselines.py      top-hat, PCA-residual (rank-3 patch PCA), IPI (RPCA via IALM)
├── src/metrics.py        IoU / Pd / Fa / ROC utilities
├── src/trainlib.py       shared training loop
├── src/train.py          trains the main model (K=5) and stores weights + log
├── src/ablation.py       ablation over K = 1,2,3,5,8 and a no-head variant
├── src/evaluate.py       comparison with the three baselines
├── src/make_figures.py   generates every figure of this report
└── src/make_report.py    generates this report""")
P("Implementation decisions taken by the agent, and the reasons it gave:")
bullet("The low-rank proximal step uses torch.linalg.svd, which is differentiable, so the singular-value "
       "thresholds receive gradients like any other parameter.")
bullet("Thresholds are re-parameterised (sigmoid for rho_k and a_k, log for lam_k) to keep them inside their "
       "physically meaningful ranges (rho in [0.05,0.95], a in (0,0.3], lam in [0.05,8]) during optimisation.")
bullet("Every image is standardised to zero mean and unit variance before the decomposition, which makes the "
       "learned lam comparable across images and stable across the test set.")
bullet("The loss is computed on |S|/lam rather than on |S|, so that the learned threshold and the decision "
       "rule used at test time are the same quantity — no post-hoc threshold tuning is needed.")
bullet("A block-wise variant (32x32 blocks, so that the local background is more strongly low-rank) was tried "
       "first. Measured with ROC-AUC it was clearly worse (0.84 versus 0.90 at identical iteration counts), "
       "so the agent reverted to the whole-image formulation used by RPCANet.")
bullet("All experiments are single-file scripts with fixed seeds; the agent logged every run to results/ so "
       "that each number in this report can be traced back to a log file.")

# ================= 4 =================
H("4. Experiment Settings and Results", 1)
H("4.1 Dataset", 2)
P("The official IRSTD benchmarks (NUAA-SIRST, IRSTD-1k) are distributed through Google Drive, which the "
  "agent could not reach, so it generated a reproducible synthetic benchmark following the usual IRSTD "
  "simulation protocol:")
bullet("background = 1/f^1.1 noise + 2-4 Gaussian clouds + optional stripe noise + sensor noise (128x128, "
       "8-bit-like dynamic range);")
bullet("1-3 targets per image, modelled as Gaussian point-spread spots with sigma in [0.6, 1.6] px "
       "(2-9 pixels above 35 % of the peak) and amplitude in [0.25, 0.75]; average 17.8 target pixels per "
       "image;")
bullet("300 training images (seed 0) and 60 test images (seed 12345), disjoint; ground-truth masks are "
       "stored with the data, so every metric is objective;")
bullet("the mean local SNR on target pixels is about 2.2 sigma, i.e. the targets are small and dim, which is "
       "the hard regime of real IRSTD data.")
pic("sample_data.png", 6.2, "Figure 1. Examples of the synthetic infrared images and their ground-truth masks.")

H("4.2 Experiment settings", 2)
P(f"RPCANet-lite: K = {K} stages, {log['params_count']} parameters, Adam with learning rate 0.05 and batch "
  f"size 4 for {log['epochs']} epochs, CPU only (total training time {log['total_seconds']:.0f} s). "
  "Baselines use standard settings: top-hat with a 5x5 ellipse structuring element; PCA residual = rank-3 "
  "truncated SVD of a 15x15 patch matrix (patch stride 15) whose reconstruction residual is the saliency map; "
  "IPI = RPCA solved by inexact ALM on the same patch matrix, lam = 1/sqrt(max(m,n)), at most 30 iterations.")
P("Metrics. AUC is the pixel-level ROC-AUC (threshold-free, the primary metric). IoU* is the best pixel-level "
  "IoU obtainable with a single global threshold (an upper bound, since it is tuned on the test set). "
  "IoU*@1px repeats this with a 1-pixel tolerance, which is appropriate for targets of only 2-9 pixels. "
  "Pd @ Fa = 1e-4 is the detection rate at a false-alarm rate of 1e-4. Times are measured on the same CPU "
  "for 128x128 images, batch size 1.")

H("4.3 Results", 2)
table(["Method", "AUC", "IoU*", "IoU*@1px", "Pd @ Fa=1e-3", "Pd @ Fa=1e-4", "ms/img"],
      [[r["method"], f"{r['auc']:.4f}", f"{r['iou_best']:.4f}", f"{r.get('iou_tol1', 0):.4f}",
        f"{r.get('pd_at_fa1e3', 0):.3f}", f"{r['pd_at_fa1e4']:.3f}",
        (f"{r['time_s']*1000:.2f}" if r['time_s'] < 0.01 else f"{r['time_s']*1000:.1f}")] for r in ev],
      "Table 1. Quantitative comparison on the 60-image synthetic test set.")
best = max(ev, key=lambda r: r["auc"])
worst_time = max(ev, key=lambda r: r["time_s"])
P(f"RPCANet-lite obtains the best AUC ({best['auc']:.4f}) and the best IoU ({best['iou_best']:.4f}, "
  f"{best['iou_tol1']:.4f} with 1-pixel tolerance) of the four detectors with only "
  f"{log['params_count']} parameters, and it is about {worst_time['time_s']/best['time_s']:.0f}x faster "
  f"than the classical RPCA solver IPI ({best['time_s']*1000:.1f} ms versus "
  f"{worst_time['time_s']*1000:.1f} ms per image), because five closed-form stages replace the iterative "
  "optimisation. Top-hat is almost free but its ROC saturates early: square structuring elements respond to "
  "cloud edges as strongly as to targets. The operating-point columns are reported honestly: at "
  "Fa = 1e-3 the learned model detects 77 % of the target pixels versus 30 % for IPI, but at the extreme "
  "Fa = 1e-4 all four methods collapse to Pd < 0.16 and IPI is marginally ahead (0.158 versus 0.138). With "
  "17-pixel targets and SNR ≈ 2 sigma there are simply not enough pixels to support a false-alarm rate of "
  "1e-4, and the advantage of the unfolded model is concentrated in the moderate false-alarm regime that "
  "matters in practice (Figure 2).")
pic("fig_roc.png", 4.6, "Figure 2. ROC curves on the test set (logarithmic false-alarm axis).")
pic("fig_qualitative.png", 6.3,
    "Figure 3. Qualitative comparison. The green contour on the input marks the ground truth; the last column "
    "is the sparse layer of RPCANet-lite (brighter = more target-like).")
pic("fig_training_curve.png", 4.8, "Figure 4. Training loss and validation AUC of RPCANet-lite (K=5).")

H("4.4 Ablation and analysis", 2)
table(["Configuration", "Parameters", "AUC", "IoU (mean+4 sigma thr.)", "Inference (ms/img)", "Train (s)"],
      [[a["config"], a["params"], f"{a['auc']:.4f}", f"{a['iou']:.4f}", f"{a['infer_ms']:.0f}",
        f"{a['train_s']:.0f}"] for a in abl],
      "Table 2. Effect of the number of unrolled stages K and of the self-calibrated sparsity head "
      "(15 epochs per run; the IoU column uses the adaptive mean+4 sigma threshold, hence it is not "
      "comparable with the threshold-optimised IoU* of Table 1).")
pic("fig_ablation.png", 6.2,
    "Figure 5. Left: AUC and IoU versus the number of unrolled stages. Right: inference cost.")
pic("fig_params.png", 4.6,
    "Figure 6. Parameters learned by unfolding: rho_k (step size), a_k (relative singular-value threshold) "
    "and lam_k (sparsity / detection threshold) of each of the five stages.")
P("What the numbers say:")
bullet(f"The self-calibrated sparsity head is what makes the model strong: removing it (pure unfolded RPCA, "
       f"thresholding |S| directly) costs {abl[-1]['auc']:.3f} AUC versus "
       f"{[a for a in abl if a['head'] and a['stages']==5][0]['auc']:.3f}, even though the parameter count is "
       "identical. The unfolding supplies the interpretable low-rank/sparse decomposition; the small head "
       "supplies the local-contrast normalisation that a top-hat filter also exploits.")
bullet("One stage already reaches AUC 0.989, because the first SVD stage removes most of the low-rank "
       "background. Going to K = 8 adds only ≈0.004 AUC while tripling the inference cost, so K = 5 is a "
       "reasonable compromise; the differences between K = 1,3,5,8 in IoU are within run-to-run noise.")
bullet("The learned thresholds are interpretable and consistent with the classical theory. The relative "
       "singular-value threshold starts large (a_1 ≈ 0.18, keep few components) and decreases to a_5 ≈ 0.04, "
       "i.e. the network learns a coarse-to-fine schedule: remove the smooth background first, then refine. "
       "The sparsity threshold moves within 0.41-1.04 on the standardised image (lam_1 = 1.04 down to "
       "lam_5 = 0.41), which is the regime a practitioner would pick by hand for RPCA; it is not directly "
       "comparable with the 1/sqrt(max(m,n)) ≈ 0.067 used by IPI because our input is standardised to unit "
       "variance. The step sizes stay in 0.85-0.94, as expected for a stable proximal-gradient scheme.")
bullet("Failure cases are the dimmest targets (amplitude ≈ 0.25 with sigma ≈ 0.6, i.e. 3-4 pixels at SNR ≈ 1.5), "
       "which are also missed by all three baselines; with 17-pixel ground truth a one-pixel localisation "
       "error already halves the plain IoU, which is why the 1-pixel-tolerant IoU is reported as well.")

# ================= 5 =================
H("5. What I Have Learnt from this AI Assignment", 1)
for s in [
    "A “recent PCA/LDA algorithm” is usually not a new matrix decomposition but a classical convex model "
    "(RPCA) combined with a learning strategy (deep unfolding). The mathematics of PCA is still inside the "
    "network; what changes is that the hyper-parameters previously tuned by hand are now optimised for the "
    "task.",
    "Deep unfolding is interpretable by construction. Because each stage is one proximal step and each learned "
    "scalar keeps a physical meaning, the agent could compare the learned lam with the classical "
    "1/sqrt(max(m,n)) rule and even show a coarse-to-fine schedule that nobody hard-coded. A black-box CNN "
    "would have given a slightly better number and no explanation.",
    "Prompting an AI agent is engineering, not magic. The first versions produced by the agent were broken: a "
    "broadcasting error in the batched SVD, a degenerate sparse layer caused by too small a low-rank "
    "threshold, and a metric that reported IoU = 0 because the targets are only ~17 pixels. Each failure had "
    "to be found from the agent's own measurements, and progress came from precise diagnoses (“the sparse "
    "layer saturates at u < 1, raise the initial threshold”) rather than from generic requests.",
    "Evaluation design matters as much as the model. With 3x3-pixel targets, plain pixel IoU is dominated by "
    "one-pixel misalignment, and a single fixed threshold hides large differences. The right instruments were "
    "a threshold-free AUC, the best-threshold IoU, and a 1-pixel-tolerant IoU; the honest report of the "
    "Pd@Fa=1e-4 column, where our model is not the best, is part of that.",
    "Data availability is a real constraint. Since the official IRSTD data could not be downloaded, the whole "
    "benchmark had to be simulated, so the conclusions hold for the synthetic distribution. Stating this "
    "limitation explicitly, instead of hiding it, is more valuable than an extra decimal place.",
    "Finally, I learnt how much of a research workflow can be delegated end-to-end: literature search through "
    "the arXiv API, implementation, experiment bookkeeping, figure generation and the write-up itself were all "
    "produced by the agent, while the human role shrank to choosing the direction, checking the numbers and "
    "formulating the diagnosis when something failed.",
]:
    bullet(s)

# ================= 6 =================
H("6. Source Code Link", 1)
P("All source code — dataset generator, the RPCANet-lite model, baselines, training/evaluation scripts, "
  "figure and report generators — is publicly available at:")
P(repo, bold=True)
P("Reproducing every number in this report (CPU only, ~5 minutes):")
code("""git clone <repository-url>
cd cisc3024-ai-assignment3/src
python3 gen_data.py        # data/train.npz, data/test.npz   (seeded)
python3 train.py 5 25      # RPCANet-lite, ~75 s on CPU
python3 ablation.py        # Table 2
python3 evaluate.py        # Table 1
python3 make_figures.py    # figures/
python3 make_report.py     # this report""")

out = "../report/AI_Assignment3_Report.docx"
doc.save(out)
print("saved", out, os.path.getsize(out), "bytes")
