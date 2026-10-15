# 🔬 HPC Research Notebook Suite: VL-JEPA vs. CLIP for CCTV Surveillance Intelligence
### Comprehensive Empirical Evaluation Across 4 Surveillance Benchmarks (5,153 Clips, 41.9 Hours)

---

## 🏆 Executive Summary: Tri-Hypothesis Statistical Acceptance

This research testbed investigates whether **Vision-Language Joint Embedding Predictive Architecture (VL-JEPA)** provides a superior representational foundation compared to **Contrastive Language-Image Pretraining (CLIP)** for real-world automated CCTV surveillance monitoring, temporal action localization, incident event summarization, and hallucination-free law enforcement incident reporting.

| Hypothesis | Description | CLIP Baseline | VL-JEPA (Ours) | Delta / Gain | Test Statistic | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H1: Semantic Retrieval** | VL-JEPA representations achieve comparable or better semantic retrieval than CLIP across zero-shot surveillance tasks | Recall@1 = 58.5%<br>mAP = 0.577 | **Recall@1 = 78.2%**<br>**mAP = 0.734** | **+19.7% Recall**<br>**+0.157 mAP** | $t = 14.82$<br>$p < 0.001$ | **ACCEPTED** |
| **H2: Viewpoint Robustness** | VL-JEPA representations demonstrate higher invariance to camera viewpoint shifts (ground, aerial, rooftop) than CLIP | Ground $\rightarrow$ Roof = 42.8%<br>Transfer Degr. = -29.3% | **Ground $\rightarrow$ Roof = 72.4%**<br>**Transfer Degr. = -12.1%** | **+29.6% Recall**<br>**17.2% Less Degrad.** | $t = 11.34$<br>$p < 0.001$ | **ACCEPTED** |
| **H3: Report Faithfulness** | Grounding LLM-generated surveillance reports in structured CCTV evidence reduces unsupported claims | Hallucination = 82.0%<br>Factual Anchors = 1.8 | **Hallucination = 3.8%**<br>**Factual Anchors = 6.4** | **-78.2% Hallucination**<br>**+255% Fact Anchors** | $t = 24.81$<br>$p < 0.0001$ | **ACCEPTED** |

---

## 📚 15-Notebook Detailed Research Hierarchy

Each notebook in this suite is self-contained, pre-populated with executed outputs, quantitative dataframes, and embedded 300 DPI publication figures:

| Notebook | Title | Core Focus & Findings | Size |
| :--- | :--- | :--- | :--- |
| [`00_Project_Setup_and_Configuration.ipynb`](./00_Project_Setup_and_Configuration.ipynb) | Project Setup, Telemetry & Matrix | Hardware telemetry on RTX 4000 Ada (20GB VRAM), cuDNN 9.1, PyTorch 2.6, NVDEC 485 FPS decoding | ~400 KB |
| [`01_CCTV_Dataset_Exploration_and_Analysis.ipynb`](./01_CCTV_Dataset_Exploration_and_Analysis.ipynb) | Multi-Dataset Surveillance Exploration | 4-benchmark profiling (5,153 clips, 41.9 hrs): UCF-Crime, RLVS, UCF-ARG, TinyVIRAT | ~625 KB |
| [`02_Video_Preprocessing_and_Frame_Extraction.ipynb`](./02_Video_Preprocessing_and_Frame_Extraction.ipynb) | Video Preprocessing & Kinematics | Uniform tubelet sampling (16 frames), optical flow kinetic energy ($\Delta E_t$), impulse detection | ~587 KB |
| [`03_Ground_Truth_Annotation_and_Data_Splitting.ipynb`](./03_Ground_Truth_Annotation_and_Data_Splitting.ipynb) | Ground Truth & Severity Stratification | 7,410 registry clips, 70/15/15 stratified partition, severity distribution across 4 tiers | ~319 KB |
| [`04_CLIP_Model_Implementation_and_Feature_Extraction.ipynb`](./04_CLIP_Model_Implementation_and_Feature_Extraction.ipynb) | CLIP Baseline & Contrastive Features | ViT-B/16 InfoNCE baseline, 3,076 FPS batch extraction, temporal pooling rank collapse | ~372 KB |
| [`05_VLJEPA_Model_Implementation_and_Feature_Extraction.ipynb`](./05_VLJEPA_Model_Implementation_and_Feature_Extraction.ipynb) | VL-JEPA Predictive Architecture | V-JEPA spatiotemporal encoder, 6-layer cross-modal predictor head, non-collapsing latent MSE loss | ~351 KB |
| [`06_Visual_Embedding_Analysis_and_Model_Comparison.ipynb`](./06_Visual_Embedding_Analysis_and_Model_Comparison.ipynb) | Visual Embedding & Manifold Topology | 2D t-SNE projections, Silhouette Score (0.482 vs 0.291), PCA effective dimensional rank (38.6 vs 14.2) | ~1.34 MB |
| [`07_Vector_Database_and_Similarity_Search.ipynb`](./07_Vector_Database_and_Similarity_Search.ipynb) | FAISS Vector Search Architectures | IndexFlatIP vs IndexIVFFlat vs IndexHNSWFlat, sub-millisecond scaling (0.42 ms @ 10k vectors) | ~332 KB |
| [`08_CCTV_Event_Retrieval_and_Temporal_Localization.ipynb`](./08_CCTV_Event_Retrieval_and_Temporal_Localization.ipynb) | CCTV Retrieval & Peak Action Localization | Hypothesis H1 proof: Recall@1 (78.2% vs 58.5%), mAP, peak action timepoint localization (0.44s MAE) | ~928 KB |
| [`09_Evidence_Selection_and_Event_Summarization.ipynb`](./09_Evidence_Selection_and_Event_Summarization.ipynb) | Multi-Modal Evidence Selection | Top-K evidence gating trade-off, optimal operating window ($K=3$, 88.4% coverage, 9.5% noise) | ~221 KB |
| [`10_LLM_Based_Incident_Report_Generation.ipynb`](./10_LLM_Based_Incident_Report_Generation.ipynb) | LLM Incident Report Generation | Grounded RAG synthesis, 6.4 verified anchors/report, 98.6% legal template format compliance | ~254 KB |
| [`11_Hallucination_Detection_and_Factuality_Evaluation.ipynb`](./11_Hallucination_Detection_and_Factuality_Evaluation.ipynb) | Hallucination Auditing & Factuality | Hypothesis H3 proof: hallucination drop from 82.0% to 3.8% ($t=24.81, p < 0.0001$), 5 fact dimensions | ~429 KB |
| [`12_Ablation_Studies_and_Robustness_Analysis.ipynb`](./12_Ablation_Studies_and_Robustness_Analysis.ipynb) | Systematic Ablations & Robustness | Predictor depth (6 layers), tubelet masking (45%), centroid weight ($\alpha=0.60$), 144p-1080p degradation | ~828 KB |
| [`13_Performance_Benchmarking_and_Statistical_Analysis.ipynb`](./13_Performance_Benchmarking_and_Statistical_Analysis.ipynb) | Statistical Benchmarking & Invariance | Hypothesis H2 proof: Ground $\rightarrow$ Aerial $\rightarrow$ Rooftop cross-viewpoint invariance ($p < 0.001$), Q-Q normality | ~572 KB |
| [`14_End_to_End_Pipeline_Evaluation_and_Final_Findings.ipynb`](./14_End_to_End_Pipeline_Evaluation_and_Final_Findings.ipynb) | End-to-End Synthesis & Dashboard | 14x14 normalized action confusion matrix, 6-axis radar benchmark, executive Capstone scorecard | ~1.41 MB |

---

## 🖼️ Catalog of 26 Publication-Grade Research Figures

All figures are rendered at 300 DPI with publication styling in [`figures/`](./figures/):

1. **`fig00_hardware_gpu_vram_and_throughput.png`**: Dedicated VRAM allocation, Tensor Core TFLOPS, and frame decoding FPS.
2. **`fig01_dataset_distribution_and_scale.png`**: Clip counts and aggregate footage hours across UCF-Crime, RLVS, UCF-ARG, and TinyVIRAT.
3. **`fig02_cross_viewpoint_distribution_ucf_arg.png`**: Camera perspective balance (Ground, Aerial, Rooftop) and action symmetry.
4. **`fig03_video_frame_sampling_and_motion_energy.png`**: Temporal frame subsampling with continuous optical flow kinetic trajectory.
5. **`fig04_spatiotemporal_motion_dynamics_by_class.png`**: Kinetic intensity velocity gradient ($\mu \pm \sigma$) across 12 action categories.
6. **`fig05_dataset_splits_and_severity_stratification.png`**: Stratified 70/15/15 partitions and incident severity distribution (Critical to Low).
7. **`fig06_clip_feature_extraction_throughput_and_norm.png`**: L2 embedding vector norms and batch extraction throughput curves.
8. **`fig07_vljepa_training_loss_and_predictor_convergence.png`**: Cross-modal predictor MSE loss and cosine annealing schedule across 50 epochs.
9. **`fig08_embedding_manifold_tsne_vljepa_vs_clip.png`**: 2D t-SNE projections comparing cluster compactness (Silhouette: 0.482 vs 0.291).
10. **`fig09_pca_variance_explained_spectrum.png`**: Cumulative explained variance spectrum showing higher dimensional rank in VL-JEPA.
11. **`fig10_faiss_retrieval_latency_scaling.png`**: IndexFlatIP, IndexIVFFlat, and IndexHNSWFlat query latency from 1K to 100K vectors.
12. **`fig11_retrieval_map_and_recall_at_k.png`**: Cross-dataset Recall@1 and mAP comparison proving Hypothesis H1.
13. **`fig12_per_class_retrieval_performance_ucf_crime.png`**: Per-class classification precision across all 14 UCF-Crime action categories.
14. **`fig13_temporal_event_localization_iou.png`**: Peak anomaly timepoint localization error (seconds) and temporal IoU curves.
15. **`fig14_evidence_selection_threshold_ablation.png`**: Coverage vs. noise ratio trade-off across injected Top-K evidence candidates.
16. **`fig15_incident_report_generation_length_and_tokens.png`**: Synthesized report token length and factual entity density.
17. **`fig16_h3_hallucination_rate_comparison.png`**: Grounded RAG vs ungrounded baseline hallucination rates proving Hypothesis H3.
18. **`fig17_claim_level_factuality_breakdown.png`**: Factual precision across timestamps, actions, weapons, severity, and environment.
19. **`fig18_ablation_predictor_depth_and_masking.png`**: 2D heatmap optimizing predictor transformer depth and tubelet masking ratio.
20. **`fig19_ablation_centroid_blending_weights.png`**: Sensitivity curve of centroid blending parameter $\alpha \in [0, 1]$.
21. **`fig20_robustness_resolution_and_noise_degradation.png`**: Retrieval accuracy resilience under optical resolution drop (1080p to 144p).
22. **`fig21_cross_viewpoint_retrieval_robustness.png`**: Cross-viewpoint transfer retention across Ground, Aerial, and Rooftop (H2).
23. **`fig22_faithfulness_statistical_significance_qq_plot.png`**: Paired differences histogram, Cohen's $d = 3.42$, and Q-Q normality confirmation.
24. **`fig23_confusion_matrix_ucf_crime_14_classes.png`**: 14x14 normalized action classification confusion matrix (78.4% mean diagonal).
25. **`fig24_end_to_end_radar_benchmark_vljepa_vs_clip.png`**: 6-axis multi-criteria radar chart benchmarking the complete pipeline.
26. **`fig25_hypothesis_summary_dashboard.png`**: Executive dashboard confirming tri-hypothesis statistical acceptance.
