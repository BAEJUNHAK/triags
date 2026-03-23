# TriaGS 시스템 분석 및 실험 가이드

## 1. 프로젝트 개요

**TriaGS (Triangulation-Guided Geometric Consistency for 3D Gaussian Splatting)**는 3D Gaussian Splatting의 기하학적 일관성을 멀티뷰 삼각측량(multi-view triangulation)을 통해 개선하는 방법론이다.

- **논문**: WACV 2026 accepted ([arXiv:2512.06269](https://arxiv.org/abs/2512.06269))
- **저자**: Quan Tran, Tuan Dang (University of Arkansas, Cognitive Robotics Laboratory)
- **핵심 성과**: DTU 데이터셋에서 평균 Chamfer Distance **0.50mm** (SOTA)

---

## 2. 풀고자 하는 문제

### 기존 3DGS의 한계
- 3D Gaussian Splatting은 **photometric loss만으로** 최적화되어 기하학적 불일치 발생
- 반투명한 Gaussian "floater"들이 공간에 떠다니며 올바른 이미지를 렌더링하지만 **실제 표면과 무관**
- 결과적으로 고품질 surface mesh 추출이 어려움

### 기존 해결 시도의 한계
- Pairwise supervision (두 뷰 간 비교)은 **국소적(local)**이라 오차 누적과 전역 기하학적 드리프트에 취약

### TriaGS의 접근
- **다수의 뷰에서 삼각측량한 합의점(consensus point)**과 렌더링된 3D 점을 비교
- 전역적(global) 기하학적 일관성을 self-supervised 방식으로 강제

---

## 3. 핵심 방법론

### 3.1 전체 파이프라인

```
입력 이미지 (COLMAP/Blender)
    ↓
Scene 로딩 (카메라, SfM 초기 포인트 클라우드)
    ↓
Gaussian 초기화
    ↓
학습 루프 (30,000 iterations)
  ├─ Gaussian Splatting 렌더링
  ├─ Photometric Loss (L1 + SSIM)
  ├─ Depth/Normal 정규화
  ├─ Multi-view Geometry Loss (reprojection)
  ├─ Multi-view NCC Loss (패치 기반 photometric)
  ├─ ★ TGPC Loss (삼각측량 기반 일관성) ← 핵심 기여
  ├─ Backward & 최적화
  ├─ Densification (추가/제거)
  └─ 3D Filter 업데이트
    ↓
메쉬 추출 (TSDF 또는 Marching Tetrahedra)
    ↓
평가 (Chamfer Distance, F1-Score, PSNR/SSIM/LPIPS)
```

### 3.2 TGPC Loss (Triangulation-Guided Point Consistency) - 핵심 기여

**Step 1**: 렌더링된 3D 점 Xr을 k개의 이웃 뷰로 투영
```
pi' = si · [ui', vi', 1]^T = Pi · [xr, yr, zr, 1]^T
```

**Step 2**: 선형 삼각측량 시스템 구성 (각 뷰당 2개의 제약 조건)
```
(ui'·Pi3^T - Pi1^T)·X = 0
(vi'·Pi3^T - Pi2^T)·X = 0
→ A·X = 0 (2(k+1)×4 행렬)
```

**Step 3**: SVD로 합의점(consensus point) X* 계산
```
A = U·Σ·V^T → X* = V의 마지막 열 (최소 특이값에 해당)
```

**Step 4**: Geman-McClure robust loss 적용
```
LT = Σ ||Xr - X*||² / (||Xr - X*||² + σ²)
```
- σ는 학습 중 exponential decay로 어닐링
- L2 loss 사용 시 outlier에 의한 gradient explosion으로 학습 실패 → robust loss가 필수

### 3.3 전체 Loss 함수

```
L = L_photo + λt·L_T + λ_normal·L_normal + λ_photo-mv·L_photo-mv
```

| Loss | 설명 | λ (가중치) |
|------|------|-----------|
| L_photo | L1 + DSSIM photometric loss | λ_dssim = 0.2 |
| L_T | TGPC 삼각측량 일관성 loss | 0.1 |
| L_normal | Depth/Normal 정규화 | 0.05 |
| L_geo | Multi-view geometry (reprojection) | 0.03 |
| L_ncc | Multi-view NCC photometric | 0.15 |

### 3.4 기타 주요 구성 요소

| 구성 요소 | 설명 |
|-----------|------|
| **3D Mip-Splatting Filter** | 카메라 거리 기반 anti-aliasing, floater 방지 |
| **Decoupled Appearance** | 뷰별 64차원 appearance embedding + CNN 네트워크로 노출 보정 |
| **Depth Rendering** | RaDe-GS 방식의 ray-Gaussian 교차점 기반 정밀 깊이 계산 |
| **Densification** | 500~15,000 iteration 사이 gradient 기반 splitting/pruning |

---

## 4. 실험 결과 요약

### 4.1 DTU 데이터셋 (Chamfer Distance mm ↓)

| Method | Mean CD | 학습 시간 |
|--------|---------|----------|
| 3DGS | 1.97 | 11.2m |
| SuGaR | 1.32 | 1h |
| 2DGS | 0.80 | 0.32h |
| GOF | 0.74 | - |
| RaDe-GS | 0.68 | 10m |
| Neuralangelo (implicit) | 0.61 | - |
| PGSR | 0.53 | 0.6h |
| **TriaGS (Ours)** | **0.50** | **0.4h (24min)** |

→ PGSR 대비 **5.7% 개선**, implicit 방법인 Neuralangelo 대비 **18% 개선**

### 4.2 Tanks and Temples (F1-Score ↑)

| Method | Barn | Caterpillar | Courthouse | Ignatius | MeetingRoom | Truck | Mean |
|--------|------|-------------|-----------|----------|-------------|-------|------|
| 2DGS | 0.45 | 0.24 | 0.13 | 0.50 | 0.18 | 0.43 | 0.32 |
| PGSR | 0.66 | 0.44 | 0.20 | 0.81 | 0.33 | 0.66 | 0.52 |
| **TriaGS** | **0.62** | **0.40** | **0.20** | **0.75** | **0.28** | **0.71** | **0.49** |

→ PGSR보다 **33% 빠른** 학습 (0.8h vs 1.2h)

### 4.3 NeRF Synthetic (Chamfer Distance ×10⁻² ↓)

| Method | Avg CD |
|--------|--------|
| 3DGS | 3.86 |
| SuGaR | 0.92 |
| PGSR | 0.83 |
| **TriaGS** | **0.76** |

→ PGSR 대비 **8.4% 개선**

### 4.4 Ablation Study

**이웃 뷰 수 (k) 영향** (Truck scene):

| k | PSNR ↑ | F1-Score ↑ | iter/s ↑ |
|---|--------|-----------|----------|
| 1 (pairwise) | 23.21 | 0.57 | 12.8 |
| 4 | 24.11 | 0.62 | 11.6 |
| 8 | 24.38 | 0.66 | 10.6 |
| **12** | **23.94** | **0.71** | **9.8** |

→ k=12가 F1-Score 최적, pairwise(k=1) 대비 **24.6% 개선**

**Loss 함수**: L2 loss 사용 시 outlier에 의한 학습 붕괴 발생 → **Geman-McClure robust loss 필수**

---

## 5. 실험 방법 (How to Reproduce)

### 5.1 환경 설정

```bash
# 1. Conda 환경 생성
conda create -n triags python=3.10
conda activate triags

# 2. PyTorch 설치 (CUDA 버전에 맞게)
pip install torch torchvision
# CUDA toolkit 경로 설정 예시:
# export PATH=/usr/local/cuda-12.8/bin:${PATH}
# export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 커스텀 CUDA 확장 설치
pip install submodules/diff-gaussian-rasterization
pip install git+https://github.com/camenduru/simple-knn

# 5. (TNT용) Marching Tetrahedra 설치
cd submodules/tetra_triangulation
conda install cmake conda-forge::gmp conda-forge::cgal
cmake . && make && pip install -e .
```

### 5.2 데이터셋 준비

| 데이터셋 | 다운로드 | 특징 |
|----------|---------|------|
| **DTU** | [2DGS 전처리본](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) + [DTU GT](https://roboimagedata.compute.dtu.dk/) | 실내 오브젝트, Chamfer Distance 평가 |
| **Tanks & Temples** | [TNT 웹사이트](https://www.tanksandtemples.org/download/) | 대규모 실외/실내, F1-Score 평가 |
| **NeRF Synthetic** | [NeRF repo](https://www.matthewtancik.com/nerf) | Blender 렌더링, PSNR/SSIM/LPIPS |

### 5.3 실험 실행

#### Experiment A: DTU (Surface Reconstruction 품질 평가)

```bash
# 학습 (scan24 예시, 해상도 1/2)
python train.py -s data/dtu/scan24 -m output/dtu_scan24 -r 2 --use_decoupled_appearance

# 메쉬 추출 (TSDF fusion)
python mesh_extract.py -s data/dtu/scan24 -m output/dtu_scan24

# 평가 (Chamfer Distance)
python eval/eval_dtu/evaluate_single_scene.py \
    --input_mesh output/dtu_scan24/recon.ply \
    --scan_id 24 \
    --output_dir output/dtu_scan24/eval \
    --mask_dir data/dtu \
    --DTU data/dtu_eval
```

**평가 지표**: Chamfer Distance (mm) - 낮을수록 좋음

#### Experiment B: Tanks and Temples (대규모 장면 복원)

```bash
# 학습
python train.py -s data/TNT/Barn -m output/tnt_barn -r 2 --eval --use_decoupled_appearance

# 메쉬 추출 (Marching Tetrahedra)
python mesh_extract_tetrahedra.py -s data/TNT/Barn -m output/tnt_barn --iteration 30000

# 평가 (open3d==0.9 환경 필요)
python eval/eval_tnt/run.py \
    --dataset-dir data/TNT_GT/Barn \
    --traj-path data/TNT/Barn/Barn_COLMAP_SfM.log \
    --ply-path output/tnt_barn/recon.ply \
    --out-dir output/tnt_barn/eval
```

**평가 지표**: F1-Score (Precision + Recall) - 높을수록 좋음

#### Experiment C: NeRF Synthetic (Novel View Synthesis)

```bash
# 학습
python train.py -s data/nerf_synthetic/lego -m output/nerf_lego --eval

# 렌더링
python render.py -m output/nerf_lego

# 메트릭 계산
python metrics.py -m output/nerf_lego
```

**평가 지표**: PSNR (↑), SSIM (↑), LPIPS (↓)

### 5.4 주요 학습 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--iterations` | 30,000 | 총 학습 반복 횟수 |
| `-r` | 1 | 해상도 축소 비율 (2 = 1/2 해상도) |
| `--use_decoupled_appearance` | False | 뷰별 appearance 보정 활성화 |
| `--eval` | False | train/test 분할 활성화 |
| `--lambda_dssim` | 0.2 | DSSIM loss 가중치 |
| `--lambda_depth_normal` | 0.05 | Depth/Normal 정규화 가중치 |
| `--lambda_multi_view_geo` | 0.03 | Multi-view geometry loss 가중치 |
| `--lambda_multi_view_ncc` | 0.15 | Multi-view NCC loss 가중치 |
| `--lambda_tgpc` | 0.1 | TGPC 삼각측량 loss 가중치 |
| `--multi_view_num` | 8 | 이웃 뷰 수 |
| `--multi_view_max_angle` | 30 | 최대 뷰 각도 (도) |

---

## 6. 커스텀 실험 아이디어

### 6.1 Ablation 재현
```bash
# TGPC loss 없이 학습 (lambda_tgpc=0)
python train.py -s data/dtu/scan24 -m output/ablation_no_tgpc -r 2 \
    --use_decoupled_appearance --lambda_tgpc 0.0

# NCC loss 없이 학습
python train.py -s data/dtu/scan24 -m output/ablation_no_ncc -r 2 \
    --use_decoupled_appearance --lambda_multi_view_ncc 0.0

# 이웃 뷰 수 변경 (k=4)
python train.py -s data/dtu/scan24 -m output/ablation_k4 -r 2 \
    --use_decoupled_appearance --multi_view_num 4
```

### 6.2 하이퍼파라미터 튜닝
```bash
# TGPC loss 가중치 변경
python train.py -s data/dtu/scan24 -m output/tgpc_02 -r 2 \
    --use_decoupled_appearance --lambda_tgpc 0.2

# 최대 뷰 각도 변경
python train.py -s data/dtu/scan24 -m output/angle_45 -r 2 \
    --use_decoupled_appearance --multi_view_max_angle 45
```

### 6.3 자체 데이터셋 사용
1. 다양한 각도에서 장면 촬영 (30장 이상 권장)
2. COLMAP으로 SfM 실행하여 카메라 포즈 + sparse point cloud 생성
3. COLMAP 출력 구조에 맞게 데이터 정리:
   ```
   my_scene/
   ├── images/
   │   ├── 001.jpg
   │   └── ...
   └── sparse/
       └── 0/
           ├── cameras.bin
           ├── images.bin
           └── points3D.bin
   ```
4. 학습 실행:
   ```bash
   python train.py -s my_scene -m output/my_scene --use_decoupled_appearance
   ```

---

## 7. 시스템 요구사항

| 항목 | 요구사항 |
|------|---------|
| GPU | CUDA 지원 GPU (RTX 4090 기준 개발) |
| Python | 3.10 |
| PyTorch | CUDA toolkit 포함 |
| 디스크 | DTU ~10GB, TNT ~50GB+ |
| RAM | 16GB+ 권장 |
| 학습 시간 | DTU ~24분, TNT ~48분 (RTX 4090) |

---

## 8. 한계점 및 향후 방향

### 한계
- 입력 데이터 품질에 민감 (카메라 포즈 부정확 또는 sparse한 뷰에서 성능 저하)
- TNT 데이터셋에서 PGSR 대비 약간 낮은 F1-Score (0.49 vs 0.52)

### 향후 방향
- 삼각측량 잔차를 카메라 포즈 정제 신호로 활용 가능
- 민감도를 오히려 강점으로 전환할 잠재력

---

## 9. 코드 구조 요약

```
triags/
├── train.py                    # 메인 학습 스크립트
├── render.py                   # Novel view 렌더링
├── metrics.py                  # PSNR/SSIM/LPIPS 평가
├── mesh_extract.py             # TSDF 기반 메쉬 추출 (DTU)
├── mesh_extract_tetrahedra.py  # Marching Tetrahedra 메쉬 추출 (TNT)
├── arguments/                  # 하이퍼파라미터 정의
├── scene/
│   ├── gaussian_model.py       # 3D Gaussian 표현 (핵심 모델)
│   ├── cameras.py              # 카메라 클래스
│   ├── app_model.py            # Appearance 보정 모델
│   └── dataset_readers.py      # 데이터 로더
├── gaussian_renderer/          # CUDA 기반 렌더링
├── utils/
│   ├── multiple_view_loss.py   # ★ TGPC + Multi-view loss (핵심 기여)
│   ├── loss_utils.py           # L1, SSIM, LNCC loss
│   ├── mesh_utils.py           # 메쉬 추출 유틸리티
│   └── ...
├── eval/                       # 데이터셋별 평가 스크립트
└── submodules/
    └── diff-gaussian-rasterization/  # CUDA 래스터라이저
```
