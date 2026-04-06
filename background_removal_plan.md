# 배경 제거 구현 계획

> 2026-04-06 | TriaGS + GT Depth | 배경 제거를 통한 floater 근본 해결

---

## 1. 현재 문제

### 증상
- 학습된 3DGS 모델에서 **부유 Gaussian (floater)** 관찰됨
- 기하학 메트릭(AbsRel 0.0042, RMSE 2.52)은 우수하나 3D 뷰어에서 노이즈 존재
- 오브젝트는 이미지의 ~12%만 차지, 나머지 **88%가 균일 회색 배경** [70,70,70]

### 원인
- 배경 영역(88%)에서 불필요한 Gaussian이 생성됨
- RGB loss가 배경 회색을 맞추기 위해 Gaussian을 배치
- 이 Gaussian들이 floater로 남음

### 데이터 특성

| | RGB | Depth |
|---|---|---|
| 오브젝트 | 실제 색상 | **양수** (실제 거리) |
| 배경 | 균일 회색 [70,70,70] | **0** (검정) |

→ `depth > 0`으로 오브젝트/배경 완벽 분리 가능

---

## 2. 해결 방향 비교

### 방법 A: train.py에서 loss 마스킹 (현재 구현됨)

```python
# train.py:149-155 (이미 적용됨)
obj_mask = (viewpoint_cam.gt_depth > 0).float()
rendered_image = rendered_image * obj_mask_3ch
gt_image = gt_image * obj_mask_3ch
```

**문제점:**
- 입력 RGB 이미지 자체에 여전히 회색 배경 존재
- 렌더링 출력은 검정 배경, GT는 회색 배경 → **평가 메트릭 불일치**
- PSNR/SSIM/LPIPS 평가에도 마스킹 필요
- 시각화에서도 배경색 차이 (검정 vs 회색)
- 모든 downstream 코드에 마스킹 분기 필요 → 복잡도 증가

### 방법 B: 입력 데이터 자체에서 배경 제거 ✅ (채택 예정)

```python
# Cell 11 (데이터 준비 단계)에서 RGB 이미지 배경을 검정으로 교체
rgb = cv2.imread(rgb_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
mask = (depth > 0).astype(np.float32)[..., np.newaxis]
rgb_masked = (rgb * mask).astype(np.uint8)  # 배경 → [0,0,0]
cv2.imwrite(output_path, rgb_masked)
```

**원리:**
- 원본 3DGS가 NeRF Synthetic에서 사용하는 방식과 동일
- GT 이미지 배경 = `[0,0,0]`, 렌더러 bg_color = `[0,0,0]` → 자연스럽게 일치
- 배경에서 loss = |0 - 0| = 0 → gradient 없음 → Gaussian 생성 안 됨

**비교:**

| 항목 | A: loss 마스킹 | B: 데이터 배경 제거 |
|------|---------------|-------------------|
| train.py RGB loss | 마스킹 코드 필요 | **불필요** (자동으로 0) |
| depth-normal loss | 마스킹 코드 필요 | 마스킹 유지 (신호 품질) |
| PSNR/SSIM/LPIPS 평가 | 마스킹 코드 필요 | **불필요** (양쪽 다 검정) |
| 시각화 | 배경색 차이 발생 | **불필요** (양쪽 다 검정) |
| 메쉬 추출 | 동일 | 동일 |
| 코드 복잡도 | 높음 (곳곳에 분기) | **낮음** (데이터 한 곳에서 처리) |
| 경계 Gaussian 처리 | 하드 마스크로 경계 무시 | **자연스럽게 패널티** (경계 bleed 억제) |

---

## 3. 구현 계획

### Step 1: 데이터 준비 (Cell 11) 수정
- RGB 이미지 저장 시 `depth > 0` 마스크로 배경을 검정으로 교체
- 심볼릭 링크 → 실제 파일 저장으로 변경
- train/test 모두 적용

### Step 2: train.py RGB 마스킹 제거
- 현재 추가된 `obj_mask`에 의한 `rendered_image * obj_mask_3ch` 제거
- 입력 데이터에서 이미 배경이 제거되었으므로 불필요

### Step 3: train.py depth-normal 마스킹 유지
- depth-normal loss는 렌더링 depth/normal 기반 → 배경 영역의 normal이 불안정
- 오브젝트 영역만으로 `.mean()` 계산하는 코드 유지

### Step 4: 평가 코드 변경 없음
- GT와 렌더링 모두 검정 배경 → PSNR/SSIM/LPIPS 자연스럽게 정확

---

## 4. 배경 제거가 동작하는 이유 (파이프라인 전체 추적)

### Phase 0: 초기화
- Depth back-projection으로 초기 포인트 생성
- `depth > 0`인 오브젝트 표면 위에만 198K개 포인트 배치
- **배경 Gaussian 0개로 시작**

### Phase 1: iter 1~15K (RGB + depth-normal)
- 입력 RGB가 검정 배경 → 렌더러 bg_color도 검정 → 배경 loss = 0
- 배경 Gaussian에 gradient 없음 → densification 안 됨
- opacity reset 후 gradient 없으면 pruning → 배경 Gaussian 소멸
- **배경 Gaussian 생성 경로 차단**

### Phase 2: iter 15K~25K (multi-view + GT depth)
- depth-normal loss: 오브젝트만 `.mean()` (마스킹 유지)
- GT depth loss: 이미 `valid_mask = gt_depth > 0` 처리됨
- multi-view geo: `d_mask`가 배경 자동 제외 (depth ≈ 0 → reprojection 실패)
- NCC: `valid_indices` 기반 샘플링 → 배경 미포함
- **모든 loss에서 배경 gradient 없음**

### Phase 3: iter 25K~30K (fine-tuning)
- densification 이미 종료
- 새 Gaussian 생성 불가, 기존만 미세 조정
- **배경 Gaussian 여전히 0개**

---

## 5. 현재 코드 상태

### 이미 수정된 파일
- `train.py`: loss 마스킹 코드 추가됨 (line 149-155, 177-181)
  - → Step 2에서 RGB 마스킹 부분 제거 예정
  - → Step 3에서 depth-normal 마스킹 유지
- `TriaGS_Depth_Pipeline.ipynb` Cell 19: 프루닝 임계값 변경
  - OPACITY_THRESHOLD: 0.5 → 0.02
  - SCALE_PERCENTILE: 95 → 99

### 수정 필요한 파일
- `TriaGS_Depth_Pipeline.ipynb` Cell 11: 데이터 준비 시 배경 제거 추가
- `train.py`: RGB 마스킹 코드 제거 (데이터 레벨에서 처리하므로)

---

## 6. 이전 시도 기록

| 방법 | 상태 | 결과 |
|------|------|------|
| Depth loss 25K→30K 확장 | 실패 → 원복 | multi-view fine-tuning 방해 |
| 후처리 프루닝 (공격적) | 구현됨 | opacity 0.5 + 95th pctl → 너무 공격적 |
| 후처리 프루닝 (균형형) | **적용됨** | opacity 0.02 + 99th pctl |
| train.py loss 마스킹 | 구현됨 | 동작하나 downstream 문제 |
| **데이터 레벨 배경 제거** | **계획됨** | 가장 깔끔한 해결책 |

---

## 7. ChatGPT 논의 반영 (2026-04-06)

ChatGPT의 추천 파이프라인과 우리 상황을 대조 분석.

### 7.1 "원본 RGB + 바이너리 마스크로 가는 게 안전"

**동의.** 단, 우리는 GT depth가 있으므로 `depth > 0`이 완벽한 바이너리 마스크.
rembg 같은 외부 도구 불필요. depth 기반 마스크가 픽셀 단위로 더 정확함.

### 7.2 "COLMAP feature extraction에 마스크 적용"

**우리 파이프라인에 해당 없음.** COLMAP SfM을 실행하지 않고, 알려진 calibration으로
COLMAP 포맷 파일을 직접 생성. 초기 포인트도 depth back-projection 기반.

### 7.3 "RGBA alpha=0만으로는 배경이 안 없어질 수 있다"

**우리가 이미 확인한 사실과 일치.** TriaGS에서 `gt_alpha_mask`는 `cameras.py`에
저장만 되고 `train.py`의 loss에서 전혀 사용되지 않음.

### 7.4 "GT에만 마스크 적용 vs 양쪽 마스킹" — 핵심 포인트

| 방식 | 배경에서의 loss | 경계 bleed Gaussian |
|------|---------------|-------------------|
| 양쪽 마스킹 | `\|0 - 0\| = 0` | 마스크로 잘려서 **패널티 없음** → 경계 지저분 |
| GT만 마스킹 | `\|rendered - 0\|` | 배경에 뭐가 있으면 **적극적으로 패널티** |
| **데이터 배경 제거** | `\|rendered - 0\|` | **GT만 마스킹과 동일 효과** |

데이터 배경 제거(GT 이미지 배경 = `[0,0,0]` = `bg_color`)는 "GT만 마스킹"과
수학적으로 동일하면서 코드 수정 없이 자연스럽게 동작.

### 7.5 종합

| ChatGPT 추천 | 우리 상황 | 판단 |
|-------------|---------|------|
| 원본 보관 + 별도 마스크 | depth > 0 마스크 존재 | **동의** — 원본 보관, 복사본에 마스크 적용 |
| COLMAP에 마스크 적용 | COLMAP SfM 미사용 | **해당 없음** |
| 학습 loss에서 마스크 사용 | 데이터 레벨 제거가 더 깔끔 | **데이터 레벨 채택** |
| GT에만 마스크 적용 | 데이터 배경 제거 = 동일 효과 | **자연스럽게 충족** |
| rembg 사용 | GT depth 있음 | **불필요** |

---

## 8. 다음 액션

- [x] Cell 11 수정: RGB 이미지 배경 제거 (depth mask 적용)
- [x] train.py: RGB 마스킹 코드 제거, depth-normal 마스킹 유지
- [ ] 재학습 후 결과 확인
- [ ] floater 감소 여부 3D 뷰어에서 확인
