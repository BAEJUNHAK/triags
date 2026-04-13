# Git Restore Point

- **날짜**: 2026-04-13
- **브랜치**: `main`
- **커밋 해시**: `ae7d416bd1387a8166f88f20d7e44f89644f7ade`
- **커밋 메시지**: Per-view and Error-vs-Depth analysis for all iterations (15K/25K/30K)

## 복구 방법

코딩하다 망쳤을 때 이 시점으로 돌아오려면:

```bash
# 1. 현재 변경사항 버리고 이 커밋으로 되돌아오기
git checkout main
git reset --hard ae7d416

# 2. 변경사항을 살리고 싶으면 먼저 stash
git stash
git reset --hard ae7d416
```
