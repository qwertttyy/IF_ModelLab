# Model_LAB 배포본 안내

이 폴더는 기존 실험 산출물을 제외하고 새 PC에서 설치하고 실행할 수 있도록 정리한 배포본이다.

설치, 데이터 복원, Vast.ai 연결과 실행은 [`README_FIRST_RUN_KO.md`](README_FIRST_RUN_KO.md)를 따른다.

## 포함 항목

- GUI와 원격 실험 엔진
- 실험 및 모델 설정
- Vast.ai 준비/수집 스크립트
- 사전학습 모델 파일
- 비어 있는 `runs/w`, `runs/e`, `runs/user_datasets`

## 제외 항목

- 기존 실험 결과, 체크포인트, 보고서
- 기존 SQLite 실행 이력
- Python 가상환경과 캐시
- 데이터셋
- SSH 개인키와 사용자별 Vast.ai 접속 정보

## 빠른 시작

1. 전체 폴더를 짧은 영문 경로에 압축 해제한다.
2. `setup_first.bat`을 실행한다.
3. `run_gui.bat --check`로 점검한다.
4. `run_gui.bat`으로 GUI를 연다.

`models/checkpoints/pretrained`, `configs`, `src`, `scripts`는 삭제하거나 일부만 전달하지 않는다.
