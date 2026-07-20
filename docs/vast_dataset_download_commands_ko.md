# Vast AI 데이터셋 다운로드 명령어 정리

대상 데이터셋: `tank_armor_prepared_v20260630`  
서버 복원 위치: `/workspace/ironflow/prestaged/tank_armor_prepared_v20260630`

이 문서는 다른 컴퓨터에서 `D:\Model_LAB\vision_experiment_platform` 프로그램을 실행하기 전에 Vast AI 서버에 데이터셋을 미리 받아두는 절차를 정리한 것이다.

## 1. 로컬 공개키 확인

각 컴퓨터마다 SSH 키가 다를 수 있으므로, 해당 컴퓨터에서 아래 명령어로 공개키를 확인한다.

```powershell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

Vast AI 인스턴스의 `Manage SSH Keys`에서 이 공개키가 `Instance SSH Keys`에 들어가 있어야 한다.

## 2. SSH 접속 테스트

Vast AI 화면의 `Direct ssh connect`에서 IP와 포트를 확인한 뒤 아래 명령어의 `<PORT>`, `<HOST>`를 바꿔서 실행한다.

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST> "echo ok"
```

예시:

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p 40022 root@203.0.113.10 "echo ok"
```

`ok`가 출력되면 SSH 연결은 정상이다.

## 3. 다운로드 스크립트 서버로 전송

로컬 프로그램 폴더가 `D:\Model_LAB\vision_experiment_platform`라고 가정한다.

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -P <PORT> "D:\Model_LAB\vision_experiment_platform\scripts\prepare_vast_tank_armor_dataset.sh" root@<HOST>:/workspace/prepare_vast_tank_armor_dataset.sh
```

예시:

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -P 40022 "D:\Model_LAB\vision_experiment_platform\scripts\prepare_vast_tank_armor_dataset.sh" root@203.0.113.10:/workspace/prepare_vast_tank_armor_dataset.sh
```

## 4. 서버에서 데이터셋 다운로드/복원 실행

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST> "bash /workspace/prepare_vast_tank_armor_dataset.sh"
```

예시:

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p 40022 root@203.0.113.10 "bash /workspace/prepare_vast_tank_armor_dataset.sh"
```

이 스크립트는 Google Drive 폴더에서 `part-*.bin` 파일을 받고, tar 파일로 합친 뒤 `/workspace/ironflow/prestaged` 아래에 압축을 푼다.

## 5. 복원 확인

아래 명령어로 필요한 데이터셋 경로가 있는지 확인한다.

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST> 'if [ -f /workspace/ironflow/prestaged/tank_armor_prepared_v20260630/detector_tank_av/detection/data.yaml ]; then echo DETECTOR=ok; else echo DETECTOR=missing; fi; if [ -d /workspace/ironflow/prestaged/tank_armor_prepared_v20260630/classifier_mbt/crops/train ]; then echo CLASSIFIER_MBT=ok; else echo CLASSIFIER_MBT=missing; fi; if [ -d /workspace/ironflow/prestaged/tank_armor_prepared_v20260630/classifier_av/crops/train ]; then echo CLASSIFIER_AV=ok; else echo CLASSIFIER_AV=missing; fi'
```

정상 출력:

```text
DETECTOR=ok
CLASSIFIER_MBT=ok
CLASSIFIER_AV=ok
```

## 6. GUI에서 사용할 Remote Root

GUI의 `Remote Root` 또는 `Remote pre-staged path`는 아래 값이어야 한다.

```text
/workspace/ironflow/prestaged/tank_armor_prepared_v20260630
```

## 7. Proxy SSH를 써야 하는 경우

Direct SSH가 안 되고 Vast AI의 `Proxy ssh connect`를 써야 하는 경우 `<PROXY_PORT>`, `<PROXY_HOST>`를 사용한다.

접속 테스트:

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p <PROXY_PORT> root@<PROXY_HOST> "echo ok"
```

스크립트 전송:

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -P <PROXY_PORT> "D:\Model_LAB\vision_experiment_platform\scripts\prepare_vast_tank_armor_dataset.sh" root@<PROXY_HOST>:/workspace/prepare_vast_tank_armor_dataset.sh
```

스크립트 실행:

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p <PROXY_PORT> root@<PROXY_HOST> "bash /workspace/prepare_vast_tank_armor_dataset.sh"
```

## 8. 자주 나는 문제

### Permission denied (publickey)

명령어 문제가 아니라 Vast 인스턴스에 공개키가 제대로 등록되지 않은 상태일 가능성이 크다.

해결 순서:

1. `Manage SSH Keys`에서 기존 키를 삭제한다.
2. 아래 명령어로 현재 컴퓨터의 공개키를 다시 복사한다.

```powershell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

3. Vast AI의 `New SSH Key`에 붙여넣고 `ADD SSH KEY`를 누른다.
4. 30초 정도 기다린 뒤 SSH 접속 테스트를 다시 한다.

### data.yaml not found

GUI의 Remote Root가 예전 경로를 보고 있을 가능성이 있다. 반드시 아래 경로인지 확인한다.

```text
/workspace/ironflow/prestaged/tank_armor_prepared_v20260630
```
