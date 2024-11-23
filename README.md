# Korea_Computer_Congress_2024
2024 한국정보과학회 코드
본 프로젝트는 Autoencoder를 사용하여 노이즈가 포함된 MNIST 이미지를 복원하는 모델을 학습하고 테스트합니다.

---

## 설치 및 환경 설정

### 실험 Python 환경
- 필수 라이브러리 설치:
  ```bash
  pip install -r requirements.txt

## 모델 훈련: train.py
### 사용법
`train.py`는 MNIST 데이터를 사용하여 Autoencoder 모델을 학습시킵니다. 다음 명령어를 실행하면 모델을 학습시킬 수 있습니다:
```bash
python train.py --data_dir <DATA_DIR> --model_save_dir <MODEL_SAVE_DIR> --batch_size <BATCH_SIZE> --epochs <EPOCHS> --learning_rate <LEARNING_RATE> --save_epoch <SAVE_EPOCH>
```

### 주요 옵션
- `--data_dir` (필수): MNIST 데이터셋의 저장 경로 (없는 폴더라면 다운로드해 생성된다).
- `--model_save_dir` (필수): 학습된 모델을 저장할 디렉터리.
- `--batch_size` (선택): 학습에 사용할 배치 크기 (기본값: 512).
- `--epochs` (선택): 학습 에포크 수 (기본값: 20).
- `--learning_rate` (선택): 학습률 (기본값: 0.001).
- `--save_epoch` (선택): 모델 체크포인트를 저장할 에포크 간격 (기본값: 5).

### 실행 예시
```bash
python train.py --data_dir ./mnist --model_save_dir ./checkpoint
```

### 출력
(위 실행 예시 명령어 결과)
- 학습 과정의 손실 값은 매 에포크마다 콘솔에 출력됩니다.
- 학습된 모델은 지정된 `./checkpoint` 디렉터리에 저장됩니다. 가장 최근의 모델은 `./checkpoint/checkpoint.pth`에 저장됩니다.
- 손실 그래프는 `./checkpoint/loss_plot.png` 형식으로 자동 저장됩니다.

---
## 모델 테스트: test.py

### 사용법
`test.py`는 학습된 Autoencoder 모델을 사용하여 테스트 데이터의 노이즈를 제거하고 복원된 이미지를 저장합니다. 다음 명령어를 실행하면 테스트를 수행할 수 있습니다:

```bash
python test.py --data_path <DATA_PATH> --output_dir <OUTPUT_DIR> --pretrained_path <PRETRAINED_PATH> --batch_size <BATCH_SIZE>
```

### 주요 옵션
- `--data_path` (필수): 테스트 데이터셋의 저장 경로.
- `--output_dir` (필수): 복원된 이미지를 저장할 디렉터리.
- `--pretrained_path` (필수): 학습된 모델의 경로 (.pth 파일).
- `--batch_size` (선택): 테스트에 사용할 배치 크기 (기본값: 512).

### 실행 예시
```bash
python test.py --pretrained_path ./checkpoint/checkpoint.pth --data_path ./mnist --output_dir ./results
```

### 출력
- 원본 이미지, 노이즈 이미지, 복원된 이미지는 `output_dir`에 PNG 형식으로 저장됩니다.
  - 예:
    - 원본 이미지: `./results/original/original_0.png`, `./results/original/original_1.png` ...
    - 노이즈 이미지: `./results/noisy/noisy_0.png`, `./results/noizy/noisy_1.png` ...
    - 복원된 이미지: `./results/restored/restored_0.png`, `./results/restored/restored_1.png` ...
- 콘솔에는 실행 시간, 처리된 데이터 수, 평균 처리 시간이 출력됩니다.
  ```
  Total samples processed: 10000 samples
  Total execution time: 0.7451 seconds
  Average time per sample: 0.000075 seconds/sample
  ```

---

## 결과 확인
- 학습 과정의 손실 그래프(`loss_plot.png`)는 `train.py` 실행 시 자동 저장됩니다.
- 테스트 결과는 `test.py` 실행 시 지정한 `output_dir`에서 확인할 수 있습니다.
```

