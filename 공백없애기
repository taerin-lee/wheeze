
import os
import shutil
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
     classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 0. 설정
# =========================

BASE_DIR = r"/content/drive/MyDrive/BioCAS2022"

TRAIN_WAV_DIR = os.path.join(BASE_DIR, "train2022_wav_under5")
TRAIN_TXT_DIR = os.path.join(BASE_DIR, "train2022_txt_under5")

TEST_WAV_DIR = os.path.join(BASE_DIR, "test2022_wav_under5")
TEST_TXT_DIR = os.path.join(BASE_DIR, "test2022_txt_under5")

BASELINE_SAVE_PATH = "best_float_baseline.pt"
BINARY_SAVE_PATH = "best_binaryfc_finetuned.pt"

TARGET_SR = 4000
TARGET_SECONDS = 2.5
N_MELS = 48
N_FFT = 512
HOP_LENGTH = 256

BATCH_SIZE = 16

BASELINE_EPOCHS = 20
BASELINE_LR = 1e-4

BINARY_EPOCHS = 12
BINARY_LR = 3e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# 1. 파일 찾기
# =========================

train_all_files = sorted(glob(os.path.join(TRAIN_WAV_DIR, "*.wav")))
test_files = sorted(glob(os.path.join(TEST_WAV_DIR, "*.wav")))

print("train 전체 wav 개수:", len(train_all_files))
print("test wav 개수:", len(test_files))

if len(train_all_files) == 0:
    raise RuntimeError("train wav 파일을 찾지 못했습니다. TRAIN_WAV_DIR 확인하세요.")
if len(test_files) == 0:
    raise RuntimeError("test wav 파일을 찾지 못했습니다. TEST_WAV_DIR 확인하세요.")

train_files, val_files = train_test_split(
    train_all_files,
    test_size=0.2,
    random_state=SEED
)

print("train 파일 수:", len(train_files))
print("val 파일 수:", len(val_files))
print("test 파일 수:", len(test_files))


# =========================
# 1.1 event driven threshold search
# =========================

def evaluate_with_input_gating(model, file_list, txt_dir, input_gating_th):
    model.eval()
    preds = []
    targets = []
    gated_count = 0
    total_count = 0

    with torch.no_grad():
        for wav_path in file_list:
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            txt_path = os.path.join(txt_dir, base_name + ".txt")
            if not os.path.exists(txt_path): continue

            ann = load_annotation(txt_path)
            signal, sr = librosa.load(wav_path, sr=None)

            for _, row in ann.iterrows():
                start_sample = int(float(row["start"]) * sr)
                end_sample = int(float(row["end"]) * sr)
                segment = signal[start_sample:end_sample]
                if len(segment) == 0: continue

                total_count += 1
                
                # [수정] CNN 전단: 원본 오디오 신호의 입력 에너지 계산
                input_energy = np.mean(np.abs(segment))

                if input_energy < input_gating_th:
                    # 임계값 미달 시: CNN/CIM 연산 전면 Off -> 바로 정상 판정
                    preds.append(non_wheeze_idx)
                    gated_count += 1
                else:
                    # 임계값 통과 시: CNN + 8T SRAM 펄스 인가 및 연산 수행
                    # (기존 전처리 루틴 적용)
                    seg, seg_sr = fix_length(segment, sr, TARGET_SR, TARGET_SECONDS)
                    feat = extract_logmel(seg, seg_sr) # (48, 40) 등
                    
                    # 텐서 변환 및 추론
                    xb = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    out = model(xb)
                    prob = torch.softmax(out, dim=1).cpu().numpy()[0]
                    
                    # 소프트웨어 판정 threshold 적용
                    if prob[wheeze_idx] >= best_val_threshold_binary:
                        preds.append(wheeze_idx)
                    else:
                        preds.append(non_wheeze_idx)
                
                label_4class = label_from_row(int(row["crackles"]), int(row["wheezes"]))
                targets.append(list(le.classes_).index(map_binary(label_4class)))

    return np.array(targets), np.array(preds), gated_count, total_count

# =========================
# 2. annotation / label
# =========================

def load_annotation(txt_path):
    df = pd.read_csv(
        txt_path,
        sep=r"\s+",
        header=None,
        names=["start", "end", "crackles", "wheezes"]
    )
    return df


def label_from_row(crackles, wheezes):
    if crackles == 0 and wheezes == 0:
        return "none"
    elif crackles == 1 and wheezes == 0:
        return "crackles"
    elif crackles == 0 and wheezes == 1:
        return "wheezes"
    else:
        return "both"


def map_binary(label):
    if label in ["wheezes", "both"]:
        return "wheeze"
    else:
        return "non-wheeze"


# =========================
# 3. 전처리 / augmentation
# =========================

def fix_length(audio, sr, target_sr=4000, target_seconds=2.5):
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    target_len = int(target_sr * target_seconds)

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    return audio, sr


def extract_logmel(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)
    return logmel.astype(np.float32)


def augment_audio(audio):
    x = audio.copy()

    noise = np.random.randn(len(x)) * 0.003
    x = x + noise

    gain = np.random.uniform(0.85, 1.15)
    x = x * gain

    if len(x) > 1:
        shift = np.random.randint(-500, 500)
        x = np.roll(x, shift)

    return x


# =========================
# 4. 파일 리스트 -> dataset 생성
# =========================

def build_dataset_from_files(file_list, txt_dir, augment=False):
    X = []
    y = []
    cycle_lengths = []

    missing_txt = 0

    for i, wav_path in enumerate(file_list):
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        txt_path = os.path.join(txt_dir, base_name + ".txt")

        if not os.path.exists(txt_path):
            missing_txt += 1
            print("txt 없음:", txt_path)
            continue

        ann = load_annotation(txt_path)
        signal, sr = librosa.load(wav_path, sr=None)

        for _, row in ann.iterrows():
            start_sample = int(float(row["start"]) * sr)
            end_sample = int(float(row["end"]) * sr)

            segment = signal[start_sample:end_sample]

            if len(segment) == 0:
                continue

            cycle_lengths.append(len(segment) / sr)

            label_4class = label_from_row(int(row["crackles"]), int(row["wheezes"]))
            label_binary = map_binary(label_4class)

            segments_to_use = [segment]

            if augment:
                if label_binary == "wheeze":
                    segments_to_use.append(augment_audio(segment))
                    segments_to_use.append(augment_audio(segment))
                else:
                    segments_to_use.append(augment_audio(segment))

            for seg in segments_to_use:
                seg, seg_sr = fix_length(
                    seg,
                    sr,
                    target_sr=TARGET_SR,
                    target_seconds=TARGET_SECONDS
                )

                feat = extract_logmel(seg, seg_sr)
                X.append(feat)
                y.append(label_binary)

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(file_list)} 파일 처리 완료")

    print("missing txt 개수:", missing_txt)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y, cycle_lengths


print("\ntrain dataset 생성 중...")
X_train, y_train, train_lengths = build_dataset_from_files(
    train_files,
    TRAIN_TXT_DIR,
    augment=True
)

print("\nval dataset 생성 중...")
X_val, y_val, val_lengths = build_dataset_from_files(
    val_files,
    TRAIN_TXT_DIR,
    augment=False
)

print("\ntest dataset 생성 중...")
X_test, y_test, test_lengths = build_dataset_from_files(
    test_files,
    TEST_TXT_DIR,
    augment=False
)

print("\ntrain X shape:", X_train.shape)
print("train y shape:", y_train.shape)
print("train 라벨 분포:", Counter(y_train))

print("\nval X shape:", X_val.shape)
print("val y shape:", y_val.shape)
print("val 라벨 분포:", Counter(y_val))

print("\ntest X shape:", X_test.shape)
print("test y shape:", y_test.shape)
print("test 라벨 분포:", Counter(y_test))

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise RuntimeError("생성된 데이터가 비어 있습니다. wav/txt 매칭과 txt 형식을 확인하세요.")


# =========================
# 5. Dataset
# =========================

class RespDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# 6. 모델 정의
# =========================

class BetterCNN_FloatFC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BinaryLinearScaled(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        alpha = self.weight.abs().mean(dim=1, keepdim=True).detach()

        binary_weight = self.weight.sign()
        binary_weight = torch.where(
            binary_weight == 0,
            torch.ones_like(binary_weight),
            binary_weight
        )

        binary_weight = alpha * binary_weight
        return F.linear(x, binary_weight, self.bias)


class BetterCNN_BinaryFC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.binary_fc = BinaryLinearScaled(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.binary_fc(x)
        return x


# =========================
# 7. 인코딩 / DataLoader
# =========================

le = LabelEncoder()

y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
y_test_encoded = le.transform(y_test)

print("\n클래스:", list(le.classes_))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

train_ds = RespDataset(X_train, y_train_encoded)
val_ds = RespDataset(X_val, y_val_encoded)
test_ds = RespDataset(X_test, y_test_encoded)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# 8. 평가 함수
# =========================

def evaluate_with_probs(model, loader):
    model.eval()

    preds = []
    targets = []
    probs_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)

            out = model(xb)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            pred = np.argmax(probs, axis=1)

            probs_all.extend(probs)
            preds.extend(pred)
            targets.extend(yb.numpy())

    return np.array(targets), np.array(preds), np.array(probs_all)


def search_best_threshold(targets, probs_all, wheeze_idx, non_wheeze_idx):
    best_threshold = None
    best_f1 = -1
    best_preds = None
    best_precision = None
    best_recall = None

    for th in np.arange(0.40, 0.551, 0.01):
        tuned_preds = []

        for p in probs_all:
            if p[wheeze_idx] >= th:
                tuned_preds.append(wheeze_idx)
            else:
                tuned_preds.append(non_wheeze_idx)

        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            tuned_preds,
            average=None,
            labels=[non_wheeze_idx, wheeze_idx],
            zero_division=0
        )

        wheeze_precision = precision[1]
        wheeze_recall = recall[1]
        wheeze_f1 = f1[1]

        if wheeze_f1 > best_f1:
            best_f1 = wheeze_f1
            best_threshold = round(float(th), 2)
            best_preds = tuned_preds
            best_precision = wheeze_precision
            best_recall = wheeze_recall

    return best_threshold, best_f1, best_precision, best_recall, np.array(best_preds)


def print_test_results(title, targets, preds, class_names):
    print(f"\n[{title}]")
    print(classification_report(targets, preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))


# =========================
# 9. 1단계: Float baseline 학습
# =========================

print("\n" + "=" * 70)
print("[1단계] Float baseline 학습 시작")
print("=" * 70)

float_model = BetterCNN_FloatFC(num_classes=len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(float_model.parameters(), lr=BASELINE_LR)

wheeze_idx = list(le.classes_).index("wheeze")
non_wheeze_idx = list(le.classes_).index("non-wheeze")

best_val_f1_float = -1
best_epoch_float = -1
best_val_threshold_float = None

for epoch in range(BASELINE_EPOCHS):
    float_model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        out = float_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    val_targets, _, val_probs = evaluate_with_probs(float_model, val_loader)
    val_th, val_f1, val_prec, val_rec, _ = search_best_threshold(
        val_targets,
        val_probs,
        wheeze_idx,
        non_wheeze_idx
    )

    print(
        f"[Float] Epoch {epoch + 1}/{BASELINE_EPOCHS} | "
        f"Loss={avg_loss:.4f} | "
        f"Val th={val_th:.2f} | "
        f"P={val_prec:.3f}, R={val_rec:.3f}, F1={val_f1:.3f}"
    )

    if val_f1 > best_val_f1_float:
        best_val_f1_float = val_f1
        best_epoch_float = epoch + 1
        best_val_threshold_float = val_th
        torch.save(float_model.state_dict(), BASELINE_SAVE_PATH)

print("\n[Float baseline best]")
print(f"Best epoch: {best_epoch_float}")
print(f"Best val wheeze F1: {best_val_f1_float:.3f}")
print(f"Best val threshold: {best_val_threshold_float:.2f}")


# =========================
# 10. 2단계: Binary FC 모델 생성 + weight 복사
# =========================

print("\n" + "=" * 70)
print("[2단계] Float baseline -> Binary FC 초기화")
print("=" * 70)

best_float_model = BetterCNN_FloatFC(num_classes=len(le.classes_)).to(DEVICE)
best_float_model.load_state_dict(torch.load(BASELINE_SAVE_PATH, map_location=DEVICE))
best_float_model.eval()

binary_model = BetterCNN_BinaryFC(num_classes=len(le.classes_)).to(DEVICE)

binary_model.features.load_state_dict(best_float_model.features.state_dict())
binary_model.fc1.load_state_dict(best_float_model.fc1.state_dict())

with torch.no_grad():
    binary_model.binary_fc.weight.copy_(best_float_model.fc2.weight.data)
    binary_model.binary_fc.bias.copy_(best_float_model.fc2.bias.data)

print("Float baseline weight를 Binary FC 모델에 복사 완료")


# =========================
# 11. 3단계: Binary FC fine-tuning
# =========================

print("\n" + "=" * 70)
print("[3단계] Binary FC fine-tuning 시작")
print("=" * 70)

criterion_binary = nn.CrossEntropyLoss(weight=class_weights)
optimizer_binary = torch.optim.Adam(binary_model.parameters(), lr=BINARY_LR)

best_val_f1_binary = -1
best_epoch_binary = -1
best_val_threshold_binary = None

for epoch in range(BINARY_EPOCHS):
    binary_model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer_binary.zero_grad()
        out = binary_model(xb)
        loss = criterion_binary(out, yb)
        loss.backward()
        optimizer_binary.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    val_targets, _, val_probs = evaluate_with_probs(binary_model, val_loader)
    val_th, val_f1, val_prec, val_rec, _ = search_best_threshold(
        val_targets,
        val_probs,
        wheeze_idx,
        non_wheeze_idx
    )

    print(
        f"[BinaryFT] Epoch {epoch + 1}/{BINARY_EPOCHS} | "
        f"Loss={avg_loss:.4f} | "
        f"Val th={val_th:.2f} | "
        f"P={val_prec:.3f}, R={val_rec:.3f}, F1={val_f1:.3f}"
    )

    if val_f1 > best_val_f1_binary:
        best_val_f1_binary = val_f1
        best_epoch_binary = epoch + 1
        best_val_threshold_binary = val_th
        torch.save(binary_model.state_dict(), BINARY_SAVE_PATH)

print("\n[Binary FC fine-tuning best]")
print(f"Best epoch: {best_epoch_binary}")
print(f"Best val wheeze F1: {best_val_f1_binary:.3f}")
print(f"Best val threshold: {best_val_threshold_binary:.2f}")


# =========================
# 12. 최종 test 평가
# =========================

best_binary_model = BetterCNN_BinaryFC(num_classes=len(le.classes_)).to(DEVICE)
best_binary_model.load_state_dict(torch.load(BINARY_SAVE_PATH, map_location=DEVICE))
best_binary_model.eval()

test_targets, test_preds, test_probs = evaluate_with_probs(best_binary_model, test_loader)

print_test_results(
    "Test Argmax 평가 | Binary FC fine-tuned",
    test_targets,
    test_preds,
    le.classes_
)

tuned_test_preds = []
for p in test_probs:
    if p[wheeze_idx] >= best_val_threshold_binary:
        tuned_test_preds.append(wheeze_idx)
    else:
        tuned_test_preds.append(non_wheeze_idx)

print_test_results(
    "Test Best-threshold 평가 (threshold from validation) | Binary FC fine-tuned",
    test_targets,
    tuned_test_preds,
    le.classes_
)

best_test_th, best_test_f1, best_test_prec, best_test_rec, best_test_preds = search_best_threshold(
    test_targets,
    test_probs,
    wheeze_idx,
    non_wheeze_idx
)

print("\n[Test 참고용 Fine Search 결과 | Binary FC fine-tuned]")
print(f"Best test-only threshold: {best_test_th:.2f}")
print(f"Wheeze precision={best_test_prec:.3f}, recall={best_test_rec:.3f}, f1={best_test_f1:.3f}")

print_test_results(
    "Test Fine Search best 결과 | Binary FC fine-tuned",
    test_targets,
    best_test_preds,
    le.classes_
)


# =========================
# 13. Binary weight 확인
# =========================

binary_weight = best_binary_model.binary_fc.weight.sign().detach().cpu().numpy()
binary_weight[binary_weight == 0] = 1

print("\nBinary FC weight shape:", binary_weight.shape)
print("Unique values in binary weight:", np.unique(binary_weight))
print("Binary FC weight sample:")
print(binary_weight[:5, :10])


# =========================
# 14. 파라미터 수
# =========================

total_params = sum(p.numel() for p in best_binary_model.parameters())
trainable_params = sum(p.numel() for p in best_binary_model.parameters() if p.requires_grad)

print("\n총 파라미터 수:", total_params)
print("학습 가능한 파라미터 수:", trainable_params)
print(f"Best float model saved to: {BASELINE_SAVE_PATH}")
print(f"Best binary fine-tuned model saved to: {BINARY_SAVE_PATH}")


# =========================
# 15. 회로팀 전달용 Export
# =========================

EXPORT_DIR = "sram_cim_export"

if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

os.makedirs(EXPORT_DIR, exist_ok=True)

PULSE_CONFIGS = [
    ("tmin0_tmax5", 0.0, 5.0),
]


def extract_feature_vector(model, loader, max_samples=1):
    model.eval()

    feature_list = []
    label_list = []

    with torch.no_grad():
        count = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE)

            x = model.features(xb)
            x = model.flatten(x)
            x = model.fc1(x)
            x = model.relu(x)

            feature_np = x.cpu().numpy()

            feature_list.append(feature_np)
            label_list.append(yb.numpy())

            count += feature_np.shape[0]

            if count >= max_samples:
                break

    features = np.concatenate(feature_list, axis=0)[:max_samples]
    labels = np.concatenate(label_list, axis=0)[:max_samples]

    return features, labels


def make_pulse_width(feature_1d, t_min_ns, t_max_ns):
    feature_1d = np.array(feature_1d, dtype=float).flatten()

    f_min = feature_1d.min()
    f_max = feature_1d.max()

    feature_norm = (feature_1d - f_min) / (f_max - f_min + 1e-6)
    pulse_width_ns = t_min_ns + feature_norm * (t_max_ns - t_min_ns)

    return feature_norm, pulse_width_ns


def export_hardwired_mapping(weight_1d, pulse_1d, class_name, export_dir):
    weight_1d = np.array(weight_1d, dtype=int).flatten()
    pulse_1d = np.array(pulse_1d, dtype=float).flatten()

    if len(weight_1d) != 64 or len(pulse_1d) != 64:
        raise ValueError("weight와 pulse는 모두 길이 64여야 합니다.")

    pos_indices = np.where(weight_1d == 1)[0]
    neg_indices = np.where(weight_1d == -1)[0]

    rows = []

    for local_idx, feature_idx in enumerate(pos_indices):
        rows.append({
            "feature_index": int(feature_idx),
            "weight": int(weight_1d[feature_idx]),
            "pulse_width_ns": float(pulse_1d[feature_idx]),
            "branch": "positive",
            "global_line": "RBL",
            "local_index_in_branch": int(local_idx),
            "suggested_row": int(local_idx // 8),
            "suggested_col": int(local_idx % 8),
        })

    for local_idx, feature_idx in enumerate(neg_indices):
        rows.append({
            "feature_index": int(feature_idx),
            "weight": int(weight_1d[feature_idx]),
            "pulse_width_ns": float(pulse_1d[feature_idx]),
            "branch": "negative",
            "global_line": "RBLB",
            "local_index_in_branch": int(local_idx),
            "suggested_row": int(local_idx // 8),
            "suggested_col": int(local_idx % 8),
        })

    df = pd.DataFrame(rows)

    save_path = os.path.join(export_dir, f"{class_name}_hardwired_mapping_all.csv")
    df.to_csv(save_path, index=False)

    print(f"\n[{class_name}] hard-wired mapping 저장 완료")
    print("positive 개수:", len(pos_indices))
    print("negative 개수:", len(neg_indices))
    print("저장 파일:", save_path)

    if len(pos_indices) > 32:
        print("주의: positive branch가 32개 초과 → 여러 cycle 필요")
    if len(neg_indices) > 32:
        print("주의: negative branch가 32개 초과 → 여러 cycle 필요")

    return df


binary_weight = best_binary_model.binary_fc.weight.sign().detach().cpu().numpy()
binary_weight[binary_weight == 0] = 1

nonwheeze_weight = binary_weight[non_wheeze_idx]
wheeze_weight = binary_weight[wheeze_idx]

print("\n[Binary weight 확인]")
print("shape:", binary_weight.shape)
print("unique:", np.unique(binary_weight))

features, labels = extract_feature_vector(
    best_binary_model,
    test_loader,
    max_samples=1
)

feature_1d = features[0]
sample_label = labels[0]

print("\n[Sample feature 확인]")
print("feature shape:", feature_1d.shape)
print("sample label:", sample_label)

for config_name, t_min, t_max in PULSE_CONFIGS:
    config_dir = os.path.join(EXPORT_DIR, config_name)
    os.makedirs(config_dir, exist_ok=True)

    feature_norm, pulse_width_ns = make_pulse_width(
        feature_1d,
        t_min_ns=t_min,
        t_max_ns=t_max
    )

    pd.DataFrame({
        "feature_index": np.arange(64),
        "feature_norm": feature_norm,
        "pulse_width_ns": pulse_width_ns
    }).to_csv(
        os.path.join(config_dir, "input_pulse_width.csv"),
        index=False
    )

    wheeze_df = export_hardwired_mapping(
        weight_1d=wheeze_weight,
        pulse_1d=pulse_width_ns,
        class_name="wheeze",
        export_dir=config_dir
    )

    nonwheeze_df = export_hardwired_mapping(
        weight_1d=nonwheeze_weight,
        pulse_1d=pulse_width_ns,
        class_name="nonwheeze",
        export_dir=config_dir
    )

    wheeze_expected_score = (wheeze_df["pulse_width_ns"] * wheeze_df["weight"]).sum()
    nonwheeze_expected_score = (nonwheeze_df["pulse_width_ns"] * nonwheeze_df["weight"]).sum()

    print("\n[Expected software score]")
    print("wheeze expected score:", wheeze_expected_score)
    print("nonwheeze expected score:", nonwheeze_expected_score)

    if wheeze_expected_score > nonwheeze_expected_score:
        print("expected class: wheeze")
    else:
        print("expected class: non-wheeze")

    with open(os.path.join(config_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"Pulse config: {config_name}\n")
        f.write(f"T_min = {t_min} ns\n")
        f.write(f"T_max = {t_max} ns\n")
        f.write("Formula: T_pulse = T_min + feature_norm * (T_max - T_min)\n")
        f.write("Current setting: T_pulse = feature_norm x 5 ns\n")
        f.write("weight +1 -> positive branch -> RBL\n")
        f.write("weight -1 -> negative branch -> RBLB\n")
        f.write("pulse_width_ns values are final RWL on-time in ns.\n")
        f.write(f"wheeze expected score = {wheeze_expected_score}\n")
        f.write(f"nonwheeze expected score = {nonwheeze_expected_score}\n")

print("\n==================================================")
print("[Export 완료]")
print("회로팀 전달용 폴더:", EXPORT_DIR)
print("생성된 pulse config:")
for name, _, _ in PULSE_CONFIGS:
    print("-", name)
print("==================================================")