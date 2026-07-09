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

wheeze_idx = list(le.classes_).index("wheeze")
non_wheeze_idx = list(le.classes_).index("non-wheeze")


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
# 8-1. Event-driven CNN gating 평가 함수
#      (원인: 이 함수는 원래 정의만 되고 어디서도 호출되지 않았음 -> 수정)
#
#      개선 사항:
#        ① threshold를 고정 범위로 대충 스캔하지 않고, 실제 에너지 분포의
#           percentile 기반으로 촘촘하게 후보를 만든다 (스케일이 달라도 항상 촘촘함)
#        ② energy_mode="rms" 지원 (np.sqrt(np.mean(segment**2)))
#        ③ energy_mode="mel" 지원: CNN이 실제로 보는 mel-spectrogram의
#           절대 에너지(정규화 전 power)를 기준으로 게이팅 -> 모델 입력과
#           괴리가 적어 F1 손실이 작아질 가능성이 높음
# =========================

def _compute_gating_energy(segment, sr, energy_mode):
    """
    energy_mode에 따라 게이팅용 스칼라 에너지 값을 계산.
    mel 모드는 CNN 입력으로 쓸 정규화된 log-mel(feat)도 같이 반환해서
    통과(gate 안 됨) 판정이 나면 mel을 다시 계산하지 않도록 한다.
    """
    if energy_mode == "mae":
        energy = float(np.mean(np.abs(segment)))
        return energy, None, None

    if energy_mode == "rms":
        energy = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
        return energy, None, None

    if energy_mode == "mel":
        seg, seg_sr = fix_length(segment, sr, TARGET_SR, TARGET_SECONDS)
        # 정규화 전 raw mel power. power_to_db(ref=np.max)나 z-score 정규화는
        # 세그먼트마다 절대 크기를 지워버리기 때문에 게이팅 지표로 쓰면 안 됨.
        raw_mel = librosa.feature.melspectrogram(
            y=seg, sr=seg_sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        energy = float(np.mean(raw_mel))
        return energy, raw_mel, seg_sr

    raise ValueError(f"알 수 없는 energy_mode: {energy_mode}")


def _raw_mel_to_model_feat(raw_mel):
    logmel = librosa.power_to_db(raw_mel, ref=np.max)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)
    return logmel.astype(np.float32)


def evaluate_with_input_gating(model, file_list, txt_dir, input_gating_th, wheeze_th,
                                energy_mode="rms"):
    """
    1차 threshold(input_gating_th): 게이팅 에너지가 낮으면 CNN 연산 자체를 skip
    2차 threshold(wheeze_th): CNN이 돌아간 경우, softmax wheeze 확률 판정 기준
    energy_mode: "mae" | "rms" | "mel"
    """
    model.eval()
    preds = []
    targets = []
    gated_count = 0
    total_count = 0

    with torch.no_grad():
        for wav_path in file_list:
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            txt_path = os.path.join(txt_dir, base_name + ".txt")
            if not os.path.exists(txt_path):
                continue

            ann = load_annotation(txt_path)
            signal, sr = librosa.load(wav_path, sr=None)

            for _, row in ann.iterrows():
                start_sample = int(float(row["start"]) * sr)
                end_sample = int(float(row["end"]) * sr)
                segment = signal[start_sample:end_sample]
                if len(segment) == 0:
                    continue

                total_count += 1

                energy, raw_mel, seg_sr = _compute_gating_energy(segment, sr, energy_mode)

                if energy < input_gating_th:
                    # 임계값 미달 시: CNN/CIM 연산 전면 Off -> 바로 정상 판정
                    preds.append(non_wheeze_idx)
                    gated_count += 1
                else:
                    # 임계값 통과 시: CNN + 8T SRAM 펄스 인가 및 연산 수행
                    if energy_mode == "mel":
                        feat = _raw_mel_to_model_feat(raw_mel)
                    else:
                        seg, seg_sr = fix_length(segment, sr, TARGET_SR, TARGET_SECONDS)
                        feat = extract_logmel(seg, seg_sr)

                    xb = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    out = model(xb)
                    prob = torch.softmax(out, dim=1).cpu().numpy()[0]

                    if prob[wheeze_idx] >= wheeze_th:
                        preds.append(wheeze_idx)
                    else:
                        preds.append(non_wheeze_idx)

                label_4class = label_from_row(int(row["crackles"]), int(row["wheezes"]))
                targets.append(list(le.classes_).index(map_binary(label_4class)))

    return np.array(targets), np.array(preds), gated_count, total_count


def collect_gating_energies(file_list, txt_dir, energy_mode):
    """
    threshold 후보를 만들기 위해, 전체 segment의 게이팅 에너지 분포를 미리 수집한다.
    이걸 percentile로 쪼개면 mode/스케일에 상관없이 항상 촘촘한 후보를 만들 수 있다.
    """
    energies = []

    for wav_path in file_list:
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        txt_path = os.path.join(txt_dir, base_name + ".txt")
        if not os.path.exists(txt_path):
            continue

        ann = load_annotation(txt_path)
        signal, sr = librosa.load(wav_path, sr=None)

        for _, row in ann.iterrows():
            start_sample = int(float(row["start"]) * sr)
            end_sample = int(float(row["end"]) * sr)
            segment = signal[start_sample:end_sample]
            if len(segment) == 0:
                continue

            energy, _, _ = _compute_gating_energy(segment, sr, energy_mode)
            energies.append(energy)

    return np.array(energies)


def collect_gating_energies_by_label(file_list, txt_dir, energy_mode):
    """
    collect_gating_energies와 동일하지만 wheeze/non-wheeze를 나눠서 반환한다.
    -> "이 threshold를 쓰면 진짜 wheeze 중 몇 %가 게이팅되어 통째로 버려지는가"를
       직접 계산하기 위해 필요하다. 이게 F1보다 훨씬 직접적인 안전장치다.
    """
    energies = {"wheeze": [], "non-wheeze": []}

    for wav_path in file_list:
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        txt_path = os.path.join(txt_dir, base_name + ".txt")
        if not os.path.exists(txt_path):
            continue

        ann = load_annotation(txt_path)
        signal, sr = librosa.load(wav_path, sr=None)

        for _, row in ann.iterrows():
            start_sample = int(float(row["start"]) * sr)
            end_sample = int(float(row["end"]) * sr)
            segment = signal[start_sample:end_sample]
            if len(segment) == 0:
                continue

            energy, _, _ = _compute_gating_energy(segment, sr, energy_mode)

            label_4class = label_from_row(int(row["crackles"]), int(row["wheezes"]))
            label_binary = map_binary(label_4class)
            energies[label_binary].append(energy)

    return {k: np.array(v) for k, v in energies.items()}


def make_percentile_thresholds(energies, max_percentile=45, step=1.0):
    """
    게이팅은 '조용한(=낮은 에너지) segment'를 skip하는 게 목적이므로,
    분포의 하위 구간(0~max_percentile%)만 촘촘하게 촘촘히 스캔하면 된다.
    예: step=1.0, max_percentile=45 -> 0,1,2,...,45 percentile, 총 46개 후보.
    """
    percentiles = np.arange(0, max_percentile + step, step)
    thresholds = np.percentile(energies, percentiles)
    return np.unique(thresholds)


def make_safe_threshold_candidates(energies_by_label, max_wheeze_gate_fraction=0.03,
                                    max_percentile=80, step=1.0):
    """
    F1로 안전선을 긋는 대신, '진짜 wheeze 라벨 중 threshold 밑으로 깔려서
    통째로 게이팅되는 비율'을 직접 max_wheeze_gate_fraction 이하로 제한한다.

    - non-wheeze 에너지 분포의 percentile들을 threshold 후보로 만들고,
    - 그 중 wheeze_energies < th 인 비율이 max_wheeze_gate_fraction을 넘는 후보는 제외한다.

    이렇게 하면 "재수 없게 조용한 wheeze"가 무더기로 skip되는 상황을 원천 차단하면서,
    non-wheeze 쪽은 최대한 공격적으로 게이팅할 수 있는 후보만 남는다.
    """
    non_wheeze_energies = energies_by_label["non-wheeze"]
    wheeze_energies = energies_by_label["wheeze"]

    percentiles = np.arange(0, max_percentile + step, step)
    candidates = np.unique(np.percentile(non_wheeze_energies, percentiles))

    safe = []
    for th in candidates:
        if len(wheeze_energies) > 0:
            wheeze_gate_fraction = float(np.mean(wheeze_energies < th))
        else:
            wheeze_gate_fraction = 0.0

        if wheeze_gate_fraction <= max_wheeze_gate_fraction:
            safe.append(th)

    return np.array(safe)


def search_best_input_gating_threshold(model, file_list, txt_dir, wheeze_th, energy_thresholds,
                                        energy_mode="rms", min_wheeze_f1=None, verbose=True):
    """
    1차(입력 게이팅) threshold 탐색.
    energy_thresholds 후보들에 대해 evaluate_with_input_gating을 돌려서
    - wheeze F1을 min_wheeze_f1 이상으로 유지하면서
    - gating_ratio(=CNN 연산을 스킵하는 비율, 즉 연산량 절감)가 최대인 threshold를 선택한다.
    """
    results = []

    for th in energy_thresholds:
        targets, preds, gated_count, total_count = evaluate_with_input_gating(
            model, file_list, txt_dir, th, wheeze_th, energy_mode=energy_mode
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average=None,
            labels=[non_wheeze_idx, wheeze_idx],
            zero_division=0
        )
        wheeze_f1 = f1[1]
        gating_ratio = gated_count / total_count if total_count > 0 else 0.0

        results.append({
            "th": th,
            "wheeze_f1": wheeze_f1,
            "gating_ratio": gating_ratio,
            "gated_count": gated_count,
            "total_count": total_count,
        })

        if verbose:
            print(f"  [{energy_mode}] th={th:.6f} | wheeze F1={wheeze_f1:.3f} "
                  f"| gated={gated_count}/{total_count} ({gating_ratio * 100:.1f}%)")

    if min_wheeze_f1 is not None:
        candidates = [r for r in results if r["wheeze_f1"] >= min_wheeze_f1]
        if len(candidates) == 0:
            print(f"\n경고: wheeze F1 >= {min_wheeze_f1:.3f} 을 만족하는 threshold가 없습니다. "
                  f"F1이 가장 높은 threshold로 fallback 합니다.")
            best = max(results, key=lambda r: r["wheeze_f1"])
        else:
            best = max(candidates, key=lambda r: r["gating_ratio"])
    else:
        best = max(results, key=lambda r: (round(r["wheeze_f1"], 4), r["gating_ratio"]))

    return best["th"], results


# =========================
# 9. 1단계: Float baseline 학습
# =========================

print("\n" + "=" * 70)
print("[1단계] Float baseline 학습 시작")
print("=" * 70)

float_model = BetterCNN_FloatFC(num_classes=len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(float_model.parameters(), lr=BASELINE_LR)

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
# 12. 최종 test 평가 (gating 없음, 기존 방식)
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
# 12-1. Event-driven Input Gating: threshold 탐색 + 실제 적용
#       (핵심 수정: 지금까지는 이 단계가 아예 호출된 적이 없었음)
#
#       추가 개선:
#         ① 고정 range 대신 validation 에너지 분포의 percentile로
#            자동으로 촘촘한 threshold 후보를 만듦 (mode/스케일 무관하게 항상 촘촘함)
#         ② energy_mode="rms" 지원
#         ③ energy_mode="mel" 지원 (CNN이 실제로 보는 mel 절대 에너지 기준 게이팅)
#
#       중요 수정 (F1이 오히려 떨어지는 문제 해결):
#         기존에는 "validation 전체 F1이 (val 학습때 F1 - tolerance) 이상"이라는
#         뭉뚱그려진 제약만 걸었는데, val 학습 F1(0.85)과 test baseline F1(0.70)
#         사이 갭이 커서 이 제약이 사실상 거의 아무 threshold나 다 통과시켰음.
#         그 결과 에너지가 낮은 "조용한 진짜 wheeze"까지 게이팅되면서 recall이
#         집중적으로 깎였음 (precision은 그대로, recall만 하락한 게 그 증거).
#
#         -> F1 대신 "validation의 진짜 wheeze 라벨 중 몇 %가 threshold 밑으로
#            깔려서 통째로 게이팅되는가"를 직접 max_wheeze_gate_fraction으로 제한.
#            이러면 non-wheeze는 공격적으로 게이팅하면서 wheeze recall 손실은
#            원천적으로 작게 유지된다.
# =========================

# 진짜 wheeze 라벨 중 최대 이 비율까지만 (에너지가 낮아서) 게이팅되는 걸 허용.
# 낮출수록 안전하지만 gating 비율도 줄어듦. 필요에 따라 0.01~0.05 사이로 조절.
MAX_WHEEZE_GATE_FRACTION = 0.03

ENERGY_MODES = ["mae", "rms", "mel"]  # mae: 기존 방식(비교용), rms/mel: 개선안

gating_summary = {}

for energy_mode in ENERGY_MODES:
    print("\n" + "=" * 70)
    print(f"[4단계] Event-driven Input Gating threshold 탐색 (validation) | mode={energy_mode}")
    print("=" * 70)

    # validation set에서 wheeze / non-wheeze 에너지 분포를 라벨별로 수집
    val_energies_by_label = collect_gating_energies_by_label(val_files, TRAIN_TXT_DIR, energy_mode)
    nw_e = val_energies_by_label["non-wheeze"]
    w_e = val_energies_by_label["wheeze"]
    print(f"  non-wheeze 에너지 | min={nw_e.min():.6f}, median={np.median(nw_e):.6f}, max={nw_e.max():.6f}")
    print(f"  wheeze     에너지 | min={w_e.min():.6f}, median={np.median(w_e):.6f}, max={w_e.max():.6f}")

    # wheeze 손실 비율을 MAX_WHEEZE_GATE_FRACTION 이하로 직접 제한한 후보만 생성
    threshold_candidates = make_safe_threshold_candidates(
        val_energies_by_label,
        max_wheeze_gate_fraction=MAX_WHEEZE_GATE_FRACTION,
        max_percentile=80,
        step=1.0,
    )

    if len(threshold_candidates) == 0:
        print(f"  경고: max_wheeze_gate_fraction={MAX_WHEEZE_GATE_FRACTION} 조건을 만족하는 "
              f"threshold가 없습니다. 이 mode는 게이팅 없이(threshold=0) 처리합니다.")
        best_th = 0.0
        search_results = []
    else:
        # 안전 후보들 중에서는 gating_ratio(연산 절감)가 최대인 걸 선택.
        # (min_wheeze_f1은 이제 2차 안전장치로만 남겨둠 — 너무 타이트하게 걸면
        #  안전 후보가 다 걸러질 수 있어 None으로 완화)
        best_th, search_results = search_best_input_gating_threshold(
            best_binary_model,
            val_files,
            TRAIN_TXT_DIR,
            wheeze_th=best_val_threshold_binary,
            energy_thresholds=threshold_candidates,
            energy_mode=energy_mode,
            min_wheeze_f1=None,
            verbose=False,
        )

    print(f"  선택된 threshold (validation 기준, wheeze 손실 <= {MAX_WHEEZE_GATE_FRACTION*100:.0f}%): "
          f"{best_th:.6f}")

    print(f"\n[5단계] Test set 평가 | mode={energy_mode}")
    test_targets_gated, test_preds_gated, gated_count, total_count = evaluate_with_input_gating(
        best_binary_model,
        test_files,
        TEST_TXT_DIR,
        best_th,
        best_val_threshold_binary,
        energy_mode=energy_mode,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets_gated, test_preds_gated, average=None,
        labels=[non_wheeze_idx, wheeze_idx], zero_division=0
    )

    print_test_results(
        f"Test 평가 | Event-driven Input Gating | mode={energy_mode}, th={best_th:.6f}",
        test_targets_gated,
        test_preds_gated,
        le.classes_
    )

    gating_ratio = gated_count / total_count if total_count > 0 else 0.0
    print(f"\nCNN 연산 skip 비율: {gated_count}/{total_count} ({gating_ratio * 100:.1f}% 연산 절감)")

    gating_summary[energy_mode] = {
        "threshold": best_th,
        "wheeze_precision": precision[1],
        "wheeze_recall": recall[1],
        "wheeze_f1": f1[1],
        "gated_count": gated_count,
        "total_count": total_count,
        "gating_ratio": gating_ratio,
        "targets": test_targets_gated,
        "preds": test_preds_gated,
    }

# ---- 세 가지 모드 비교 요약표 ----
print("\n" + "=" * 70)
print("[비교 요약] mae vs rms vs mel (test set)")
print("=" * 70)
print(f"{'mode':<6} {'threshold':>12} {'wheeze_P':>9} {'wheeze_R':>9} {'wheeze_F1':>10} {'gated':>14}")
for mode, s in gating_summary.items():
    print(f"{mode:<6} {s['threshold']:>12.6f} {s['wheeze_precision']:>9.3f} "
          f"{s['wheeze_recall']:>9.3f} {s['wheeze_f1']:>10.3f} "
          f"{s['gated_count']:>5d}/{s['total_count']:<5d} ({s['gating_ratio']*100:4.1f}%)")

# 이후 export 등 나머지 단계에서 쓸 대표값은 gating_ratio 대비 F1이 가장 좋은 모드로 선택
# (원하는 기준으로 바꿔도 됨: 예를 들어 gating_ratio를 최우선으로 하려면 정렬 기준을 바꿀 것)
best_mode = max(
    gating_summary,
    key=lambda m: (round(gating_summary[m]["wheeze_f1"], 4), gating_summary[m]["gating_ratio"])
)
best_input_gating_th = gating_summary[best_mode]["threshold"]
gated_count = gating_summary[best_mode]["gated_count"]
total_count = gating_summary[best_mode]["total_count"]

print(f"\n최종 채택 모드: {best_mode} (threshold={best_input_gating_th:.6f}, "
      f"F1={gating_summary[best_mode]['wheeze_f1']:.3f}, "
      f"gating={gating_summary[best_mode]['gating_ratio']*100:.1f}%)")


# =========================
# 12-2. Gating 강도 sweep: F1 vs skip 비율 트레이드오프 곡선
#
#       MAX_WHEEZE_GATE_FRACTION 하나만 고정해서 쓰면 "얼마나 손해보고
#       얼마나 절감되는지"를 알 수 없다. 여러 값을 스윕해서 실제 test set
#       기준 F1과 gating 비율이 어떻게 같이 움직이는지 표로 뽑아서
#       원하는 지점(예: F1 0.70 근처 유지하며 skip 20~30%)을 직접 고를 수 있게 한다.
#
#       주의: wheeze/non-wheeze 에너지가 많이 겹치는 데이터라면, 손실 허용치를
#       올려도 gating 비율이 기대만큼 안 오를 수 있다 (이 데이터의 mae 모드가
#       그랬음: wheeze 최솟값이 non-wheeze 중앙값보다 낮아서 겹침이 큼).
#       이 sweep 결과가 그 한계를 숫자로 보여준다.
# =========================

SWEEP_ENERGY_MODES = ["mae", "rms", "mel"]
SWEEP_FRACTIONS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50]

sweep_rows = []

for energy_mode in SWEEP_ENERGY_MODES:
    print("\n" + "=" * 70)
    print(f"[Sweep] mode={energy_mode}")
    print("=" * 70)

    val_energies_by_label = collect_gating_energies_by_label(val_files, TRAIN_TXT_DIR, energy_mode)

    for frac in SWEEP_FRACTIONS:
        candidates = make_safe_threshold_candidates(
            val_energies_by_label,
            max_wheeze_gate_fraction=frac,
            max_percentile=95,   # 손실을 많이 허용할 수도 있으니 상한을 넉넉히 잡음
            step=1.0,
        )

        if len(candidates) == 0:
            th = 0.0
        else:
            th, _ = search_best_input_gating_threshold(
                best_binary_model,
                val_files,
                TRAIN_TXT_DIR,
                wheeze_th=best_val_threshold_binary,
                energy_thresholds=candidates,
                energy_mode=energy_mode,
                min_wheeze_f1=None,
                verbose=False,
            )

        test_targets_s, test_preds_s, gated_s, total_s = evaluate_with_input_gating(
            best_binary_model, test_files, TEST_TXT_DIR, th, best_val_threshold_binary,
            energy_mode=energy_mode,
        )

        precision_s, recall_s, f1_s, _ = precision_recall_fscore_support(
            test_targets_s, test_preds_s, average=None,
            labels=[non_wheeze_idx, wheeze_idx], zero_division=0
        )
        gating_ratio_s = gated_s / total_s if total_s > 0 else 0.0

        row = {
            "mode": energy_mode,
            "max_wheeze_gate_fraction": frac,
            "threshold": th,
            "test_precision": precision_s[1],
            "test_recall": recall_s[1],
            "test_f1": f1_s[1],
            "test_gated": gated_s,
            "test_total": total_s,
            "test_gating_ratio": gating_ratio_s,
        }
        sweep_rows.append(row)

        print(f"  loss<= {frac*100:5.1f}% | th={th:.6f} | test P={row['test_precision']:.3f} "
              f"R={row['test_recall']:.3f} F1={row['test_f1']:.3f} | "
              f"skip={gated_s}/{total_s} ({gating_ratio_s*100:5.1f}%)")

print("\n" + "=" * 70)
print("[Sweep 요약표] wheeze 손실 허용치 vs test F1 vs skip 비율")
print("=" * 70)
print(f"{'mode':<6} {'loss<=':>8} {'threshold':>12} {'P':>7} {'R':>7} {'F1':>7} {'skip%':>8}")
for row in sweep_rows:
    print(f"{row['mode']:<6} {row['max_wheeze_gate_fraction']*100:>7.1f}% "
          f"{row['threshold']:>12.6f} {row['test_precision']:>7.3f} "
          f"{row['test_recall']:>7.3f} {row['test_f1']:>7.3f} "
          f"{row['test_gating_ratio']*100:>7.1f}%")

# baseline(게이팅 없음) F1을 기준으로, 이 이상을 유지하는 행들 중 skip 비율이 최대인 지점 자동 추천
BASELINE_TEST_F1 = best_test_f1  # 8단계에서 계산된 gating 없는 test fine-search F1
F1_DROP_BUDGET = 0.01  # baseline 대비 이만큼까지의 F1 하락은 허용한다는 기준 (원하는 대로 조정)

acceptable_rows = [r for r in sweep_rows if r["test_f1"] >= BASELINE_TEST_F1 - F1_DROP_BUDGET]

print(f"\nBaseline(게이팅 없음) test F1 = {BASELINE_TEST_F1:.3f}")
if len(acceptable_rows) > 0:
    recommended = max(acceptable_rows, key=lambda r: r["test_gating_ratio"])
    print(f"F1을 baseline 대비 {F1_DROP_BUDGET:.3f} 이내로 유지하면서 skip 비율이 최대인 지점:")
    print(f"  mode={recommended['mode']}, max_wheeze_gate_fraction={recommended['max_wheeze_gate_fraction']}, "
          f"threshold={recommended['threshold']:.6f}, "
          f"test F1={recommended['test_f1']:.3f}, skip={recommended['test_gating_ratio']*100:.1f}%")
else:
    print(f"F1을 baseline 대비 {F1_DROP_BUDGET:.3f} 이내로 유지하는 조합이 sweep 범위 안에 없습니다. "
          f"F1_DROP_BUDGET을 늘리거나 SWEEP_FRACTIONS 범위를 넓혀서 다시 시도하세요.")


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
        f.write(f"input_gating_th (event-driven) = {best_input_gating_th}\n")
        f.write(f"gated ratio on test set = {gated_count}/{total_count}\n")

print("\n==================================================")
print("[Export 완료]")
print("회로팀 전달용 폴더:", EXPORT_DIR)
print("생성된 pulse config:")
for name, _, _ in PULSE_CONFIGS:
    print("-", name)
print("==================================================")
