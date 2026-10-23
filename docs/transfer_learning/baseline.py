# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "pillow"]
# ///
"""
전이학습 해커톤 기준선 — Flowers-102 (102종, 학습 1,020장 = 한 종에 10장), ResNet18.

  uv run baseline.py

학습 → 검증 정확도 출력 → submission.csv 저장까지 한 번에 한다.
고칠 곳은 [바꿀 곳 1]~[5]. 그 밖을 고쳐도 되지만 규칙(README)은 지킨다:
모델은 resnet18 ImageNet 가중치, 외부 데이터 금지, 입력 224px 이하.
"""

import torch
from torch import nn
from torchvision import models, transforms

from common import (IMAGENET_STATS, N_CLASSES, SEED, TestImages, accuracy, device, labeled, loader,
                    write_submission)

IMAGE_SIZE = 160
BATCH = 32
EPOCHS = 10
LR = 1e-3


# ── [바꿀 곳 1] 변환 ── 학습용과 평가용을 따로 둔다
def train_transform() -> transforms.Compose:
    return transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(), transforms.Normalize(*IMAGENET_STATS)])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(), transforms.Normalize(*IMAGENET_STATS)])


# ── [바꿀 곳 2] 모델 ── 무엇을 얼리고 무엇을 학습하나
def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False  # 몸통은 ImageNet 에서 배운 그대로
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)  # 새 머리만 학습한다
    return model


# ── [바꿀 곳 3] 손실 ──
def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, y)


# ── [바꿀 곳 4] 옵티마이저 · 학습률 스케줄 ──
def build_optimizer(model: nn.Module) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    return optimizer, None


# ── [바꿀 곳 5] 학습 루프 ── 몇 epoch, 언제 멈추고 어느 시점의 모델을 남기나
def train(model: nn.Module, dev: torch.device) -> nn.Module:
    train_data = loader(labeled("train", train_transform()), BATCH, shuffle=True)
    val_data = loader(labeled("val", eval_transform()))
    optimizer, scheduler = build_optimizer(model)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_data:
            optimizer.zero_grad()
            loss_fn(model(x.to(dev)), y.to(dev)).backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        print(f"epoch {epoch:2d}  검증 {accuracy(model, val_data, dev):.4f}")
    return model  # 마지막 epoch 의 모델


@torch.no_grad()
def predict(model: nn.Module, dev: torch.device) -> tuple[list[str], list[int]]:
    """시험 사진마다 가장 그럴듯한 라벨. 테스트 시 증강(TTA)을 넣을 자리."""
    model.eval()
    ids, labels = [], []
    for x, names in loader(TestImages(eval_transform())):
        labels += model(x.to(dev)).argmax(1).cpu().tolist()
        ids += list(names)
    return ids, labels


def main() -> None:
    torch.manual_seed(SEED)
    dev = device()
    print(f"장치: {dev}")
    model = train(build_model().to(dev), dev)
    val = accuracy(model, loader(labeled("val", eval_transform())), dev)
    print(f"최종 검증 정확도 {val:.4f}  @val_acc={val:.4f}")
    print(f"제출 파일: {write_submission(*predict(model, dev))}")


if __name__ == "__main__":
    main()
