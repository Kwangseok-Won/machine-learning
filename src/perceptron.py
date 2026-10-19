import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def perceptron(inputs, weights, bias):
    # ① 가중합: Σ wᵢxᵢ + b
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    # ② 활성화 함수 통과
    return sigmoid(z)


# 입력 2개짜리 퍼셉트론 (뉴런 하나)
inputs = [1.0, 2.0]
weights = [0.5, -1.0]
bias = 0.5

out = perceptron(inputs, weights, bias)
print(f"출력: {out:.3f}")  # 0-1 사이 값
# z = 0.5*1 + (-1)*2 + 0.5 = -1.0 → sigmoid(-1) ≈ 0.269
