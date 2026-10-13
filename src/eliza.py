import re

# ELIZA의 전부 - 정규식으로 받아치는 규칙 목록. 사전도, 의미도 없다.
rules = [
    (r"저는 (.*)고 싶어요", "왜 {1}고 싶으신가요?"),
    (r"저는 (.*)해요", "왜 {1}하다고 느끼시나요?"),
    (r"저는 (.*)어요", "언제부터 {1}었나요?"),
    (r"(.*) 때문에 (.*)", "{1}이 없었다면 어땠을까요?"),
    (r"(.*)\?", "그건 왜 물어보시나요?"),
    (r"(.*)", "조금 더 이야기해 주세요."),
]


def eliza(said):
    for pattern, reply in rules:
        m = re.fullmatch(pattern, said)
        if m:
            for i, g in enumerate(m.groups(), 1):
                reply = reply.replace("{%d}" % i, g)
            return reply


for said in ["저는 요즘 우울해요", "저는 좀 쉬고 싶어요", "시험 때문에 잠이 안 와요", "당신은 누구세요?", "저는 어제 개를 봤어요"]:
    print(f"나    : {said}")
    print(f"ELIZA : {eliza(said)}\n")
