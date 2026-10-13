# 기호주의: 세상 지식을 기호와 규칙으로 적어 놓는다
kind: dict[str, str] = {"참새": "새", "펭귄": "새", "타조": "새", "박쥐": "포유류"}
truth: dict[str, bool] = {"참새": True, "펭귄": False, "타조": False, "박쥐": True}


def score(name: str, fly) -> None:
    wrong: list[str] = [a for a in truth if fly(a) != truth[a]]
    ok: int = len(truth) - len(wrong)
    print(
        f"{name}: {ok}/{len(truth)} 맞음"
        + (f" - 틀린 것 {wrong}" if wrong else " - 전부 맞음")
    )


rule1: bool = lambda a: kind[a] == "새"
score("규칙 '새라면 난다'", rule1)

# 틀린 것을 보고 예외를 붙인다
rule2: bool = lambda a: (kind[a] == "새" and a not in ("펭귄", "타조")) or a == "박쥐"
score("예외 추가 '단 펭귄·타조 뺴고, 박쥐는 난다'", rule2)

# 규칙을 고쳤으니 이제 완벽한가? 세상에는 동물이 더 있다.
kind.update({"청둥오리": "새", "닭": "새", "벌": "곤충", "나비": "곤충"})
truth.update(
    {
        "청둥오리": True,
        "닭": False,
        "벌": True,
        "나비": True,
    }
)
score("\n같은 규칙에 4마리를 더 데려오면", rule2)
