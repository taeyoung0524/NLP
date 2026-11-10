# 10주차 Colab 실습 과제: 코드 토크나이저와 생성 코드 검증

## 안내

- `nlp_week_10_transformers-from-scratch.ipynb`의 모든 셀을 실행한 뒤, 같은 노트북 맨 아래에 새 셀을 추가하여 작업하시오.
- GPU 런타임(T4 등)을 권장한다. 문제 1–3은 CPU에서도 실행되지만, 문제 4는 GPU가 훨씬 빠르다.
- 각 문제의 코드 셀 출력과 질문에 대한 답(텍스트 셀, 2–4문장)을 남긴 `.ipynb` 파일을 제출하시오.
- 주의: 노트북 셀 65에서 `tokenizer`는 GPT-2 토크나이저가 아니라 새로 학습한 `reloaded_tokenizer`로 바뀌었다. 셀 73에서 `dataset`은 `codeparrot-train` 스트리밍 데이터셋으로 다시 지정되었다. 아래 코드는 이를 전제로 한다.
- 런타임이 재시작되었다면 노트북을 처음부터 다시 실행하시오.

---

## 문제 1. 세 토크나이저의 압축률 비교

세 토크나이저가 같은 Python 코드를 몇 개의 토큰으로 표현하는지 비교한다. 토크나이저 학습에 쓰지 않은 검증 데이터(`codeparrot-valid`)의 처음 50개 파일을 사용한다.

```python
from itertools import islice
from transformers import AutoTokenizer

gpt2_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizers = {"GPT-2 (50,257)": gpt2_tokenizer,
              "new_tokenizer (12,500)": new_tokenizer,
              "reloaded_tokenizer (32,768)": reloaded_tokenizer}

valid_data = load_dataset("transformersbook/codeparrot-valid",
                          split="validation", streaming=True)
valid_files = [ex["content"] for ex in islice(valid_data, 50)]
total_chars = sum(len(text) for text in valid_files)
print(f"Files: {len(valid_files)}, characters: {total_chars:,}")

counts = {}
for name, tok in tokenizers.items():
    # TODO: 50개 파일의 토큰 수 합계를 구하시오. (힌트: tok(text)["input_ids"])
    n_tokens = ...
    counts[name] = n_tokens
    print(f"{name:28s} tokens: {n_tokens:>8,}  chars/token: {total_chars/n_tokens:.2f}")

for name, n in counts.items():
    print(f"{name:28s} relative to GPT-2: {n/counts['GPT-2 (50,257)']:.2f}")
```

GPT-2 토크나이저에서 "sequence length is longer than ..." 경고가 나올 수 있다. 토큰 수만 세는 것이므로 무시해도 된다.

(a) 세 토크나이저의 총 토큰 수와 토큰당 문자 수를 표로 정리하시오.

(b) 새 토크나이저는 GPT-2보다 어휘가 작은데도 토큰 수가 더 적다. 그 이유를 설명하시오.

(c) 어휘 크기를 12,500에서 32,768로 늘렸을 때 줄어든 토큰 비율과, GPT-2에서 새 토크나이저로 바꿨을 때 줄어든 비율을 비교하시오. 어느 쪽 효과가 더 큰가?

---

## 문제 2. 들여쓰기 깊이와 토큰 수

들여쓰기가 깊어질 때 토큰 수가 어떻게 변하는지 확인한다.

```python
for depth in range(1, 5):
    line = "\n" + " " * (4 * depth) + "return x"
    print(f"depth {depth} ({4 * depth} spaces)")
    for name, tok in tokenizers.items():
        toks = tok(line).tokens()
        print(f"  {name:28s} {len(toks):2d} {toks}")
```

(a) 깊이 1–4에서 각 토크나이저의 토큰 수를 표로 정리하시오. GPT-2의 토큰 수를 깊이 d에 대한 식으로 나타내시오.

(b) 새 토크나이저에서 줄바꿈과 들여쓰기는 어떻게 표현되는가? 이것이 가능한 이유를 강의의 BPE 학습 과정과 연결하여 설명하시오.

(c) 문맥 길이가 128 토큰인 데모 모델에서 이 차이가 왜 중요한지 한두 문장으로 쓰시오.

---

## 문제 3. ConstantLengthDataset에서 EOS 토큰 추적하기

셀 75의 `ConstantLengthDataset`으로 길이 64의 시퀀스를 만들고, 파일 경계를 표시하는 EOS 토큰이 어디에 들어가는지 추적한다. 재현성을 위해 섞지 않은 `dataset`을 사용한다.

```python
eos_id = tokenizer.eos_token_id
print("EOS token:", tokenizer.eos_token, "id:", eos_id)

for num_of_sequences, n_take in [(5, 7), (20, 20)]:
    cld = ConstantLengthDataset(tokenizer, dataset, seq_length=64,
                                num_of_sequences=num_of_sequences,
                                chars_per_token=characters_per_token)
    print(f"\nnum_of_sequences={num_of_sequences}, "
          f"input_characters={cld.input_characters:.1f}")
    seqs = [s for _, s in zip(range(n_take), iter(cld))]
    print("lengths:", [len(s) for s in seqs])
    # TODO: 각 시퀀스에 들어 있는 EOS 토큰의 개수를 리스트로 출력하시오.
    print("EOS per sequence:", ...)

# TODO: EOS가 들어 있는 첫 번째 시퀀스를 찾아 EOS의 위치를 출력하고,
#       tokenizer.decode()로 해당 시퀀스를 복원해 출력하시오.
```

(a) 코드를 실행하기 전에, 셀 74에서 구한 `characters_per_token`으로 두 경우의 `input_characters`를 계산하시오. 출력과 일치하는지 확인하시오.

(b) `num_of_sequences=5`일 때 버퍼에는 파일이 몇 개 들어갔는가? ("Fill buffer" 출력의 개수를 보시오.) 7개 시퀀스에 EOS가 하나도 없는 이유를 설명하시오.

(c) `num_of_sequences=20`일 때 EOS가 들어 있는 시퀀스를 복원한 결과에서, EOS 앞뒤의 내용은 각각 무엇인가?

(d) 이 구현은 버퍼마다 길이가 모자란 마지막 조각을 버린다. 버퍼가 작을수록 어떤 문제가 생기는가? 원래 설정(`seq_length=1024`, `num_of_sequences=1024`)에서는 왜 이 문제가 거의 무시할 만한가?

---

## 문제 4. 생성된 코드를 단위 테스트로 평가하기

강의에서 코드 생성 모델은 BLEU보다 단위 테스트 통과율로 평가하는 편이 적절하다고 했다. `codeparrot-small`이 생성한 함수를 실제로 실행해 통과율을 구하고, 샘플링 온도의 영향을 확인한다.

```python
import re, signal, torch
from transformers import set_seed

# 노트북 셀 94 이후 generation은 codeparrot-small 파이프라인이다.
print(generation.model.name_or_path)

def generate_completions(pipe, prompt, num_completions=10, temperature=0.4,
                         max_new_tokens=64, seed=1):
    set_seed(seed)
    outputs = pipe(prompt, num_return_sequences=num_completions,
                   max_new_tokens=max_new_tokens, do_sample=True,
                   temperature=temperature, top_p=0.95, top_k=0)
    return [first_block(out["generated_text"][len(prompt):]) for out in outputs]

class Timeout(Exception):
    pass

def _handler(signum, frame):
    raise Timeout()

def passes_tests(prompt, completion, func_name, tests, seconds=2):
    """프롬프트와 생성 결과를 합쳐 실행하고, 모든 테스트를 통과하면 True."""
    namespace = {}
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)                 # 무한 루프 방지
    try:
        exec(prompt + completion, namespace)
        func = namespace[func_name]
        return all(func(*args) == expected for args, expected in tests)
    except Exception:
        return False
    finally:
        signal.alarm(0)

problems = [
    ("reverse_string", '''def reverse_string(s):
    """Return the reversed string."""''',
     [(("abc",), "cba"), (("",), "")]),
    ("mean", '''def mean(numbers):
    """Return the mean of a list of numbers."""''',
     [(([1, 2, 3],), 2), (([4],), 4)]),
    ("area_of_rectangle", '''def area_of_rectangle(a: float, b: float):
    """Return the area of the rectangle."""''',
     [((2, 3), 6), ((1.5, 2), 3.0)]),
    ("add", '''def add(a, b):
    """Return the sum of a and b."""''',
     [((1, 2), 3), ((-1, 1), 0)]),
]

for name, prompt, tests in problems:
    for temperature in [0.2, 0.8]:
        completions = generate_completions(generation, prompt,
                                           temperature=temperature)
        # TODO: 각 completion이 테스트를 통과하는지 리스트로 구하고,
        #       통과 개수/10 형식으로 출력하시오.
        results = ...
        print(f"{name:20s} T={temperature}: {sum(results)}/10")
```

생성된 코드는 신뢰할 수 없는 코드이므로, 이처럼 간단한 함수에 대해서만 Colab 같은 격리된 환경에서 실행하시오.

(a) 네 함수와 두 온도에 대한 통과 개수를 표로 정리하시오.

(b) 온도 0.2와 0.8에서 통과율이 어떻게 달라지는가? 6주차에 배운 샘플링 온도의 의미와 연결하여 설명하시오.

(c) 10개 중 하나라도 통과하면 성공으로 보는 기준(pass@10)을 쓰면, 어느 온도가 유리한 함수가 있는가? 실제 코드 자동 완성 도구라면 어떤 전략을 쓰겠는가?

(d) 실패한 생성 결과 중 하나를 골라, 그 코드가 왜 틀렸는지 설명하시오. BLEU로 평가했다면 이 결과를 어떻게 판단했을지도 함께 쓰시오.