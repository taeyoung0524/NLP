# 2026년 2학기 성신여자대학교 자연어처리 수업 중간고사 정답 및 풀이

## 문제 1 풀이. Scaled Dot-Product Self-Attention

### 1-1. Query, key, value 계산

`learn`의 입력 벡터는 $x_3=[1,1]$이므로 query는

$$
q_3=x_3W_Q
=[1,1]
\begin{bmatrix}
1&0\\
1&1
\end{bmatrix}
=[2,1]
$$

이다.

모든 key는 다음과 같다.

$$
k_1=[1,0]W_K=[1,1],
\qquad
k_2=[0,1]W_K=[0,1],
$$

$$
k_3=[1,1]W_K=[1,2],
\qquad
k_4=[-1,1]W_K=[-1,0].
$$

모든 value는 다음과 같다.

$$
v_1=[1,0]W_V=[1,0],
\qquad
v_2=[0,1]W_V=[0,2],
$$

$$
v_3=[1,1]W_V=[1,2],
\qquad
v_4=[-1,1]W_V=[-1,2].
$$

### 1-2. Scaled attention score 계산

$$
s_1=\frac{[2,1]\cdot[1,1]}{\sqrt2}
=\frac{3}{\sqrt2}\approx2.121,
$$

$$
s_2=\frac{[2,1]\cdot[0,1]}{\sqrt2}
=\frac{1}{\sqrt2}\approx0.707,
$$

$$
s_3=\frac{[2,1]\cdot[1,2]}{\sqrt2}
=\frac{4}{\sqrt2}\approx2.828,
$$

$$
s_4=\frac{[2,1]\cdot[-1,0]}{\sqrt2}
=\frac{-2}{\sqrt2}\approx-1.414.
$$

따라서

$$
\boxed{s\approx[2.121,\ 0.707,\ 2.828,\ -1.414]}
$$

이다.

### 1-3. Softmax로 attention weight 계산

각 score의 지수값과 그 합은

$$
[e^{2.121},e^{0.707},e^{2.828},e^{-1.414}]
\approx[8.342,\ 2.028,\ 16.919,\ 0.243],
$$

$$
Z\approx8.342+2.028+16.919+0.243=27.532
$$

이다. 각 지수값을 $Z$로 나누면

$$
\boxed{[a_1,a_2,a_3,a_4]\approx[0.303,\ 0.074,\ 0.615,\ 0.009]}
$$

이다.

### 1-4. Attention output 계산

$$
h_3
=a_1[1,0]+a_2[0,2]+a_3[1,2]+a_4[-1,2]
=[a_1+a_3-a_4,\ 2(a_2+a_3+a_4)]
$$

이므로

$$
h_3\approx[0.303+0.615-0.009,\ 2(0.074+0.615+0.009)]
$$

이다. 반올림하지 않은 weight로 계산하면

$$
\boxed{h_3\approx[0.909,\ 1.394]}
$$

이다. (반올림된 weight를 그대로 대입하면 두 번째 성분이 $1.396$이 되며, 이는 반올림 오차이다.)

---

## 문제 2 풀이. Decoder Self-Attention의 Causal Masking

각 query 위치는 자기 자신과 앞선 위치만 볼 수 있다. 즉 행 $i$, 열 $j$에 대해 $j\le i$이면 1, $j>i$이면 0이다. 따라서 causal mask는 다음과 같다.

$$
\boxed{
M=
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
}
$$

---

## 문제 3 풀이. Token Classification과 Label Alignment

특수 토큰 `[CLS]`와 `[SEP]`에는 $-100$을 할당한다. `York`는 `Yo`와 `##rk`로 나뉘므로 첫 번째 subword인 `Yo`에는 `I-LOC`, 나머지 subword인 `##rk`에는 $-100$을 할당한다.

| 토큰 위치 | 토큰 | word_id | 정렬된 레이블 |
|---:|---|---:|---|
| 0 | `[CLS]` | None | -100 |
| 1 | Alice | 0 | B-PER |
| 2 | visited | 1 | O |
| 3 | New | 2 | B-LOC |
| 4 | Yo | 3 | I-LOC |
| 5 | `##rk` | 3 | -100 |
| 6 | City | 4 | I-LOC |
| 7 | `[SEP]` | None | -100 |

따라서 정렬된 레이블 시퀀스는

$$
\boxed{
[-100,\ \text{B-PER},\ \text{O},\ \text{B-LOC},\ \text{I-LOC},\ -100,\ \text{I-LOC},\ -100]
}
$$

이다.

---

## 문제 4 풀이. Sequence Classification의 Logit과 Cross-Entropy

### 4-1. Softmax 확률

예시 1:

$$
[e^{2},e^{1},e^{0}]\approx[7.389,\ 2.718,\ 1.000],\qquad Z\approx11.107,
$$

$$
\operatorname{softmax}([2,1,0])\approx[0.665,\ 0.245,\ 0.090].
$$

예시 2:

$$
[e^{0},e^{1.2},e^{1}]\approx[1.000,\ 3.320,\ 2.718],\qquad Z\approx7.038,
$$

$$
\operatorname{softmax}([0,1.2,1])\approx[0.142,\ 0.472,\ 0.386].
$$

예시 3:

$$
[e^{0.5},e^{0.4},e^{1.2}]\approx[1.649,\ 1.492,\ 3.320],\qquad Z\approx6.461,
$$

$$
\operatorname{softmax}([0.5,0.4,1.2])\approx[0.255,\ 0.231,\ 0.514].
$$

### 4-2. 예측 레이블 ID

확률(또는 logit)이 가장 큰 레이블을 선택한다.

| 예시 | 예측 레이블 ID | 정답 레이블 ID | 정답 여부 |
|---:|---:|---:|---|
| 1 | 0 (joy) | 0 | 정답 |
| 2 | 1 (sadness) | 1 | 정답 |
| 3 | 2 (anger) | 0 | 오답 |

### 4-3. Cross-entropy loss

$$
L_1=-\ln(0.665)\approx0.408,
$$

$$
L_2=-\ln(0.472)\approx0.751,
$$

$$
L_3=-\ln(0.255)\approx1.366.
$$

예시 3은 예측 레이블이 아니라 정답 레이블 0의 확률 $0.255$를 사용한다는 점에 주의한다.

### 4-4. 평균 loss와 accuracy

$$
\bar L=\frac{L_1+L_2+L_3}{3}
=\frac{0.408+0.751+1.366}{3}
\approx\boxed{0.842}
$$

세 예시 중 두 개가 정답이므로

$$
\text{accuracy}=\boxed{\frac{2}{3}\approx0.667}
$$

이다.

---

## 문제 5 풀이. Text Generation의 Search와 Sampling

### 5-1. Greedy search

첫 번째 단계에서 확률이 가장 큰 A($0.40$)를 선택하고, A 다음에는 확률이 가장 큰 x($0.60$)를 선택한다. 따라서

$$
\boxed{Ax},\qquad P(Ax)=0.40\times0.60=\boxed{0.240}
$$

이다.

### 5-2. Beam search ($b=2$)

첫 번째 단계에서 확률이 큰 두 가설 **A**($0.40$)와 **B**($0.35$)를 유지한다. 각 가설의 확장은 다음과 같다.

| 시퀀스 | 확률 |
|---|---:|
| $A\texttt{<eos>}$ | $0.40\times0.10=0.040$ |
| $Ax$ | $0.40\times0.60=0.240$ |
| $Ay$ | $0.40\times0.30=0.120$ |
| $B\texttt{<eos>}$ | $0.35\times0.80=0.280$ |
| $Bx$ | $0.35\times0.10=0.035$ |
| $By$ | $0.35\times0.10=0.035$ |

두 번째 단계에서 상위 두 시퀀스는 $B\texttt{<eos>}$($0.280$)와 $Ax$($0.240$)이며, 최종 최적 시퀀스는

$$
\boxed{B\texttt{<eos>}},\qquad P=0.280
$$

이다.

### 5-3. 전체 9개 시퀀스 중 최고 확률 시퀀스

5-2에서 계산하지 않은 C로 시작하는 시퀀스의 확률은

$$
P(C\texttt{<eos>})=0.25\times0.05=0.0125,
$$

$$
P(Cx)=0.25\times0.90=0.225,
$$

$$
P(Cy)=0.25\times0.05=0.0125
$$

이다. 9개 시퀀스 중 가장 큰 확률은 $P(B\texttt{<eos>})=0.280$이므로 최고 확률 시퀀스는

$$
\boxed{B\texttt{<eos>}}
$$

이다. $b=2$인 beam search는 이 시퀀스를 **찾는다**. (반면 greedy search는 첫 단계에서 B를 버리므로 찾지 못한다.)

### 5-4. 최고 확률 시퀀스의 자연로그 확률

$$
\ln P(B\texttt{<eos>})=\ln(0.35)+\ln(0.80)\approx-1.050+(-0.223)=\boxed{-1.273}
$$

### 5-5. Top-$k$ sampling ($k=3$)

확률이 가장 큰 세 토큰 `cat`, `mat`, `table`만 남긴다. 확률의 합은

$$
0.34+0.26+0.19=0.79
$$

이므로 재정규화된 확률은

$$
P(\text{cat})=\frac{0.34}{0.79}\approx0.430,
\qquad
P(\text{mat})=\frac{0.26}{0.79}\approx0.329,
\qquad
P(\text{table})=\frac{0.19}{0.79}\approx0.241
$$

이다. 나머지 `dog`, `<eos>`의 확률은 0이다.

$$
\boxed{[\text{cat}:0.430,\ \text{mat}:0.329,\ \text{table}:0.241]}
$$

### 5-6. No-repeat bigram과 nucleus sampling ($p=0.75$)

이 문항은 5-5의 top-$k$ 결과가 아니라 **5-5에 주어진 원래 확률 분포**

$$
[\text{cat}:0.34,\ \text{mat}:0.26,\ \text{table}:0.19,\ \text{dog}:0.13,\ \texttt{<eos>}:0.08]
$$

에서 출발한다.

**(1) 금지되는 토큰.** 현재 텍스트 `the cat chased the`에 이미 나온 bigram은 `the cat`, `cat chased`, `chased the`이다. 마지막 토큰이 `the`이므로 다음 토큰 $w$는 bigram `the` $w$를 만든다. 이 중 이미 나온 것은 `the cat`뿐이므로 금지되는 토큰은

$$
\boxed{\text{cat}}
$$

이다.

**(2) 재정규화.** `cat`의 확률을 0으로 만들면 남은 확률의 합은

$$
0.26+0.19+0.13+0.08=0.66
$$

이다. 재정규화하면

$$
P(\text{mat})=\frac{0.26}{0.66}\approx0.394,\qquad
P(\text{table})=\frac{0.19}{0.66}\approx0.288,
$$

$$
P(\text{dog})=\frac{0.13}{0.66}\approx0.197,\qquad
P(\texttt{<eos>})=\frac{0.08}{0.66}\approx0.121
$$

이다.

**(3) Nucleus sampling.** 확률이 큰 순서대로 누적하면

| 토큰 | 재정규화 확률 | 누적 확률 |
|---|---:|---:|
| mat | 0.394 | 0.394 |
| table | 0.288 | 0.682 |
| dog | 0.197 | 0.879 |
| `<eos>` | 0.121 | 1.000 |

이다. 누적 확률이 처음으로 $0.75$ 이상이 되는 것은 `dog`까지 포함했을 때이므로 nucleus는

$$
\boxed{\{\text{mat},\ \text{table},\ \text{dog}\}}
$$

이다. (이후 이 세 토큰의 확률을 합 $0.879$로 다시 나누어 sampling한다.)

---

## 문제 6 풀이. Tokenisation 방식 비교

### 6-1. 방식과 tokenisation의 연결

| 방식 | Tokenisation 방식 |
|---|---|
| A | Character tokenisation |
| B | Word tokenisation |
| C | Subword tokenisation (일반) |
| D | WordPiece tokenisation |

WordPiece는 subword tokenisation에 속하는 특정 알고리즘이다. 즉 C는 일반적인 범주이고, D는 C의 구체적인 구현 방식 중 하나이다(WordPiece ⊂ subword tokenisation).

### 6-2. 가장 긴 토큰 시퀀스

**Character tokenisation(방식 A)**이다. 각 문자를 별도의 토큰으로 분리하므로 단어나 subword 단위보다 토큰 개수가 가장 많다. 예를 들어 `unhappiness` 하나만으로도 11개의 토큰이 된다.

### 6-3. Unknown token이 필요할 가능성이 가장 큰 방식

**Word tokenisation(방식 B)**이다. 완전한 단어 전체가 vocabulary 항목으로 존재해야 하므로, 처음 보는 단어(예: `unfamiliar`가 vocabulary에 없는 경우)는 더 작은 단위로 나누어 표현할 수 없어 `[UNK]`로 대체된다. 반면 character와 subword 방식은 알려진 작은 조각들의 조합으로 표현할 수 있다.

### 6-4. `##` 접두사의 의미

`##`는 해당 토큰이 새로운 단어의 시작이 아니라 바로 앞 토큰에 공백 없이 이어지는 continuation piece임을 나타낸다. 따라서 `un`, `##happi`, `##ness`는 하나의 원래 단어 `unhappiness`를 구성한다.

---

## 문제 7 풀이. Sequence Classification과 Token Classification

Sequence classification은 입력 시퀀스 전체에 하나의 레이블을 예측하고, token classification은 각 토큰 위치마다 하나의 레이블을 예측한다.

1. 전체 tweet에 감정 레이블 하나를 할당: **sequence classification**
2. 문장의 각 토큰에 NER 레이블을 할당: **token classification**
3. 전체 email의 spam 여부를 판단: **sequence classification**
4. 뉴스 기사의 각 토큰에 named-entity type을 할당: **token classification**

---

## 문제 8 풀이. Padding과 Attention Mask

### 8-1. `max_len`

두 시퀀스의 길이는 각각 4와 6이므로

```python
max_len = 6
```

이다.

### 8-2. `input_ids`와 `attention_mask`

첫 번째 시퀀스에는 $6-4=2$개의 padding token ID 0을 뒤에 추가하고, 두 번째 시퀀스는 그대로 유지한다.

```python
input_ids = [
    [101, 2023, 2003, 102, 0, 0],
    [101, 19081, 3073, 2146, 10071, 102]
]
```

실제 토큰 위치에는 1, padding 위치(`token_id == 0`)에는 0을 할당한다.

```python
attention_mask = [
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1]
]
```
