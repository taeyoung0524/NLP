# Solutions: Text Generation and Decoding

## Solution 1. Greedy Search

Greedy search selects token **B**, because it has the highest probability.

$$
\ln(0.45)\approx -0.799
$$

**Answer:** B; log probability $\approx -0.799$.

---

## Solution 2. Sequence Probability

The sequence probability is

$$
P(w_1,w_2,w_3)
=
P(w_1)P(w_2\mid w_1)P(w_3\mid w_1,w_2)
$$

Therefore,

$$
P(w_1,w_2,w_3)=0.5\times0.4\times0.2=0.04
$$

The log probability is

$$
\ln(0.04)
=
\ln(0.5)+\ln(0.4)+\ln(0.2)
\approx -3.219
$$

Log probabilities convert products into sums and reduce numerical underflow when many small probabilities are multiplied.

---

## Solution 3. Beam Search

With $b=2$, A and B are retained after the first step.

$$
P(AX)=0.5\times0.2=0.10
$$

$$
P(AY)=0.5\times0.5=0.25
$$

$$
P(AZ)=0.5\times0.3=0.15
$$

$$
P(BX)=0.3\times0.6=0.18
$$

$$
P(BY)=0.3\times0.1=0.03
$$

$$
P(BZ)=0.3\times0.3=0.09
$$

The two highest-probability sequences are **AY** with 0.25 and **BX** with 0.18.

---

## Solution 4. Greedy Search vs. Global Sequence Probability

Greedy search selects A first. Given A, X and Y are tied at 0.5, so the stated tie-breaking rule selects X. The greedy sequence is **AX**.

$$
P(AX)=0.6\times0.5=0.30
$$

$$
P(AY)=0.6\times0.5=0.30
$$

$$
P(BX)=0.4\times0.9=0.36
$$

$$
P(BY)=0.4\times0.1=0.04
$$

The most probable complete sequence is **BX**, with probability 0.36. Greedy search therefore does not find the globally most probable sequence in this example. A locally optimal choice at each step does not necessarily maximise the probability of the complete sequence.

---

## Solution 5. Temperature and Softmax

For $T=1$,

$$
P_1=\frac{e^2}{e^2+e^1},\qquad
P_2=\frac{e^1}{e^2+e^1}
$$

so

$$
P\approx[0.731,\;0.269]
$$

For $T=0.5$, the scaled logits are

$$
[2/0.5,\;1/0.5]=[4,2]
$$

and

$$
P_1=\frac{e^4}{e^4+e^2},\qquad
P_2=\frac{e^2}{e^4+e^2}
$$

so

$$
P\approx[0.881,\;0.119]
$$

Reducing the temperature makes the distribution sharper, concentrating more probability on the token with the larger logit.

---

## Solution 6. Top-k Sampling

For $k=3$, the retained probabilities are

$$
[0.35,\;0.25,\;0.20]
$$

Their sum is

$$
0.35+0.25+0.20=0.80
$$

After renormalisation,

$$
\frac{0.35}{0.80}=0.4375,\qquad
\frac{0.25}{0.80}=0.3125,\qquad
\frac{0.20}{0.80}=0.25
$$

Thus,

$$
[0.4375,\;0.3125,\;0.25]
$$

The other three tokens are excluded from the candidate set and have zero sampling probability for this step.

---

## Solution 7. Nucleus Sampling

The cumulative probabilities are

$$
0.40,\qquad
0.40+0.25=0.65,\qquad
0.40+0.25+0.15=0.80
$$

The nucleus therefore contains the first three tokens.

After renormalisation,

$$
\frac{0.40}{0.80}=0.50,\qquad
\frac{0.25}{0.80}=0.3125,\qquad
\frac{0.15}{0.80}=0.1875
$$

Thus,

$$
[0.50,\;0.3125,\;0.1875]
$$

Top-k sampling retains a fixed number of tokens, whereas nucleus sampling retains a variable number determined by the cumulative probability threshold.
