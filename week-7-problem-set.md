# Problem Set: GPT-2 Self-Attention for the Token “it”

**Instructions:** Show your calculations. Round numerical results to three decimal places. Use a calculator for exponentials.

## Sentence and model setup

Consider the sentence:

> **A robot must obey the orders given it.**

For this tutorial, assume that the sentence is tokenised into the following **nine tokens**, with no subword splitting:

| Position (0-based) | Token |
|:---:|:---|
| 0 | A |
| 1 | robot |
| 2 | must |
| 3 | obey |
| 4 | the |
| 5 | orders |
| 6 | given |
| 7 | it |
| 8 | . |

We will calculate **only the representation of the token `it` at position 7** as it passes through a simplified GPT-2 Transformer block. Because GPT-2 uses *causal* self-attention, `it` may attend to positions **0 through 7**, but not to the period at position 8.

Use a simplified model with embedding dimension $d_{\mathrm{model}}=4$, two attention heads ($h=2$), and head dimension $d_k=d_v=2$. Assume that the input embeddings **already include positional information**. All biases are zero. Ignore residual connections and layer normalisation, and replace GELU with ReLU for hand calculation. The MLP hidden dimension is also reduced from the standard GPT-2 expansion to make arithmetic manageable.

The input embedding matrix $X\in\mathbb{R}^{9\times4}$ is:

$$
X=\begin{bmatrix}
0&0&0&0\\
1&0&0&0\\
0&0&0&0\\
0&0&0&0\\
0&0&0&0\\
0&1&0&0\\
0&0&0&0\\
1&0&1&0\\
0&0&0&0
\end{bmatrix}.
$$

The rows correspond to the tokens in the table above. **These are artificial embeddings chosen to make hand calculation possible; they are not embeddings extracted from a trained GPT-2 model.**

## Problem 1. Create Query, Key and Value

For this example, let

$$
W_Q=W_K=W_V=I_4,
$$

where $I_4$ is the $4\times4$ identity matrix. Compute

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

**(a)** Write the query vector $q_{\mathrm{it}}$ for `it` and the key and value vectors for `robot`, `orders`, and `it`.

**(b)** Give the shape of each of the full matrices $Q$, $K$, and $V$.

**(c)** GPT-2 can compute these projections in one operation using a combined weight matrix. Write $W_{\mathrm{c\_attn}}=[W_Q\;W_K\;W_V]$ and state its shape.

## Problem 2. Split into Attention Heads

Split the **four features** of every query, key, and value into two heads: head 1 uses features 1–2, and head 2 uses features 3–4 (feature numbers here are **1-based**).

**(a)** Write $q_{\mathrm{it}}^{(1)}$ and $q_{\mathrm{it}}^{(2)}$.

**(b)** Write $k_j^{(1)},v_j^{(1)},k_j^{(2)},v_j^{(2)}$ for $j\in\{1,5,7\}$ (the tokens `robot`, `orders`, and `it`).

## Problem 3. Attention Scores and Causal Mask

For each head $r\in\{1,2\}$, calculate the attention scores **from `it` to every allowed token**, using

$$
s_j^{(r)}=\frac{q_{\mathrm{it}}^{(r)}\cdot k_j^{(r)}}{\sqrt{2}},\qquad j=0,1,\ldots,7.
$$

**(a)** Calculate the eight scores for each head. Give the two vectors in token order, from `A` to `it`.

**(b)** What score is assigned to the future token `.` (position 8) by the causal mask, and why?

**(c)** Calculate the attention weights for both heads using softmax **over the eight allowed positions only**:

$$
a_j^{(r)}=\frac{\exp(s_j^{(r)})}{\sum_{m=0}^{7}\exp(s_m^{(r)})}.
$$

Show the denominator for each head and give each weight to three decimal places. The weights within each head must sum to approximately 1.

## Problem 4. Weighted Sum of Values

For each head, compute the output representation for `it`:

$$
h_{\mathrm{it}}^{(r)}=\sum_{j=0}^{7}a_j^{(r)}v_j^{(r)}.
$$

**(a)** Calculate $h_{\mathrm{it}}^{(1)}$.

**(b)** Calculate $h_{\mathrm{it}}^{(2)}$.

**(c)** In head 1, compare the attention weight assigned to `robot` with that assigned to `orders`. Which is larger? **Do not interpret this artificial result as evidence of how a trained GPT-2 resolves pronouns.**

## Problem 5. Merge Heads and Apply Output Projection

Concatenate the two head outputs in order:

$$
h_{\mathrm{it}}=\operatorname{Concat}\!\left(h_{\mathrm{it}}^{(1)},h_{\mathrm{it}}^{(2)}\right).
$$

Apply the output projection (a separate weight matrix from the QKV projection):

$$
z_{\mathrm{it}}=h_{\mathrm{it}}W_O,\qquad
W_O=\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&1\\
0&0&0&1
\end{bmatrix}.
$$

**(a)** Calculate the merged vector $h_{\mathrm{it}}$.

**(b)** Calculate $z_{\mathrm{it}}$ and state its dimension.

**(c)** Is this output-projection matrix the same matrix used to create Q, K, and V?

## Problem 6. Position-wise Fully Connected Neural Network

Pass the projected representation of `it` through a simplified two-layer feed-forward network:

$$
f_{\mathrm{it}}=\operatorname{ReLU}(z_{\mathrm{it}}W_1),
\qquad y_{\mathrm{it}}=f_{\mathrm{it}}W_2,
$$

with

$$
W_1=\begin{bmatrix}
2&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&1
\end{bmatrix},\qquad
W_2=\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&1\\
0&0&0&1
\end{bmatrix}.
$$

**(a)** Calculate $z_{\mathrm{it}}W_1$.

**(b)** Apply ReLU to obtain $f_{\mathrm{it}}$.

**(c)** Calculate the final output $y_{\mathrm{it}}$.

**(d)** State whether the feed-forward network mixes information **across token positions** or processes the representation of each token **independently**.

---

**Scope note:** This exercise calculates the representation of the *existing input token* `it`; it does **not** predict `it` as the next token. In an autoregressive GPT-2 model, the logits used to *predict* `it` would be obtained from the preceding position (`given`).
