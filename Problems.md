# 문제: Self-Attention의 계산

다음 문장을 Transformer가 처리한다고 하자.

$$
\text{Time flies like an arrow}
$$

각 단어를 하나의 토큰으로 사용하면

$$
t_1=\text{Time},\quad
t_2=\text{flies},\quad
t_3=\text{like},\quad
t_4=\text{an},\quad
t_5=\text{arrow}
$$

이다.

각 토큰의 임베딩을 $x_1,\ldots,x_5$라 하고, 다음과 같이 주어진다고 하자.

$$
x_1=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\quad
x_2=
\begin{bmatrix}
1\\
1
\end{bmatrix},
\quad
x_3=
\begin{bmatrix}
0\\
1
\end{bmatrix},
\quad
x_4=
\begin{bmatrix}
0\\
-1
\end{bmatrix},
\quad
x_5=
\begin{bmatrix}
2\\
1
\end{bmatrix}.
$$

각 임베딩은 다음 토큰에 대응한다.

| 토큰 번호 | 토큰 | 임베딩 |
|---|---|---|
| $t_1$ | Time | $x_1=[1,0]^T$ |
| $t_2$ | flies | $x_2=[1,1]^T$ |
| $t_3$ | like | $x_3=[0,1]^T$ |
| $t_4$ | an | $x_4=[0,-1]^T$ |
| $t_5$ | arrow | $x_5=[2,1]^T$ |

Self-Attention에서 Query, Key, Value는 각각

$$
q_i=W_Qx_i,
\qquad
k_i=W_Kx_i,
\qquad
v_i=W_Vx_i
$$

로 계산한다.

계산을 단순하게 하기 위해 다음 행렬을 사용한다고 하자.

$$
W_Q=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
W_K=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
W_V=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}.
$$

이 문제에서는 두 번째 토큰인 **flies**의 Self-Attention을 계산한다.

---

## 문제 1. Query, Key, Value 계산

`flies`에 해당하는 Query $q_2$와 각 토큰의 Key

$$
k_1,k_2,k_3,k_4,k_5
$$

및 Value

$$
v_1,v_2,v_3,v_4,v_5
$$

를 계산하시오.

---

## 문제 2. Attention Score 계산

`flies`의 Query $q_2$와 각 토큰의 Key $k_j$ 사이의 scaled dot-product attention score를 계산하시오.

$$
s_{j2}
=
\frac{q_2^T k_j}{\sqrt{d_k}}
$$

단,

$$
d_k=2,
\qquad
\sqrt{2}\approx1.414
$$

로 한다.

---

## 문제 3. Attention Weight 계산

Softmax를 이용하여 attention weight $w_{j2}$를 계산하시오.

$$
w_{j2}
=
\frac{\exp(s_{j2})}
{\displaystyle\sum_{m=1}^{5}\exp(s_{m2})}
$$

계산을 위해 다음 값을 사용할 수 있다.

$$
e^{-0.707}\approx0.493,
\qquad
e^{0.707}\approx2.028,
$$

$$
e^{1.414}\approx4.113,
\qquad
e^{2.121}\approx8.340.
$$

또한 어떤 단어가 `flies`에 가장 큰 attention weight를 갖는지 설명하시오.

---

## 문제 4. 새로운 토큰 표현 계산

Self-Attention 이후 $i$번째 토큰의 새로운 표현을 다음과 같이 정의한다.

$$
\boxed{
x_i'=\sum_{j=1}^{n}w_{ji}x_j
}
$$

`flies`는 두 번째 토큰이므로

$$
\boxed{
x_2'=\sum_{j=1}^{5}w_{j2}x_j
}
$$

이다.

문제 3에서 구한 attention weight를 사용하여 $x_2'$를 계산하시오.

---

## 문제 5. 출력 단어 결정

Self-Attention으로 계산된 $x_2'$가 출력층으로 전달된다고 하자.

출력층에서 후보 단어 `flies`, `soars`, `likes`에 대한 점수를 다음과 같이 계산한다.

$$
z_{\text{flies}}
=
\begin{bmatrix}
0&0.5
\end{bmatrix}
x_2',
$$

$$
z_{\text{soars}}
=
\begin{bmatrix}
1&0
\end{bmatrix}
x_2',
$$

$$
z_{\text{likes}}
=
\begin{bmatrix}
0&1
\end{bmatrix}
x_2'.
$$

각 점수를 계산하고 가장 높은 점수를 갖는 단어를 선택하시오.

이를 이용하여 `flies`가 Self-Attention을 통해 문맥 정보를 반영하고 최종적으로 `soars`로 출력되는 과정을 설명하시오.
