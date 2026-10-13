## 문제 1. Query, Key, Value 계산

`flies`의 임베딩은

$$
x_2=
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

이다.

$W_Q$가 단위행렬이므로

$$
q_2=W_Qx_2
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\begin{bmatrix}
1\\
1
\end{bmatrix}
=
\boxed{
\begin{bmatrix}
1\\
1
\end{bmatrix}
}.
$$

$W_K$도 단위행렬이므로

$$
k_i=W_Kx_i=x_i
$$

이다.

따라서

$$
k_1=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\quad
k_2=
\begin{bmatrix}
1\\
1
\end{bmatrix},
\quad
k_3=
\begin{bmatrix}
0\\
1
\end{bmatrix},
$$

$$
k_4=
\begin{bmatrix}
0\\
-1
\end{bmatrix},
\quad
k_5=
\begin{bmatrix}
2\\
1
\end{bmatrix}.
$$

마찬가지로 $W_V$도 단위행렬이므로

$$
v_i=W_Vx_i=x_i
$$

이다.

따라서

$$
v_1=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\quad
v_2=
\begin{bmatrix}
1\\
1
\end{bmatrix},
\quad
v_3=
\begin{bmatrix}
0\\
1
\end{bmatrix},
$$

$$
v_4=
\begin{bmatrix}
0\\
-1
\end{bmatrix},
\quad
v_5=
\begin{bmatrix}
2\\
1
\end{bmatrix}.
$$

이 예제에서는 $W_Q$, $W_K$, $W_V$를 단위행렬로 설정하였기 때문에 Q, K, V의 개념을 구분하면서 계산을 단순화하였다.

---

## 문제 2. Attention Score 계산

`flies`의 Query는

$$
q_2=
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

이다.

### Time

$$
q_2^Tk_1
=
\begin{bmatrix}
1&1
\end{bmatrix}
\begin{bmatrix}
1\\
0
\end{bmatrix}
=1
$$

따라서

$$
s_{12}
=
\frac{1}{\sqrt{2}}
\approx
0.707.
$$

### flies

$$
q_2^Tk_2
=
\begin{bmatrix}
1&1
\end{bmatrix}
\begin{bmatrix}
1\\
1
\end{bmatrix}
=2
$$

따라서

$$
s_{22}
=
\frac{2}{\sqrt{2}}
\approx
1.414.
$$

### like

$$
q_2^Tk_3
=
\begin{bmatrix}
1&1
\end{bmatrix}
\begin{bmatrix}
0\\
1
\end{bmatrix}
=1
$$

따라서

$$
s_{32}
=
\frac{1}{\sqrt{2}}
\approx
0.707.
$$

### an

$$
q_2^Tk_4
=
\begin{bmatrix}
1&1
\end{bmatrix}
\begin{bmatrix}
0\\
-1
\end{bmatrix}
=-1
$$

따라서

$$
s_{42}
=
\frac{-1}{\sqrt{2}}
\approx
-0.707.
$$

### arrow

$$
q_2^Tk_5
=
\begin{bmatrix}
1&1
\end{bmatrix}
\begin{bmatrix}
2\\
1
\end{bmatrix}
=3
$$

따라서

$$
s_{52}
=
\frac{3}{\sqrt{2}}
\approx
2.121.
$$

따라서 전체 attention score는

$$
\boxed{
(s_{12},s_{22},s_{32},s_{42},s_{52})
=
(0.707,1.414,0.707,-0.707,2.121)
}
$$

이다.

가장 큰 attention score는 `arrow`에 해당하는

$$
\boxed{s_{52}=2.121}
$$

이다.

---

## 문제 3. Attention Weight 계산

각 score에 exponential을 취하면

$$
e^{s_{12}}
=
e^{0.707}
\approx2.028,
$$

$$
e^{s_{22}}
=
e^{1.414}
\approx4.113,
$$

$$
e^{s_{32}}
=
e^{0.707}
\approx2.028,
$$

$$
e^{s_{42}}
=
e^{-0.707}
\approx0.493,
$$

$$
e^{s_{52}}
=
e^{2.121}
\approx8.340.
$$

따라서 분모는

$$
2.028+4.113+2.028+0.493+8.340
=
17.002
$$

이다.

각 attention weight는 다음과 같다.

$$
w_{12}
=
\frac{2.028}{17.002}
\approx0.119,
$$

$$
w_{22}
=
\frac{4.113}{17.002}
\approx0.242,
$$

$$
w_{32}
=
\frac{2.028}{17.002}
\approx0.119,
$$

$$
w_{42}
=
\frac{0.493}{17.002}
\approx0.029,
$$

$$
w_{52}
=
\frac{8.340}{17.002}
\approx0.491.
$$

따라서

$$
\boxed{
(w_{12},w_{22},w_{32},w_{42},w_{52})
=
(0.119,0.242,0.119,0.029,0.491)
}
$$

이다.

합을 확인하면

$$
0.119+0.242+0.119+0.029+0.491
=
1.000.
$$

가장 큰 attention weight는

$$
\boxed{w_{52}=0.491}
$$

이다.

따라서 `flies`의 Query는 `arrow`의 Key와 가장 높은 유사도를 가지며, `flies`가 새로운 문맥 표현을 만들 때 `arrow`의 정보를 가장 강하게 참고한다.

---

## 문제 4. 새로운 토큰 표현 계산

Self-Attention 이후 `flies`의 새로운 표현은

$$
x_2'
=
\sum_{j=1}^{5}w_{j2}x_j
$$

이다.

따라서

$$
x_2'
=
0.119
\begin{bmatrix}
1\\
0
\end{bmatrix}
+
0.242
\begin{bmatrix}
1\\
1
\end{bmatrix}
+
0.119
\begin{bmatrix}
0\\
1
\end{bmatrix}
+
0.029
\begin{bmatrix}
0\\
-1
\end{bmatrix}
+
0.491
\begin{bmatrix}
2\\
1
\end{bmatrix}.
$$

첫 번째 성분은

$$
0.119+0.242+0+0+2(0.491)
=
1.343
$$

이다.

두 번째 성분은

$$
0+0.242+0.119-0.029+0.491
=
0.823
$$

이다.

따라서

$$
\boxed{
x_2'
=
\begin{bmatrix}
1.343\\
0.823
\end{bmatrix}
}
$$

이다.

원래 `flies`의 embedding은

$$
x_2=
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

이었지만 Self-Attention 이후에는

$$
\boxed{
\begin{bmatrix}
1\\
1
\end{bmatrix}
\rightarrow
\begin{bmatrix}
1.343\\
0.823
\end{bmatrix}
}
$$

로 바뀌었다.

원래 $x_2$는 `flies`라는 토큰 자체의 표현이다. 반면 새로운 표현은

$$
x_2'
=
w_{12}x_1
+
w_{22}x_2
+
w_{32}x_3
+
w_{42}x_4
+
w_{52}x_5
$$

이므로 문장의 다른 토큰 정보를 모두 포함한다.

특히

$$
w_{52}=0.491
$$

이므로 `arrow`의 정보가 $x_2'$에 가장 크게 반영되었다.

엄밀한 Transformer의 Self-Attention에서는

$$
x_i'
=
\sum_j w_{ji}v_j
$$

로 표현한다.

그런데 이 문제에서는

$$
W_V=I
$$

이므로

$$
v_j=W_Vx_j=x_j
$$

가 된다.

따라서

$$
\boxed{
x_i'
=
\sum_{j=1}^{n}w_{ji}x_j
}
$$

로 직접 표현할 수 있다.

---

## 문제 5. 출력 단어 결정

계산된 `flies`의 contextual representation은

$$
x_2'
=
\begin{bmatrix}
1.343\\
0.823
\end{bmatrix}
$$

이다.

### flies

$$
z_{\text{flies}}
=
\begin{bmatrix}
0&0.5
\end{bmatrix}
\begin{bmatrix}
1.343\\
0.823
\end{bmatrix}
$$

이므로

$$
z_{\text{flies}}
=
0.5(0.823)
=
\boxed{0.412}.
$$

### soars

$$
z_{\text{soars}}
=
\begin{bmatrix}
1&0
\end{bmatrix}
\begin{bmatrix}
1.343\\
0.823
\end{bmatrix}
$$

이므로

$$
\boxed{
z_{\text{soars}}
=
1.343
}.
$$

### likes

$$
z_{\text{likes}}
=
\begin{bmatrix}
0&1
\end{bmatrix}
\begin{bmatrix}
1.343\\
0.823
\end{bmatrix}
$$

이므로

$$
z_{\text{likes}}
=
\boxed{0.823}.
$$

따라서

$$
z_{\text{soars}}
>
z_{\text{likes}}
>
z_{\text{flies}}
$$

이므로 가장 높은 점수를 갖는 단어는

$$
\boxed{\text{soars}}
$$

이다.