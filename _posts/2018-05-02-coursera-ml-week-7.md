---
title:  Week 7 Support Vector Machines
category: CourseraMachineLearning
excerpt: |
  서포트벡터머신(SVM)은 또다른 supervised 머신러닝 알고리즘이다. 때때로 이 알고리즘은 더 나은 결과를 가져오기도 한다. 로지스틱회기에서 아래의 룰을 사용했던 것을 떠올려 보자. 
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  
  ref: [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning "Coursera ML")
  
  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")
  
feature_image: "../assets/images/coursera_ML/title.png"
image: "./assets/images/coursera_ML/title.png"
---

### Optimization Objective

서포트벡터머신(SVM)은 또다른 supervised 머신러닝 알고리즘이다. 때때로 이 알고리즘은 더 파워풀한 결과를 가져오기도 한다. 

로지스틱회기를 떠올려 아래의 룰을 사용했던 것을 상기하자.  

if y=1, then $h_\theta(x) \approx 1$ and $\Theta^Tx \gg 0$

if y=0, then $h_\theta(x) \approx 0$ and $\Theta^Tx \gg 0$

정규화 항을 제외한 비용함수도 아래와 같이 구했었다. 

$$
\begin{align*}J(\theta) & = \frac{1}{m}\sum_{i=1}^m -y^{(i)} \log(h_\theta(x^{(i)})) - (1 - y^{(i)})\log(1 - h_\theta(x^{(i)}))\\ & = \frac{1}{m}\sum_{i=1}^m -y^{(i)} \log\Big(\dfrac{1}{1 + e^{-\theta^Tx^{(i)}}}\Big) - (1 - y^{(i)})\log\Big(1 - \dfrac{1}{1 + e^{-\theta^Tx^{(i)}}}\Big)\end{align*}
$$

서포트벡터머신을 만들기 위해 비용 함수의 첫번째 항을 다음과 같이 변형하였다. 

$-\log(h_{\theta}(x)) = -\log\Big(\dfrac{1}{1 + e^{-\theta^Tx}}\Big)$ 그리고 여기서 $\Theta^Tx$가 1보다 크면 출력은 0이 된다. z가 1보다 작으면 감소하는 직선을 sigmoid 곡선 대신 사용한다. 이러한 그래프를 hinge loss 함수라 한다. 

![](http://cfile6.uf.tistory.com/image/2450613C586E3FEB074D55 "Optimization Objective" "width:600px;height:200px;float:center;padding-left:10px;")

와 비슷하게 두 번째 항도 다음과 같이 변형 할 수 있다. $-\log(1 - h_{\theta(x)}) = -\log\Big(1 - \dfrac{1}{1 + e^{-\theta^Tx}}\Big)$ 여기서 z가 -1보다 작으면, 결과는 0이 된다. 여기서는 -1보다 큰 부분을 직선으로 나타낸다. 

위의 식들을 각각 $\text{cost}_1(z)$과 $\text{cost}_0(z)$으로 나타내고, 아래와 같이 정의 할 수 있다. 

$z = \theta^Tx$

$\text{cost}_0(z) = \max(0, k(1+z))$

$\text{cost}_1(z) = \max(0, k(1-z))$

로지스틱 회귀에서 정규화 항을 추가한 전체 식을 떠올려 보자. 

$J(\theta) = \frac{1}{m} \sum_{i=1}^m y^{(i)}(-\log(h_\theta(x^{(i)}))) + (1 - y^{(i)})(-\log(1 - h_\theta(x^{(i)}))) + \dfrac{\lambda}{2m}\sum_{j=1}^n \Theta^2_j$

위의 비용 함수와 같이 SVM에서도 $\text{cost}_1(z)$과 $\text{cost}_0(z)$로 나타 낼 수 있다.

$$J(\theta) = \frac{1}{m} \sum_{i=1}^m y^{(i)} \ \text{cost}_1(\theta^Tx^{(i)}) + (1 - y^{(i)}) \ \text{cost}_0(\theta^Tx^{(i)}) + \dfrac{\lambda}{2m}\sum_{j=1}^n \Theta^2_j$$

위의 식에서 m을 곱해 분모항을 제거 할 수 이다. 비용함수에 상수항을 곱하는 것은 최적화에 영향을 주지 않는다.

$$J(\theta) = \sum_{i=1}^m y^{(i)} \ \text{cost}_1(\theta^Tx^{(i)}) + (1 - y^{(i)}) \ \text{cost}_0(\theta^Tx^{(i)}) + \dfrac{\lambda}{2}\sum_{j=1}^n \Theta^2_j$$

표준에 맞춰 정규화 항에 $\lambda$대신 C를 사용해서 나타내자. 

$$J(\theta) = C\sum_{i=1}^m y^{(i)} \ \text{cost}_1(\theta^Tx^{(i)}) + (1 - y^{(i)}) \ \text{cost}_0(\theta^Tx^{(i)}) + \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j$$

위 식은 앞에서 $C = \dfrac{1}{\lambda}$ 곱한 것과 같다. 

![](http://cfile9.uf.tistory.com/image/2257874F5853E410085E4A "SVM" "width:600px;height:200px;float:center;padding-left:10px;")

정규화를 더 세게 주려면(오버피팅을 해결하기 위해) C를 감소시키고, 반대로 정규화를 더 약화 시키려면 C를 증가시킨다. 

끝으로 SVM에서 가설함수는 y가 0인지 1인지에 대한 확률을 추정하지는 않는다. 대신 출력이 0 또는 1로 나온다. 이러한 함수를 판별함수(discriminant function)라 한다. 

$h_\theta(x) =\begin{cases} 1 & \text{if} \ \Theta^Tx \geq 0 \\ 0 & \text{otherwise}\end{cases}$

##### Large Margin Intuition

![](http://cfile3.uf.tistory.com/image/225A99475853E4230D0582 "Large Margin Intuition" "width:600px;height:400px;float:center;padding-left:10px;")

Large Margin Classifiers은 SVM을 유용하게 사용 할 수 있는 방법 중 하나다.
If y=1, we want $\Theta^Tx \geq 1$ (not just $\ge$ 0)
If y=0, we want $\Theta^Tx \leq -1$ (not just $\le$ $0)

위의 식에서 상수 C를 100,000 정도로 매우 큰 값으로 설정하면 최적화 과정에 의해 식 A가 0이 되게끔 Θ가 설정된다. 위에서 Θ를 다음과 같이 설정하였다. 

$\Theta^Tx \geq 1$ If y=1 and $\Theta^Tx \leq -1$ If y=0.

즉 만약 C가 매우 크다면 아래 식이 성립되어야 한다.

$\sum_{i=1}^m y^{(i)}\text{cost}_1(\Theta^Tx) + (1 - y^{(i)})\text{cost}_0(\Theta^Tx) = 0$

따라서 비용함수를 아래와 같이 간략화 할 수 있다. 

$$
\begin{align*} J(\theta) = C \cdot 0 + \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j \newline = \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j \end{align*}
$$

로지스틱 회귀에서 특정군(Negative, Positive)을 나누는 경계면(Decision Boundary)을 떠올려보자. SVM에서는 특정군의 데이터에서 가능한 가장 먼 지점으로 경계면이 결정된다. 가장 가까운 데이터와 경계면 사이의 거리를 마진(Margin)이라한다. SVM은 이 마진을 최대화 하는 방향으로 학습되므로, 이를 Large Margin Classifiers라 부른다. 
SVM은 Negative, Positive 데이터를 가장 큰 마진으로 분리한다. C가 클수록 Margin도 커진다. Negative, Positive 데이터들을 분리하는 경계선이 직선인 경우 데이터가 linearly separable하다. Decision Boundary의 영향을 받질 않아도 되는 예외적 데이터가 있을때 C를 줄일 수 있다. C를 증가/감소 시키는 것은 $\lambda$을 감소/증가 시키는 것과 비슷한 결과를 내며, 경계선을 단순화 할 수 있다.        

##### Kernels I

![](http://cfile4.uf.tistory.com/image/230B4C4A5853E441123AC0 "Kernels" "width:600px;height:200px;float:center;padding-left:10px;")

SVM을 사용해서 Kernel로 복잡하고 비선형적인 분류기를 만들 수 있다. x가 주어질 때 랜드마크 $l^{(1)}, l^{(2)}, l^{(3)}$에 근접하게끔 새로운 피쳐를 계산한다. 이를 위해 랜드마크 $l^{(i)}$와 x사이의 "유사도(Similarity)" 아래와 같이 구한다. 
$f_i = similarity(x, l^{(i)}) = \exp(-\dfrac{||x - l^{(i)}||^2}{2\sigma^2})$
이러한 유사도 함수를 가우시안 커널(Gaussian Kernel)이라 한다. 이것은 여러 커널중 하나다. 이러한 유사도 함수는 아래와 같이 나타 내기도 한다. 
$f_i = similarity(x, l^{(i)}) = \exp(-\dfrac{\sum^n_{j=1}(x_j-l_j^{(i)})^2}{2\sigma^2})$
아래 식은 이러한 유사도 함수의 특징을 보여준다.
if $x \approx l^{(i)}$, then  $f_i = \exp(-\dfrac{\approx 0^2}{2\sigma^2}) \approx 1$
if $x$ is far from $l^{(i)}$, then  $f_i = \exp(-\dfrac{(large\ number)^2}{2\sigma^2}) \approx 0$
즉 x가 랜드마크 근처 있으면 유사도는 1에 가까워지며, 반대로 x가 랜드마크와 멀리 떨어져 있으면 유사도는 0에 가까워 진다. 그리고 각 랜드마크의 합이 가설함수로 주어진다.  
$$
\begin{align*}l^{(1)} \rightarrow f_1 \newline l^{(2)} \rightarrow f_2 \newline l^{(3)} \rightarrow f_3 \newline\dots \newline h_\Theta(x) = \Theta_1f_1 + \Theta_2f_2 + \Theta_3f_3 + \dots\end{align*}
$$

![](http://cfile4.uf.tistory.com/image/256B954B5853E459149D37 "kernel parameter" "width:600px;height:200px;float:center;padding-left:10px;")

커널내 $\sigma^2$ 파라미터는 가우시안 커널이 얼마나 가파른지, 완만한지를 결정한다. 그리고 가중치인 Θ값에 의해 전체적인 경계선의 모양이 결정된다. 

##### Kernels II

![](http://cfile3.uf.tistory.com/image/224FBF4B5853E4652A1550 "Kernels II" "width:600px;height:200px;float:center;padding-left:10px;")

랜드마크를 얻는 한가지 방법은 트레이닝 데이터에서 가져오는 것이다. 이 때 트레이닝 데이터의 수 m개 만큼 랜드마크를 얻을 수 있다. 

$f_1 = similarity(x,l^{(1)})$, $f_2 = similarity(x,l^{(2)})$, $f_3 = similarity(x,l^{(3)})$ ...

이런식으로 하면 각 데이터 $x_{(i)}$마다 피쳐벡터 $f_{(i)}$를 얻을 수 있다. 그리고 여기서 각 $Θ_0$에 대해 $f_0 = 1$이다. 

$x^{(i)} \rightarrow \begin{bmatrix}f_1^{(i)} = similarity(x^{(i)}, l^{(1)}) \newline f_2^{(i)} = similarity(x^{(i)}, l^{(2)}) \newline\vdots \newline f_m^{(i)} = similarity(x^{(i)}, l^{(m)}) \newline\end{bmatrix}$

SVM식에서 x를 f로 치환하여 아래와 같이 최적화 식으로 쓸 수 있다. 

$$\min_{\Theta} C \sum_{i=1}^m y^{(i)}\text{cost}_1(\Theta^Tf^{(i)}) + (1 - y^{(i)})\text{cost}_0(\theta^Tf^{(i)}) + \dfrac{1}{2}\sum_{j=1}^n \Theta^2_j$$

커널을 이용하여 f(i)로 나타내는 것은 SVM 뿐만 하니라 로지스틱 회귀에서도 사용 할 수 있다. 하지만 SVM의 가설함수형식의 알고리즘에서 커널을 사용할 때 계산 속도가 빠르므로, 커널 함수는 주로 SVM에서 사용된다. 

##### Choosing SVM Parameter
C를 선택한다. 위에서 $C = \dfrac{1}{\lambda}$로 나타 냈던것을 떠올려 보자. 

- C가 크면, Higher Variance/Lower Bias 결과를 얻을 수 있으며, 
- C가 작으면, Lower Variance/Higher Bias 한 결과를 얻을 수 있다. 

가우시안 커널에서 설정해야 할 다른 파라미터로는 $\sigma^2$이 있다. $\sigma^2$이 크면, 커널은 완만한 형태를 띄며, Higher Variance/Lower Bias 한 특징을 보이고, 반대로 $\sigma^2$이 작으면, 커널은 가파른 형태를 띄며, Lower Variance/Higher Bias 한 특징을 보인다. 

##### Using An SVM

좋은 SVM 라이브러리는 많으며 또 쉽게 구할 수 있다. 앤드류 응 교수는 주로 'liblinear', 'libsvm'를 사용한다. 실제 이러한 알고리즘을 적용할 때는 직접 함수를 짜기 보다는 라이브러리를 사용하기를 권한다. 실제 사용에서 다음과 같은 선택을 해야한다. 

- 파라미터 C 선택하기
- 커널 종류 선택하기
- No Kernel은 선형분류기(linear classifier)를 의미한다. 
- n이 큰 경우, m이 작은 경우 선택
- 가우시안 커널일 경우 $\sigma^2$ 값을 설정한다. 
- n이 작은 경우, m이 큰 경우 선택

특정 라이브러리는 커널 함수 제공을 요구하는 경우도 있다. 

Note: 가우시안 커널 사용전 피쳐 스케일링 작업을 수행하자. 

Note: 모든 커널이 유사도를 나타내지는 않는다. "Mercer's Theorem"를 보장하는 커널만이 유사도를 나타내는데 사용 할 수 있다. 

C나 다른 파라미터들은 학습을 통하여 값을 정할 수 있다. 

##### Multi-class Classification

![](http://cfile25.uf.tistory.com/image/264B42495853E4721F4F7E "Kernels II" "width:600px;height:200px;float:center;padding-left:10px;")

많은 SVM 라이브러리에서 다중 클래스 분류기를 제공한다. 

$y \in {1,2,3,\dots,K}$고, $\Theta^{(1)}, \Theta^{(2)}, \dots,\Theta{(K)}$와 같을 때 로지스틱 회귀에서 다룬 one-vs-all 방법을 사용한다. $(\Theta^{(i)})^Tx$가 가장 큰 클래스 i를 선택 한다. 

##### Logistic Regression vs. SVMs

피쳐 수 n이 큰 경우(데이터 개수 m에 비해 상대적으로), 로지스틱 회귀나 커널을 적용하지 않은 SVM을 사용한다. n이 작고 m이 보통값을 가지면 가우시안 커널을 적용한 SVM을 사용한다. n이 작고 m이 큰 경우, 수동적으로 많은 피쳐를 생성/추가하고 로지스틱 회귀나 커널을 적용하지 않은 SVM을 사용한다. 첫번째 케이스에는 복잡한 다항식 형태의 가설함수를 생성하는데 필요한 데이터가 충분하지 않다. 두번째 케이스에는 복잡한 비선형 가설함수를 표현하기에 충분한 데이터가 존재한다. 마지막 케이스에서는 로지스틱회귀가 잘 동작하기위해 피쳐수를 늘려야한다. 

Note: 신경망학습은 모든 경우에 대해 잘 동작한다. 하지만 학습에 시간이 많이 걸린다.        

