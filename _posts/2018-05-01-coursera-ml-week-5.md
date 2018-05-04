---
title: Week 5 Neural Networks - Learning
category: CourseraMachineLearning
excerpt: |
  먼저 밑에서 사용 할 몇몇 변수에 대해 정의하자.a) L = network 내 모든 레이어의 수 b) $s_l$ = 레이어에 있는 유닛의 수(바이어스 유닛 제외) c) K = 출력 유닛의 수
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리

  ref: [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning "Coursera ML")

  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")

feature_image: "https://i.imgur.com/jds8Wur.png"
image: "https://i.imgur.com/jds8Wur.png"
---

### ML: Neural Networks: Learning

##### Cost Function

먼저 밑에서 사용 할 몇몇 변수에 대해 정의하자.

a) L = network 내 모든 레이어의 수

b) $s_l$ = 레이어에 있는 유닛의 수(바이어스 유닛 제외)

c) K = 출력 유닛의 수

![](http://cfile10.uf.tistory.com/image/2578D74B5841495310F874 "Cost Function" "width:600px;height:250px;float:center;padding-left:10px;")

지난 강의에서 신경망 학습 시 여러 출력 노드를 가지는 것에 대해 떠올려 보자. $k^{th}$번째 출력의 가설 함수를 다음과 같이 나타 낼 수 있다. $h_\Theta(x)_k$
신경망 학습에서의 비용 함수는 로지스틱 회기를 사용하여 일반화 할 수 있다.
지난 강의에서 다룬 로지스틱 회기를 떠올려 보자.
$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$$

 신경망 학습에서는 조금 더 복잡해진다.
$$
\begin{gather*}\large J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}
$$

다중 출력 노드를 설명하기위해 몇가지 합연산을 추가하였다. 첫번째 항에서, 출력 노드의 수 K 만큼 반복해서 합연산을 한다. 두번째 정규화 항에서는 여러 개의 theta 행렬을 고려해야한다. 여기서 i,j는 행렬을 구성하는 노드의 인덱스를 뜻하며 L은 전체 레이어의 수와 같다.  

##### Backpropagation Algorithm
"Backpropagation(역전파)"은 비용함수의 최소 값을 찾는 기술이다.  이는 선형회귀, 로지스틱 회귀에서 Gradient descent를 사용한 것과 비슷하다.
따라서 목적은 다음과 같다.
$$\min_\Theta J(\Theta)$$

위 식은 적절한 $\Theta$ 값을 사용하여 비용 함수 J를 최소화 하는 것을 표현한다.  이번 장에서는 비용함수 $J(\Theta)$의 편미분 방정식을 살펴본다.
$$\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)$$

Back propagation에서는 모든 노드에 대해 다음과 같은 계산을 수행한다. $\delta_j^{(l)}$=레이어 l의 노드 j의 "error"
지난 장에서 $a_j^{(l)}$가 레이어 l의 노드 j의 활성화 노드였다는 것을 떠올려 보자.
마지막 레이어에서 오차(delta)의 벡터를 계산 할 수 있다.
$$\delta^{(L)} = a^{(L)} - y$$

여기서 L은 레이어의 총 수이며 따라서 $a^{(L)}$은 마지막 레이어의 활성화 유닛의 출력과 같다. 오차는 위 식과 같이 간단히 마지막 레이어의 출력값 $a^{(L)}$과 학습에 사용되는 결과 값 y의 차로 표현 된다.
마지막 레이어 이전 레이어들의  델타값은 아래와 같이 오른 쪽에서 왼쪽으로 즉 역방향으로 이동하며 구할 수 있다.

$$\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ g'(z^{(l)})$$

위 식과 같이 delta 값은 다음 레이어(l+1)의 delta값과 현제 레이어(l)의 Theta값의 곱을 사용하여 계산 할 수 있다. 여기에 활성화 함수(ex. sigmoid)의 편미분(g')과 element-wise 곱(행렬의 각 요소마다 곱함)을 해서 최종 delta 벡터를 계산한다.
sigmoid 함수의 경우 편미분은 아래와 같다.

$$g'(z^{(l)}) = a^{(l)}\ .*\ (1 - a^{(l)})$$
다음과 같이 증명 할 수 있다.

$$g(z) = \frac{1}{1 + e^{-z}}$$
$$\frac{\partial g(z)}{\partial z} = -\left( \frac{1}{1 + e^{-z}} \right)^2\frac{\partial{}}{\partial{z}} \left(1 + e^{-z} \right)$$

따라서 위 Backpropagation식은 아래와 같이 다시 쓸 수 있다.

$$\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .*\ (1 - a^{(l)})$$

학습시 각 data t에 대한 비용함수의 편미분 값은 아래와 같이 나타낼 수 있다.

$$\dfrac{\partial J(\Theta)}{\partial \Theta_{i,j}^{(l)}} = \frac{1}{m}\sum_{t=1}^m a_j^{(t)(l)} {\delta}_i^{(t)(l+1)}$$

위의 식에서 정규화 항은 무시하였다.

![](http://cfile4.uf.tistory.com/image/2370BA4E58414AAE25BD87 "Cost Function" "width:600px;height:350px;float:center;padding-left:10px;")

##### Back propagation algorithm
학습 데이터 $\lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace$가 주어 졌을 때, 모든 (l,i,j)에 대해 $\Delta^{(l)}_{i,j}=0$으로 설정한다.

한 개 data t=1에 대해서

- $a^{(1)} := x^{(t)}$으로 설정
- 모든 $a^{(l)}$ (l=2,3,...,L)을 구하기 위해 Forward Propagation 수행
- $y^{(t)}$를 이용하여 $\delta^{(L)} = a^{(L)} - y^{(t)}$를 계산
- $\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)}) a^{(l)} (1 - a^{(l)})$ 식을 통해 나머지  $\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)}$ 를 계산
- $$\Delta^{(l)}_{i,j} := \Delta^{(l)}_{i,j} + a_j^{(l)} \delta_i^{(l+1)}$$ 를 계산
  벡터화 하면 다음과 같다. $$\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$$

- $D^{(l)}_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right)$ if j≠0

- $D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j}$ if j=0

$\Delta$는 값을 합산하여 편미분을 구하는 accumulator로 사용된다.
실제 증명은 복잡하고 어렵지만 $D^{(l)}_{i,j}$는 다음과 같이 편미분을 뜻한다.

$D_{i,j}^{(l)} = \dfrac{\partial J(\Theta)}{\partial \Theta_{i,j}^{(l)}}$.

##### Backpropagation Intuition
비용 함수는 다음과 같다.

$$
\begin{gather*}J(\theta) = - \frac{1}{m} \sum_{t=1}^m\sum_{k=1}^K \left[ y^{(t)}_k \ \log (h_\theta (x^{(t)}))_k + (1 - y^{(t)}_k)\ \log (1 - h_\theta(x^{(t)})_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_l+1} ( \theta_{j,i}^{(l)})^2\end{gather*}
$$

Class가 1개 밖에 없고, 정규화 항을 고려하지 않는다면 각 데이터의 비용(cost)는 아래와 같이 재정의 할 수 있다.  

$cost(t) =y^{(t)} \ \log (h_\theta (x^{(t)})) + (1 - y^{(t)})\ \log (1 - h_\theta(x^{(t)}))$

위 식을 좀 더 직관적으로 보면 아래와 같다.

$cost(t) \approx (h_\theta(x^{(t)})-y^{(t)})^2$

직관적으로 $\delta_j^{(l)}$는 $a^{(l)}_j$(레이어 l의 유닛 j)에서의 오차다.
따라서 delta 값은 비용함수의 미분 값이라 볼 수 있다.

$\delta_j^{(l)} = \dfrac{\partial}{\partial z_j^{(l)}} cost(t)$

미분은 비용함수의 접선의 기울기이므로, 오차가 클수록 기울기가 커짐을 생각하면 이해하기 쉽다.

##### Implementation Note: Unrolling Parameters
신경망 학습에서는 아래와 같이 행렬의 집합을 다룬다.
$$
\begin{align*} \Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}, \dots \newline D^{(1)}, D^{(2)}, D^{(3)}, \dots \end{align*}
$$

"fminunc()"와 같은 최적와 함수를 사용하려면 아래와 같이 하나의 긴 벡터에 이러한 집합을 넣는 "unroll" 작업이 필요하다.
``` python
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
```
Theta1이 10x11, Theta2가 10x11, Theta3가 1x11의 크기를 가진다고 가정하면, 다음과 같이 unroll 된다.

``` python
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

##### Gradient Checking
Gradient Checking을 통해 backpropagation이 제대로 동작하는지 확인 할 수 있다.
비용함수의 미분 값은 아래와 같이 표현된다.

$\dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon}$

여러 Theta를 다루는 경우에는 아래처럼 확장하여 생각 할 수 있다.

$\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}$

적절히 작은 값을 가지는 epsilon 값을 선택 함으로써, 위의 식을 참으로 만들 수 있다. 값이 너무 작아지면 연산 과정에서 오류가 발생하므로 주의하자. Androw 교수가 추천하는 값은 ${\epsilon = 10^{-4}}$이다

Octave프로그램에서는 다음과 같은 방법을 통해 위 식을 구현 하였다.

``` python
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

그 다음 gradApprox와 deltaVector를 비교한다.

backpropagation 제대로 수행 되면 gradApprox와 deltaVector가 비슷한 값을 가진다. gradApprox 코드는 느리기 때문에 처음 한번만 계산하자.
![](http://cfile27.uf.tistory.com/image/21282337586CF0D12DDAC3 "Gradient Checking" "width:600px;height:200px;float:center;padding-left:10px;")


##### Random Initialization
신경망 학습에서 theta값을 모두 0으로 초기화 하면 제대로 동작하지 않는다. Backpropagation 과정에서 모든 노드가 같은 값으로 반복되어 업데이트 되기 때문이다. 따라서 $\Theta^{(l)}_{ij}$를 $[-\epsilon,\epsilon]$사이의 무작위 값으로 초기화 한다.

$\epsilon = \dfrac{\sqrt{6}}{\sqrt{\mathrm{Loutput} + \mathrm{Linput}}}$

$\Theta^{(l)} = 2 \epsilon \; \mathrm{rand}(\mathrm{Loutput}, \mathrm{Linput} + 1) - \epsilon$

여기서 Theta1이 10x11, Theta2가 10x11 Theta3가 1x11일때 초기화 코드는 아래와 같다.
``` python
Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```
rand(x,y)는 0~1 사이의 실수값으로 행렬을 초기화 하는 함수이다. (참조: 여기의 Epsilon 값은 Gradient Checking과 무관하다.)

왜 이 방법을 사용하는가에 대한 설명은 다음 논문을 참조하자.(https://web.stanford.edu/class/ee373b/nninitialization.pdf)

### Putting in Together
먼저 히든 레이어의 수, 레이어당 유닛 수 등 레이어의 아키텍쳐를 정한다.
- 입력 유닛의 수 = $x^{(i)}$의 크기
- 출력 유닛의 수 = class의 수
- 레이어당 히든 유닛의 수 = 보통 많을 수록 정확함(하지만 많을 수록 연산량이 늘어나므로 정확도와 속도 사이에서 절절한 값을 찾아야 한다.)
- 디폴트: 1개의 히든 레이어, 히든 레이어를 늘릴 때마다, 같은 수의 유닛을 추가한다.

##### Training a Neural Network
1. Theta(가중치)를 무작위 값으로 초기화 한다.
2. $h_\theta(x^{(i)})$를 구하기 위해 Forward Propagation을 수행한다.
3. Cost Function을 구현한다.
4. 미분 값을 계산하기 위해 Back Propagation을 수행한다.
5. Gradient Checking을 이용하여 위의 Back Propagation이 제대로 되었는지 확인한다. 그리고 Gradient Checking을 비활성화 한다.
6. Gradient descent 나 built-in 최적화 함수를 이용하여 cost function을 최소화하는 theta를 구한다.

Forward/Backward Propagation을 수행 할 때 마다. 전체 학습 데이터를 반복문에 넣는다.  
``` python
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```
