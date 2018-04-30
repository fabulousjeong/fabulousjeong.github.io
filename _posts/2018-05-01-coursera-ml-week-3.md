---
title: Week 2 Linear Regression with Multiple Variables
category: CourseraMachineLearning
excerpt: |
  여러 변수에 대한 선형 회귀는 다변량 선형 회귀(multivariate linear regression)로 알려져 있다. 아래에 여러 입력 변수를 가지는 식에 대한 표기법이 있다. 
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  
  ref: [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning "Coursera ML")
  
  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")
  
feature_image: "../assets/images/coursera_ML/title.png"
image: "./assets/images/coursera_ML/title.png"
---

### ML: Logistic Regression

이제 회귀(Regression)문제에서 분류(Classification)문제로 바꿔 생각해보자. 로지스틱회귀(Logistic Regression)이라는 단어에 대해 헷갈릴수 있는데 이는 관용적으로 붙여진 이름이며, 실제로는 회귀(Regression)가 아니라 분류(Classification) 문제를 다룬다.


##### Binary Classification

바이너리 분류에서 결과백터 y를 연속적인 값 대신, {0, 1}로 이루어져 있다.

$$y∈{0,1}$$

여기서 0은 보통 "Negative class"며, 1은 "Positive class"다. 하지만 이는 문제에 맞게 조정 할 수 있다. 

이때 2가지 클래스만 다루므로 "binary classification problem"이라 부른다. 

이를 구현하는 한가지 방법은 선형 회귀를 이용하여 그 결과 값이 0.5 이상인 경우 1에 맵핑하고, 0.5 이하인 경우 0에 맵핑을 하는 것이다. 하지만 이러한 방법은 분류 문제가 선형 함수가 아니기 때문에 실제 사용시에 잘 동작하지 않는다. 

가설함수를 생각해 보자. 가설 함수는 아래와 같은 범위 내에 있다. 

$0 \leq h_\theta (x) \leq 1$

아래에 있는 식은 시그모이드 함수("Sigmoid Function")이라고 하며, 또한 "Logistic Function"이라고도 불린다.     

$$ h_\theta (x) = g ( \theta^T x ) $$
$$ z = \theta^T x $$
$$ g(z) = \dfrac{1}{1 + e^{-z}}$$

![](http://cfile25.uf.tistory.com/image/27232850586BA4DC2F86E1 "Sigmoid Function" "width:600px;height:100px;float:center;padding-left:10px;")

위 함수 g(z)는 모든 실수를 (0,1)의 범위안에 맵핑 시킬 수 있으므로, 특정 함수를 Classification으로 바꾸는데 적합해 보인다. 아래에 링크에서 Sigmoid 함수에 대해 더 알아 볼 수 있다. (https://www.desmos.com/calculator/bgontvxotm)

이전 강의에서 소개한 가설 함수에서 시작해 보자. 0과 1사이에 범위를 제한 하는것은 위의 Logistic Function에 가설함수를 연립 하는 것으로 이루어 진다. 

 $h_{\theta}$ 는 결과가 1이 되는 확률을 보여준다. 가령 $h_{\theta(x)}=0.7$인 경우 결과가 1일 확률은 70%라고 볼 수 있다. 

$$ h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) $$
$$ P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1$$

결과가 0일 확률은 결과가 1일 확률의 보수(Complement)와 같다. 예를 들어 1일 확률이 70%라면 0일 확률은 30%다.


##### Decision Boundary
0 또는 1의 이산 값을 얻기 위해 다음과 같이 가설함수의 결과 값을 정할수 있다. 

$$ h_\theta(x) \geq 0.5 \rightarrow y = 1 $$
$$ h_\theta(x) < 0.5 \rightarrow y = 0 $$

위에서 본 logistic function g는 아래와 같이 입력값이 0보다 크거나 같은 경우 출력값이 0.5보다 크거나 같음을 알 수 있다. 

$$ g(z) \geq 0.5 $$
$$ when \; z \geq 0$$

앞 절에서 다룬 내용을 떠올려 보자. 

$$z=0, e^{0}=1 \Rightarrow g(z)=1/2$$
$$z \to \infty, e^{-\infty} \to 0 \Rightarrow g(z)=1 $$
$$ z \to -\infty, e^{\infty}\to \infty \Rightarrow g(z)=0 $$

따라서 함수 g의 입력 값이 $\theta^T X$라면, 아래와 같이 다시 쓸 수 있다. 

$$h_\theta(x) = g(\theta^T x) \geq 0.5 $$
$$when \; \theta^T x \geq 0$$

위의 식으로 부터 아래와 같이 정리 할 수 있다. 

$$\theta^T x \geq 0 \Rightarrow y = 1 $$
$$\theta^T x < 0 \Rightarrow y = 0 $$

Decision Boundary는 y=0인 구역과 y=1인 구역을 나누는 선으로 표현되며, 가설 함수로부터 만들 수 있다.  

예:

$$ \theta = \begin{bmatrix}5 \newline -1 \newline 0\end{bmatrix} $$
$$ y = 1 \; if \; 5 + (-1) x_1 + 0 x_2 \geq 0 $$
$$ 5 - x_1 \geq 0 $$
$$ x_1 \geq -5 $$
$$ x_1 \leq 5 $$

위의 경우 Decision Boundary는 $x_1=5$로 표현 되는 수직선이며, 여기서 입력이 왼쪽에 있는 경우 $y=1$이고 오른쪽에 있는 경우 $y=0$이다. 

다시 한번 언급하지만, sigmoid function g(z)의 입력은 선형일 필요가 없으며 다음과 같이 원으로 표현 되는 경우도 있다. 

$$z = \theta_0 + \theta_1 x_1^2 +\theta_2 x_2^2$$ 

##### Cost function

선형회귀 때와 같은 cost function을 사용하는 경우 로지스틱 함수에서는 결과값이 파도처럼 요동치는 형태로 나오며, 많은 국소값(Local Optima)들이 나온다. 즉 convex 함수로 표현 되지 않는 문제가 발생한다. 대신 아래와 같이 cost function을 세울 수 있다.  

$$ J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) $$
$$ \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \;  \text{if y = 1} $$
$$ \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \;  \text{if y = 0}$$

![](http://cfile6.uf.tistory.com/image/2675CE4F586BA57D315865 "cost function" "width:600px;height:400px;float:center;padding-left:10px;")
![](http://cfile10.uf.tistory.com/image/2473E44F586BA57D327E9C "cost function" "width:600px;height:400px;float:center;padding-left:10px;")

 위의 그래프를 보면 가설함수의 출력 값이 설정된 y값에서 멀어 질수록 값이 커진다. 가설함수의 출력 값이 y와  같다면 cost는 0이 된다.  

$$ \mathrm{Cost}(h_\theta(x),y) = 0 \text{ if } h_\theta(x) = y $$
$$ \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 $$
$$\mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 $$

y의 참 값이 0인 경우, 가설함수의 출력이 0일때 cost function은 0이 된다. 가설함수의 출력이 1이면 cost function은 무한대 값을 가지게 된다. 반면 y의 참 값이 1인 경우, 가설함수의 출력이 1일때 cost function은 0이 된다. 가설함수의 출력이 0이면 cost function은 무한대 값을 가지게 된다. 위의 방법으로 세운 cost function J(θ)는 Logistic regression에서 convex를 보장한다.   

##### Simplified Cost Function and Gradient Descent
위의 두 조건부 함수를 한 함수로 나타낼 수 있다. 

$\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$

y가 1일때 두번째 항은 0이 되므로 적용 되지 않고, y가 0인 경우 첫번째 항이 0이 되므로 식에 영향을 주지 않는다. 
전체 Cost function은 아래와 같이 쓸 수 있다. 

$J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]$

벡터화 하면 아래와 같다. 

$$ h = g(X\theta)$$
$$ J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right)$$

##### Gradient Descent
Gradient Descent는 아래와 같은 식으로 표현되는 것을 떠올려보자. 

$\begin{align*}& Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \newline & \rbrace\end{align*}$

앞의 식을 미분하여 아래와 같은 식을 구할 수 있다. 

$\begin{align*} & Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}$

위 알고리즘은 앞에서 다룬 선형 회귀와 동일하다. 이 경우 역시 동시에 모든 theta값을 업데이트 해야한다.   

백터화 하면 아래와 같다. 

$\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})$

Partial derivative of $J(\theta)$

먼저 sigmoid function의 미분을 구해보자

$\begin{align*}\sigma(x)'&=\left(\frac{1}{1+e^{-x}}\right)'=\frac{-(1+e^{-x})'}{(1+e^{-x})^2}=\frac{-1'-(e^{-x})'}{(1+e^{-x})^2}=\frac{0-(-x)'(e^{-x})}{(1+e^{-x})^2}=\frac{-(-1)(e^{-x})}{(1+e^{-x})^2}=\frac{e^{-x}}{(1+e^{-x})^2} \newline &=\left(\frac{1}{1+e^{-x}}\right)\left(\frac{e^{-x}}{1+e^{-x}}\right)=\sigma(x)\left(\frac{+1-1 + e^{-x}}{1+e^{-x}}\right)=\sigma(x)\left(\frac{1 + e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}}\right)=\sigma(x)(1 - \sigma(x))\end{align*}$

그리고 이를 이용하여 cost function을 미분해보자. 

$\begin{align*}\frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta_j} \frac{-1}{m}\sum_{i=1}^m \left [ y^{(i)} log (h_\theta(x^{(i)})) + (1-y^{(i)}) log (1 - h_\theta(x^{(i)})) \right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} \frac{\partial}{\partial \theta_j} log (h_\theta(x^{(i)})) + (1-y^{(i)}) \frac{\partial}{\partial \theta_j} log (1 - h_\theta(x^{(i)}))\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ \frac{y^{(i)} \frac{\partial}{\partial \theta_j} h_\theta(x^{(i)})}{h_\theta(x^{(i)})} + \frac{(1-y^{(i)})\frac{\partial}{\partial \theta_j} (1 - h_\theta(x^{(i)}))}{1 - h_\theta(x^{(i)})}\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ \frac{y^{(i)} \frac{\partial}{\partial \theta_j} \sigma(\theta^T x^{(i)})}{h_\theta(x^{(i)})} + \frac{(1-y^{(i)})\frac{\partial}{\partial \theta_j} (1 - \sigma(\theta^T x^{(i)}))}{1 - h_\theta(x^{(i)})}\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ \frac{y^{(i)} \sigma(\theta^T x^{(i)}) (1 - \sigma(\theta^T x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{h_\theta(x^{(i)})} + \frac{- (1-y^{(i)}) \sigma(\theta^T x^{(i)}) (1 - \sigma(\theta^T x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{1 - h_\theta(x^{(i)})}\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ \frac{y^{(i)} h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{h_\theta(x^{(i)})} - \frac{(1-y^{(i)}) h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{1 - h_\theta(x^{(i)})}\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} (1 - h_\theta(x^{(i)})) x^{(i)}_j - (1-y^{(i)}) h_\theta(x^{(i)}) x^{(i)}_j\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} (1 - h_\theta(x^{(i)})) - (1-y^{(i)}) h_\theta(x^{(i)}) \right ] x^{(i)}_j \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} - y^{(i)} h_\theta(x^{(i)}) - h_\theta(x^{(i)}) + y^{(i)} h_\theta(x^{(i)}) \right ] x^{(i)}_j \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} - h_\theta(x^{(i)}) \right ] x^{(i)}_j \newline&= \frac{1}{m}\sum_{i=1}^m \left [ h_\theta(x^{(i)}) - y^{(i)} \right ] x^{(i)}_j\end{align*}$

이 역시 벡터화하면 아래와 같다. 

$\nabla J(\theta) = \frac{1}{m} \cdot X^T \cdot \left(g\left(X\cdot\theta\right) - \vec{y}\right)$ 



