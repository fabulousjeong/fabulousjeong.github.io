---
title: Week 3 Logistic Regression
category: CourseraMachineLearning
excerpt: |
  이제 회귀(Regression)문제를 분류(Classification)문제로 바꿔 생각해보자. 로지스틱회귀(Logistic Regression)이라는 단어에 대해 헷갈릴수 있는데 이는 관용적으로 붙여진 이름이며, 실제로는 회귀(Regression)가 아니라 분류(Classification) 문제를 다룬다.
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  
  ref: [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning "Coursera ML")
  
  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")
  
feature_image: "../assets/images/coursera_ML/title.png"
image: "./assets/images/coursera_ML/title.png"
---

### ML: Logistic Regression

이제 회귀(Regression)문제를 분류(Classification)문제로 바꿔 생각해보자. 로지스틱회귀(Logistic Regression)이라는 단어에 대해 헷갈릴수 있는데 이는 관용적으로 붙여진 이름이며, 실제로는 회귀(Regression)가 아니라 분류(Classification) 문제를 다룬다.


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

$$
\begin{align*}& Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \newline & \rbrace\end{align*}
$$

앞의 식을 미분하면 아래와 같다. 

$$
\begin{align*} & Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}
$$

위 알고리즘은 앞에서 다룬 선형 회귀와 동일하다. 이 경우 역시 동시에 모든 theta값을 업데이트 해야한다.   

백터화 하면 아래와 같다. 

$\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})$

##### Partial derivative of $J(\theta)$

먼저 sigmoid function의 미분을 구해보자

$$
\begin{align*}\sigma(x)'&=\left(\frac{1}{1+e^{-x}}\right)'=\frac{-(1+e^{-x})'}{(1+e^{-x})^2}=\frac{-1'-(e^{-x})'}{(1+e^{-x})^2}=\frac{0-(-x)'(e^{-x})}{(1+e^{-x})^2}=\frac{-(-1)(e^{-x})}{(1+e^{-x})^2}=\frac{e^{-x}}{(1+e^{-x})^2} \newline &=\left(\frac{1}{1+e^{-x}}\right)\left(\frac{e^{-x}}{1+e^{-x}}\right)=\sigma(x)\left(\frac{+1-1 + e^{-x}}{1+e^{-x}}\right)=\sigma(x)\left(\frac{1 + e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}}\right)=\sigma(x)(1 - \sigma(x))\end{align*}
$$

그리고 이를 이용하여 cost function을 미분해보자. 

$$
\begin{align*}\frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta_j} \frac{-1}{m}\sum_{i=1}^m \left [ y^{(i)} log (h_\theta(x^{(i)})) + (1-y^{(i)}) log (1 - h_\theta(x^{(i)})) \right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} \frac{\partial}{\partial \theta_j} log (h_\theta(x^{(i)})) + (1-y^{(i)}) \frac{\partial}{\partial \theta_j} log (1 - h_\theta(x^{(i)}))\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ \frac{y^{(i)} \frac{\partial}{\partial \theta_j} h_\theta(x^{(i)})}{h_\theta(x^{(i)})} + \frac{(1-y^{(i)})\frac{\partial}{\partial \theta_j} (1 - h_\theta(x^{(i)}))}{1 - h_\theta(x^{(i)})}\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ \frac{y^{(i)} \frac{\partial}{\partial \theta_j} \sigma(\theta^T x^{(i)})}{h_\theta(x^{(i)})} + \frac{(1-y^{(i)})\frac{\partial}{\partial \theta_j} (1 - \sigma(\theta^T x^{(i)}))}{1 - h_\theta(x^{(i)})}\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ \frac{y^{(i)} \sigma(\theta^T x^{(i)}) (1 - \sigma(\theta^T x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{h_\theta(x^{(i)})} + \frac{- (1-y^{(i)}) \sigma(\theta^T x^{(i)}) (1 - \sigma(\theta^T x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{1 - h_\theta(x^{(i)})}\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ \frac{y^{(i)} h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{h_\theta(x^{(i)})} - \frac{(1-y^{(i)}) h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{1 - h_\theta(x^{(i)})}\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} (1 - h_\theta(x^{(i)})) x^{(i)}_j - (1-y^{(i)}) h_\theta(x^{(i)}) x^{(i)}_j\right ] \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} (1 - h_\theta(x^{(i)})) - (1-y^{(i)}) h_\theta(x^{(i)}) \right ] x^{(i)}_j \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} - y^{(i)} h_\theta(x^{(i)}) - h_\theta(x^{(i)}) + y^{(i)} h_\theta(x^{(i)}) \right ] x^{(i)}_j \newline&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} - h_\theta(x^{(i)}) \right ] x^{(i)}_j \newline&= \frac{1}{m}\sum_{i=1}^m \left [ h_\theta(x^{(i)}) - y^{(i)} \right ] x^{(i)}_j\end{align*}
$$

이 역시 벡터화하면 아래와 같다. 

$\nabla J(\theta) = \frac{1}{m} \cdot X^T \cdot \left(g\left(X\cdot\theta\right) - \vec{y}\right)$ 

##### Advanced Optimization

"Conjugate gradient", "BFGS", "L-BFGS"는 $\theta$를 구함에 있어 gradient descent보다 더 정교하다. 앤드류응(A. Ng)은 이러한 방법을 사용 할때 수치해석의 전문가가 아니라면 본인이 직접 알고리즘을 짜기보다는 기존에 구현 된 최적 알고리즘을 사용하기를 제안한다. 이러한 알고리즘은 옥타브(Octave) 라이브러리에서 제공한다. 

먼저 주어진 입력값 $\theta$가 들어 갈때 그 결과를 보기위한 함수가 필요하다. 

$\begin{align*} & J(\theta) \newline & \dfrac{\partial}{\partial \theta_j}J(\theta)\end{align*}$

위의 두 결과를 반환하는 함수를 아래와 같이 코딩 할 수 있다.  

``` matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Octave의 "fminunc()" 최적화 알고리즘을 "optimset()"함수와 함께 사용 할 수 있다.  "optimset()"함수는 "fminunc()"함수로 아래와 같이 옵션들을 전달한다. 
``` python
options = optimset('GradObj', 'on', 'MaxIter', 100);
      initialTheta = zeros(2,1);
      [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```
위 식에서 보듯 fminunc()함수에 cost function, 초기값, 그리고 기존에 생성한 옵션을 넣는다.

##### Multiclass Classification: One-vs-all

지금부터는 2가지 카테고리 이상의 경우에서 데이터 분류를 다룬다. y={0,1} 대신, y의 범주를 늘려 y={0,1,...,n] 같이 정의 한다. 

이 경우 위의 문제를 n+1개의 바이너리 분류법으로 나눠 생각한다.(+1을 해주는 이유는 y의 인덱스가 0부터 시작하기 때문이다.) 위 방법에서는 각 경우에서 y중 하나에 해당하는 확률을 예측한다.  

$$
\begin{align*}& y \in \lbrace0, 1 ... n\rbrace \newline& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline& \cdots \newline& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline\end{align*}
$$

위에서는 각각의 경우에서 일단 하나의 클래스를 선택한 다음 나머지 모두를 하나로 묶는 방법으로 바이너리 분류를 만든다. 이러한 binary logistic regression  작업을 각각의 경우에서 반복한다. 가설함수는 그 중 가장 확률이 높은 경우 선택한다. 

### ML:Regularization

![](http://cfile26.uf.tistory.com/image/210AA245586BA7342D4D6F "Regularization" "width:600px;height:200px;float:center;padding-left:10px;")

##### Overfitting(과적합) 문제

Regularization(정규화)는 과적합 문제를 다루기 위해 고안되었다. 

High bias 또는 under fitting은 생성한 가설함수 h가 데이터의 추이에 잘 맞지 않는 경우를 말한다. 이 경우, 보통 함수가 너무 간단하거나 변수를 너무 적게 쓴는 것이 원인이다. 만약 가설함수로 $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2$를 사용 한다면 선형 모델이 일반적으로 데이터에 잘 맞을 수도 있지만 그렇지 않는 경우가 있을 수도 있다. 

반면, 기존의 학습데이터에는 잘 맞지만 새로운 데이터에 대해서는 잘 예측하지 못하는 Overfitting 이나 high variance 경우도 있다. 이러한 경우는 복잡한 함수를 사용하는 것으로 인해 데이터와 상관없는 불필요한 곡선에 의해 발생한다. 

이러한 현상은 선형 및 로지스틱 회귀에서 공통적으로 일어난다. 다음으로 과적합을 해결하기위한 두가지 방법을 소개한다. 

1) 변수의 수를 줄인다. 

  a) 사용할 변수를 직접 선택한다. 

  b) 모델 선택 알고리즘을 사용한다.(다음 장에서 다룸)

2) Regularization(정규화) 

  모든 변수를 그대로 사용한다, 하지만 파라미터($theta_j$)의 크기를 줄인다. 

정규화를 통해 다수의 유용한 변수를 사용하는 경우에도 잘 동작하게 된다.  



##### Cost Function

가설 함수에 Overfitting이 발생 하는 경우, 몇몇 항의 코스트를 증가시키는 방법을 통해 그 항에서의 Weight를 줄일수 있다. 아래의 함수를 이차함수로 바꾸고 싶다고 하자. 

$\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4$

이 경우 $\theta_3x^3$, $\theta_4x^4$항에서 받는 영향을 없애야한다. 저 항에 해당하는 변수를 제거하는 대신 Cost function의 형태를 아래와 같이 바꾼다. 

$min_\theta\ \dfrac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000\cdot\theta_3^2 + 1000\cdot\theta_4^2$

가 장 끝 두 변수에 해당하는 항 $\theta_3$, $\theta_4$에 1000씩 곱함으로써 받는 영향을 크게 만든다. Cost function을 0에 가깝게 만들기 위해 $\theta_3$, $\theta_4$은 거의 0에 가까운 값을 가지게 된다. 이렇게 함으로써 $\theta_3x^3$와 $\theta_4x^4$는 매우 작은 값을 가진다. 

또한 아래와 같은 방법을 통해 모든 $\theta$ 파라미터의 값을 정규화 할 수 있다.    

$min_\theta\ \dfrac{1}{2m}\ \left[ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2 \right]$ 

여기서 $\Lambda$를 Regularization 파라미터라고 불린다. 이 파라미터는 $\theta$ 항의 Cost를 얼마나 증가시키는 지를 보여준다. 아래 링크를 통해 정규화의 영향을 시각적으로 볼 수 있다.  https://www.desmos.com/calculator/1hexc8ntqp

위의 식을 통해 가설함수를 조금 부드럽게 하고, overfitting 문제를 줄 일 수 있다. 반면 \lambda\가 너무 커지게 되면 함수의 곡면이 너무 단조로워지며 ,따라서 Under fitting 문제가 발생한다. 


##### Regularized Liner Regression

성형 회귀와 로지스틱 회귀 둘 모두 정규화를 할 수 있다. 먼저 선형회기의 경우에 대해 알아보자. 

![](http://cfile9.uf.tistory.com/image/242B8545586BA7331052A8 "Regularized Liner Regression" "width:600px;height:200px;float:center;padding-left:10px;")

##### Gradient Descent

$\theta_0$에 대해서는 정규화를 하지 않으므로 앞서 세운 Gradient Descent에서 함수에서 $\theta_0$항을 떼어 내서 생각하자.

$$
\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace \end{align*}
$$

위 식에서 $\frac{\lambda}{m}\theta_j$ 항이 정규화 과정을 수행한다. 

위 업데이트 룰을 번형하여 아래와 같이 나타낼 수도 있다. 

$\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$

위 식의 첫번째 항인 $1 - \alpha\frac{\lambda}{m}$는 항상 1보다 작다. 직관적으로 매 업데이트 마다 $\theta_j$ 값이 작아짐을 알 수 있다. 

 두번째 항은 기존에서 구한 값과 같음에 주목하자. 


##### Normal Equation

반복문을 사용하지 않는 Normal Equation에서의 정규화 과정에 대해 알아보자. 정규화 작업을 하기 위해 아래와 같이 식의 괄호내에 새로운 항을 더한다. 

$$
\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}
$$

행렬 L은 가장 왼쪽 위의 값이 0이고, 대각성분의 값은 1이, 나머지 부분은 0으로 채워져 있다. 이 행렬은 (n+1)×(n+1)크기를 가져야한다. 직관적으로 위에서는 단위행렬($x_0$항은 제외)에 실수 $\lambda$를 곱한 형태를 띈다. 앞에서 다룬 내용을 상기해 보면 m이 n보다 작거나 같은 경우 $X^TX$는 역행렬을 가지지 못했다. 하지만  $X^TX + \lambda \cdot L$는 역행렬을 가진다.

##### Regularized Logistic Regression

![](http://cfile2.uf.tistory.com/image/252DC245586BA7330D53E4 "Regularized Logistic Regression" "width:600px;height:170px;float:center;padding-left:10px;")
로지스틱 회귀에서의 정규화도 위의 선형 회귀에서와 비슷한 과정을 통해 수행 할수 있다. 먼저 cost function부터 살펴보자.

##### Cost Function

위에서 다룬 로지스틱 회귀에서의 비용 함수를 다시 한번 살펴보자

$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)})) \large]$

아래와 같이 가장 오른쪽 항에 정규화 항을 더한다.   

$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$

가장 오른쪽 항에서 theta는 1부터 시작하는 것에 주의하자

##### Gradient Descent

선형 회귀의 경우와 동일하게 $\Theta_0$항은 분리하여 업데이트 한다. 

$$
\begin{align*}& \text{Repeat}\ \lbrace \newline& \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline& \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline& \rbrace\end{align*}
$$

위 식은 위에서 다룬 선형회귀에서의 정규화 Gradient Descent 과정과 동일하다. 


##### Initial Ones Feature Vector: Constant Feature

학습 시작전에 변수 벡터의 값을 설정하는 것은 중요하다. 일반적으로 전체 변수값을 1로 설정한다. 좀 더 자세히 설명하자면, X를 변수의 행렬이라고 하면, $X_0$를 1로 채운다. 다음은 constant 변수를 설정하는 몇 가지 방법이 있다. Bias 변수는 데이터에 가장 잘 맞는 벡터를 학습하는 간단한 방법이다. 예를 들어 변수 $X_1$에 대해 학습하는 경우를 생각해 보자. $X_0$변수를 고려하지 않는다면, 식은 다음과 같다. $\theta_1*X_1=y$위 그래프는 원점을 지나며, 기울기는 $y/\theta$와 같다. $x_0$ 변수는 앞의 직선이 y축에서 어떤 점을 지나는 지를 결정한다. 



