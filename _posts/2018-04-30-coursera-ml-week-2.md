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
  
feature_image: '../assets/images/coursera_ML/title.png'
image: './assets/images/coursera_ML/title.png'
---
### ML:Linear Regression with Multiple Variables
![](http://cfile24.uf.tistory.com/image/2512693A58246F0D0D6F5D "머신러닝" "width:600px;height:200px;float:center;padding-left:10px;")
여러 변수에 대한 선형 회귀는 다변량 선형 회귀(multivariate linear regression)로 알려져 있다. 아래에 여러 입력 변수를 가지는 식에 대한 표기법이 있다. 

$$j^{(i)} = \text{value of feature } j \text{ in the }i^{th}\text{ training example} $$ 
$$x^{(i)} = \text{the column vector of all the feature inputs of the }i^{th}\text{ training example} $$
$$m = \text{the number of training examples} $$
$$n = \left| x^{(i)} \right| ; \text{(the number of features)} $$

다변수일 때 가설함수(hypothesis function)는 아래와 같다. 

$h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$

위 함수를 직관적으로 보기 위해서, 위의 표와 같이 $\theta_{0}$를 기본 집 값, $\theta_{1}$을 제곱미터당 가격, $\theta_{2}$를 측 당 가격으로 생각해 볼 수 있다. 이때 $x_{1}$은 집의 면적이 되고, $x_{2}$는 집의 층 수다. 

위 식은 행렬 곱을 사용하여 아래와 같이 나타 낼 수 있다.  

$$h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x$$

위는 하나의 데이터 셋에 대한 표기법이다. 여러 데이터에 대해서는 다음과 같이 표기한다. $x_{0}^{(i)} =1 \text{ for } (i\in { 1,\dots, m } )$

 $\theta$와 $x$의 행렬로 표기할 때, 모든 데이터 $i$에 대해 $x^{(i)}_0=1$이다. 이는 행렬과 벡터의 크기를 맟주기 위해서다. 

$$X = \begin{bmatrix}x^{(1)}_0  x^{(1)}_1 x^{(2)}_0  x^{(2)}_1 x^{(3)}_0  x^{(3)}_1 \end{bmatrix}$$
$$,\theta = \begin{bmatrix}\theta_0\theta_1 \end{bmatrix}$$

(m x 1)의 크기를 가지는 열 백터를 통해 아래와 같이 나타낼 수도 있다. 

$h_\theta(X) = X \theta$

##### 비용함수

$\theta$에 대한 비용함수는 아래와 같이 나타낸다. 

$J(\theta) = \dfrac {1}{2m} \displaystyle \sum_{i=1}^m \left (h_\theta (x^{(i)}) - y^{(i)} \right)^2$

위 식을 벡터화 하면 아래와 같다.

$J(\theta) = \dfrac {1}{2m} (X\theta - \vec{y})^{T} (X\theta - \vec{y})$

여기서 y값을 다음과 같이 $\vec{y}$ 벡터로 표기했다. 

##### 다 변수에서의 경사하강법(Gradient Descent)
경사하강식 역시 같은 형식을 띄며 변수의 수 'n' 만큼 반복하면된다.

$$ \text{repeat until convergence:} \; \lbrace $$
$$ \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}$$ 
$$\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} $$
$$\theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} $$
$$ \cdots $$
$$\rbrace $$

 

또는 아래와 같이 표기한다. 

 

$$ \text{repeat until convergence:} \; \lbrace $$
$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \;  \text{for j := 0..n} $$
$$\rbrace$$

##### 행렬 표기법
경사하강법을 아래와 같이 간략히 표현 할 수 있다.  

$\theta := \theta - \alpha \nabla J(\theta)$

여기서 $\nabla J(\theta)$는 아래와 같은 열 백터로 표현된다. 

$\nabla J(\theta) = \begin{bmatrix}\frac{\partial J(\theta)}{\partial \theta_0} \newline \frac{\partial J(\theta)}{\partial \theta_1} \newline \vdots \newline \frac{\partial J(\theta)}{\partial \theta_n} \end{bmatrix} $

Gradient의 j번째 행은 두 항의 곱들을 합합 값이다. 

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum\limits_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)} \right) \cdot x_j^{(i)} $$
$$= \frac{1}{m} \sum\limits_{i=1}^{m} x_j^{(i)} \cdot \left(h_\theta(x^{(i)}) - y^{(i)} \right) $$

종종 위의 식은 벡터의 곱(내적)으로 표현되며, 아래와 같이 쓸 수 있다. 

$\begin{align*}\; &\frac{\partial J(\theta)}{\partial \theta_j} &=& \frac1m \vec{x_j}^{T} (X\theta - \vec{y}) \newline\newline\newline\; &\nabla J(\theta) & = & \frac 1m X^{T} (X\theta - \vec{y}) \newline\end{align*}$

최종 경사하강법을 행렬로 표기하면 아래와 같다. 

$\theta := \theta - \frac{\alpha}{m} X^{T} (X\theta - \vec{y})$

##### 정규화(Feature Normalization)
![](http://cfile22.uf.tistory.com/image/2579EC3A58246F0E183DD8 "Feature Normalization" "width:600px;height:300px;float:center;padding-left:10px;")

입력 변수 값의 범위를 비슷하게 만들어 gradient descent의 속도를 높일 수 있다. $\theta$는 변수 값들의 범위가 작은 경우 빨리 수렴하며, 큰 경우 천천히 수렴한다. 또한 변수가 고르지 않는 경우 위의 왼쪽과 같이 요동치며 비효율적으로 수렴한다.

이러한 문제을 막기 위한 방법 중 하나는 아래와 같이 거의 같은 범위 내에 데이터가 있게 하는 것이다.

$-1 < x_{(i)} < 1$

or

$-0.5 < x_{(i)} < 0.5$

이것은 학습에 꼭 필요하지는 않으나 속도를 올리는데 유용하다. 두가지 방법, Feature Scaling, Mean Normalization을 사용하여 입력 변수의 범위를 조정한다.  Feature Scaling에서는 입력 변수를 변수의 범위($v_{max}-v_{min}$)로 나눈다. 그 결과 범위는 항상 1이 된다. Mean Normalization에서는 입력 변수에 평균을 뺀 다음 범위로 나눈다. 그 결과로 나온 변수들의 평균은 항상 0이 된다. 아래 식과 같이 나타낼 수 있다. 

$x_i := \dfrac{x_i - \mu_i}{s_i}$

여기서 $\mu_i$는 모든 변수 $x_i$의 평균이며, $s_i$는 변수의 범위를 사용하거나 표준편차를 사용한다. 

##### Gradient Descent Tips
![](http://cfile21.uf.tistory.com/image/272D803A58246F0F350EEE "Gradient Descent Tips" "width:600px;height:500px;float:center;padding-left:10px;")

Debugging gradient descent. x축으로 반복(iteration)횟수를, y축으로 비용함수($J(\theta)$)를 그린다. 이 때 $J(\theta)$가 증가하면 $\alpha$ 값을 줄인다. 

Automatic convergence test. 반복문 내에서 $J(\theta)$가 특정값 E 보다 작은 경우 수렴(convergence)했다고 하며, 여기서 E는 보통 $10^{-3}$과 같은 작은 값을 사용한다. 하지만 이러한 임계값(threshold value)을 결정하는 것은 어렵다. 

  $\alpha$가 충분히 작은 경우 $J(\theta)$는 점점 수렴하며. Andrew Ng은 1/3씩 줄여 나가는 것을 권한다. 

##### Features and Polynomial Regression
몇 가지 방법으로 가설함수와 변수 값을 향상 할 수 있다. 여러 변수를 하나로 곱하는 것으로 새로운 변수를 만들 수 있다. $x_{3}=x_{1} * x_{2}$.

Polynomial Regression
![](http://cfile3.uf.tistory.com/image/2160573A58246F10252E16 "Polynomial Regression" "width:600px;height:150px;float:center;padding-left:10px;")
주어진 데이터에 잘 일치하지 않는 경우 가설함수는 굳이 선형(직선)일 필요는 없다. 제곱, 세제곱, 제곱근과 같은 항을 가설함수에 추가하는 것으로 비선형 함수로 바꿀 수 있다. 예를 들어 다음과 같은 가설함수 $h_\theta(x) = \theta_0+\theta_1 x_1$가 있을 때  ,$x_{1}$ 을 기반으로 변수를 더 추가하는 과정을 통해 다음과 같은 이차함수 $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2$, 그리고 삼차 함수 $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3$를 얻을 수 있다. 여기서 새로운 변수 $x_2$, $x_3$를 다음과 같이 생각 할 수 있다. $x_2 = x_1^2$, $x_3 = x_1^3$.

제곱근을 사용해서 다음과 같은 가설함수를 만들 수 있다. $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}$.

이러한 방법으로 변수를 추가하는 경우 Feature Scaling을 고려 해야 한다.

ex. $x_1$이 1~1000의 범위를 가질때 $x_1^2$은 1~10000000, $x_1^3$은 1~1000000000의 범위를 가진다. 여기서 정규화의 유용성을 볼 수 있다. 

##### Normal Equation
![](http://cfile23.uf.tistory.com/image/2466E43A58246F112228BE "Normal Equation" "width:600px;height:300px;float:center;padding-left:10px;")
정규방정식(Normal Equation)은 반복문 없이 한번에 최소 값을 찾는 방법이다. 

$\theta = (X^T X)^{-1}X^T y$

여기서는 Feature Scaling을 할 필요가 없다. 

이것의 증명은 선형대수적 지식을 필요로 하며 다행이 세부적인 부분에 대해서는 크게 신경 쓰지 않아도 무방하다. 관심이 있다면 아래 링크를 참조하길 바란다. 

https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)

http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression

아래 표에 gradient descent와 normal equation을 비교하였다. 

|  gradient descent | normal equation |
|--------|--------|
|  $\alpha$ 값 선택 필요       |  $\alpha$ 값 선택 할 필요 없음       |
|   많은 반복문을 통해 수렴      |  반복문이 필요 없음        |
|  $O(kn^2)$       |  $O(kn^3)$, $(X^T X)$의 inverse 계산       |
|  n이 큰 경우에도 잘 동작      | n이 큰 경우 매우 느림        |

normal equation을 사용하는 경우 역행렬을 계산하는데 $\mathcal{O}(n^3)$이 소요 된다. 따라서 변수가 매우 많은 경우, normal equation은 느릴 수 밖에 없다. 실제로 n이 10000 이상인 경우 normal equation을 사용 하는 것보다 gradient descent를 사용하는게 더 낫다. 

Normal Equation Noninvertibility

ocative에서 normal equation을 구현 할 때 'inv'함수를 사용 하는 것 보다 'pinv' 함수를 사용하기를 권한다. 다음과 같은 이유로 $(X^T X)$의 역행렬이 없을 수 (noninvertible) 있기 때문이다. 

linear dependent와 같이 변수들 사이에 관련성이 있는 경우, (한 변수를 다른 변수들을 사용하여 표현할 수 있음)

데이터 보다 변수가 많은 경우 ($m < n$), 

위 문제를 해결 하기 위해 몇몇 변수를 지우거나,  정규화(Regularization) 과정을 거친다. 
















