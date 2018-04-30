---
title: Week 1 Introduction
category: CourseraMachineLearning
excerpt: |
  머신러닝이란? 머신러닝에 관한 2가지 정의가 있다. 첫 번째는 Arthur Samuel의 정의인 "구체적인 프로그래밍의 도움 없이 학습하는 능력을 컴퓨터에게 주는 학문"이다.
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  
  ref: [https://www.coursera.org/learn/machine-learning/home/week/1](https://www.coursera.org/learn/machine-learning/home/week/1 "Coursera ML")
  
  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")
  
feature_image: ../assets/images/coursera_ML/title.png
image: ../assets/images/coursera_ML/title.png
---
### ML: 서론

머신러닝이란?
머신러닝에 관한 2가지 정의가 있다. 첫 번째는 Arthur Samuel의 정의인 "구체적인 프로그래밍의 도움 없이 학습하는 능력을 컴퓨터에게 주는 학문"이다. 두 번째는 Tom Mitchell이 제안한 보다 현대적인 관점의 정의다. 여기서 머신러닝이란 "어떤 경험(experience) E와 관련된 작업(Task) T, 성능(performance)P가 있을 때 경험 E로 부터 작업 T의 성능 P를 증가시키는 학습 컴퓨터 프로그램"을 말한다.
예) 체스 게임

E = 체스 기사들의 체스 게임 경험

T = 체스 게임의 시행

P = 다음 게임에서 이길 확률

![](https://raw.githubusercontent.com/mahmoudparsian/data-algorithms-book/master/misc/machine_learning.jpg "머신러닝" "width:600px;height:400px;float:center;padding-left:10px;")
ref: https://github.com/mahmoudparsian/data-algorithms-book/

위의 그림을 보면 머신러닝이란 주어진 데이터(입력, 결과)를 사용하여 알고리즘(예측모델)을 얻어내는 것 이라고 볼 수 있다. 일반적으로, 많은 머신러닝 문제는 지도학습(supervised learning)과 자율학습(unsupervised learning) 두 가지로 나뉜다.


##### 지도학습
![](http://cfile7.uf.tistory.com/image/2115833F5819E3A627DC7B "지도학습" "width:600px;height:300px;float:center;padding-left:10px;")

지도학습에서는 위와 같이 입력과 출력의 관계에 있는 데이터 셋(X 표시)이 주어진다. 지도학습문제는 일반적으로 회귀(Regression)와 분류(Classification)로 나눌 수 있다. 회귀 문제에서는 연속적인 출력에서의 결과 값을 예상하며, 다시 말해 입력 변수를 연속 함수로 맵핑하는 것과 같다. 반면, 분류문제에서는 불연속적인 출력에서의 결과 값을 예상한다. 다시 말해, 여기서는 입력변수를 분리되어 있는 카테고리에 맵핑한다.

예 1:

부동산 시장에서 집 크기가 데이터가 주어 졌을때 그 가격을 예상하는 것을 예로 들 수 있다. 여기서 집 크기 함수에 관한 가격은 연속적이며 따라서 회귀 문제다. 또한 이 문제를 주어진 가격(ex. 200k)이 있을 때 이를 넘는지 안 넘는지에 관한 문제로 바꾸는 것을 통해 분류 문제처럼 생각 할 수 있다. 이 때 학습 데이터는 집값에 대한 두 불연속 카테고리로 분류된다.

예 2:

(a) 회귀 - 남/녀 사진이 있을 때, 사진에 기반하여 그/그녀의 나이를 예상 하는 것

(b) 분류 - 남/녀 사진이 있을 때, 그/그녀가 고등학생/대학생/대학원생 중 어디에 속하는지 예상하는 것, 다른 예로는 은행에서 신용 정보에 기반하여 대출을 할 지 하지 않을지 결정 하는 것이 있다.

##### 자율학습
![](http://cfile3.uf.tistory.com/image/21128C3F5819E3A22C490F "지도학습" "width:600px;height:400px;float:center;padding-left:10px;")

반면, 자율학습에서는 결과 값이 무엇인지에 대한 정보가 없거나 적은 문제를 다룬다. 결과 값에 대한 변수 없이 데이터로 부터 특정 구조를 생성할 수 있다. 주어진 데이터 사이의 관련성에 기반한 클러스트링 과정을 통해 특정 카테고리를 생성한다. 자율학습에서는 예상 값에 기반한 피드백을 받을 수 없다, 즉 나온 결과가 맞는지 틀렸는지 알 수 없다.

예:

클러스트링(Clustering): US Economy에서 쓰여진 1000개의 에세이가 주어 질 때, 관련성(단어 빈도, 페이지 수, 문장길이)에 기반하여 자동으로 에세이를 그룹핑함

비클러스트링(Non-Clustering): "Cocktail Party Algorithm", 지저분한 데이터에서 특정 구조를 찾는 작업. 가령 굉장히 시끄러운 파티에서 특정인의 목소리나, 노래를 찾는 작업. 아래에 이해에 도움이 되는 Quora의 답변을 링크로 남긴다. https://www.quora.com/What-is-the-difference-between-supervised-and-unsupervised-learning-algorithms

### 단 변수 선형 회기(Linear regression)

##### 모델 표현

회기 문제를 다시 보면, 입력 변수에 대한 출력 변수를 연속 함수에 맞추고 있다. 일반적으로 선형회기(Linear regression)은 "단변량 선형회기 (univariate linear regression)"로 알려져 있다. 이 모델은 하나의 입력 \\(x\\) 가 있을때 출력 \\(y\\)를 예상하는데 사용된다. 입/출력의 상관관계를 이미 알고 있으므로 지도 학습을 수행할 수 있다.

##### 가설 함수(Hypothesis Function)

일반적인 가설 함수는 아래와 같은 형태를 지닌다.

$$\hat{y}=h_{\theta}(x)=\theta_0+\theta_1x$$

위 수식이 직선의 방정식과 유사하다는데 주목하자. \\( \theta_0 \\)와 \\(\theta_1\\)에 의해 표현 되는 함수 \\(h_{\theta}(x)\\)에 의해 출력 \\(y\\)를 예측 한다. 즉 입력을 출력에 맵핑하는 함수(직선) \\(h_{\theta}\\)를 구하는 작업이다.

아래와 같은 학습데이터가 있다고 가정하자.
(x,y): {(0,4), (1,7), (2,7), (3,8)}
![](http://cfile22.uf.tistory.com/image/2574A041586B9EE623EE5A "가설함수" "width:600px;height:400px;float:center;padding-left:10px;")
임의 추정으로 $h_{theta}$ 를 다음과 같이 생성할 수 있다. $\theta_0=2$, $\theta_1=2$, 이때 가설 함수는  다음과 같다. $h_{\theta}=2+2x$. 여기서 입력을 1로 두면 $y$는 4가 된다. 이는 실제 값과 3차이가 난다. $\theta_0$와 $\theta_1$ 값을 조정하여  $x-y$ 평면에 있는 데이터에 최대한 맞는 직선이 되게 한다.

##### 비용함수(Cost Function)

 비용함수를 통해 가설함수가 얼마나 정확한지 측정 할 수 있다. 아래 식과 같이 비용함수는 입력(x)를 가설함수에 넣은 결과 값과 실제 출력(y) 간의 차이의 평균 값으로 표현 된다.

$$J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2$$

잠시 살펴 보면, $\dfrac {1}{2} \bar{x}$ 에서 $\bar{x}$는 $(h_\theta (x_{i})-y_{i})$ 제곱의 평균, 즉 예상 값과 평균 값의 차이와 같다. 이 함수는 다른 말로 제곱 오차 함수(Squared error function) 또는 평균 제곱근 편차(Mean squared error)로 불린다. 평균은 절반으로 나눠지게 표현$(\dfrac {1}{2m})$ 되는데, 보통 그래디언트 하강(gradient descent) 계산의 편의를 위해 $\dfrac {1}{2}$ 항을 생략한다. 이제 실제 값과 비교하여 가설함수의 정확도를 측정 할 수 있다. 

 위 그림과 같이 학습 데이터가 X-Y 평면에 뿌려져 있다고 시각적인 측면에서 생각해 보자. 뿌려진 데이터 위를 지나는 $h_{\theta}(x)$로 정의 된 직선을 그릴수 있다. 여기서 우리의 목적은 데이터와 직선사이의 거리의 제곱의 평균 값이 최소인 직선을 찾는 것이다. 가장 최선인 경우는 직선이 모든 데이터 위를 지나는 경우로 이때 $J(\theta_0, \theta_1)$는 $0$이다. 
![](http://cfile2.uf.tistory.com/image/262B763F5819E3A50D7981 "비용함수" "width:600px;height:300px;float:center;padding-left:10px;")
![](http://cfile23.uf.tistory.com/image/2612633F5819E3A42C7512 "비용함수" "width:600px;height:300px;float:center;padding-left:10px;")

### ML:경사 하강법(Gradient Descent)

 지금 까지 가설함수와 이 함수가 주어진 데이터에 얼마나 잘 맞는지를 측정하는 법에 대해서 알아 보았다. 이제 가설함수 내 파라미터를 예측하는 과정이 남았다. 경사하강법을 통해 알아 보자.

 $\theta_{0}$와 $\theta_{1}$의 평면위에 가설함수의 그래프를 생각해보자(비용함수의 그래프를 그려보자).  이것은 높은 단계의 추상화 과정을 동반하므로, 다소 어려울 수 있다. 
![](http://cfile4.uf.tistory.com/image/261C423F5819E3A21F6A8F "비용함수" "width:600px;height:300px;float:center;padding-left:10px;")
위 그림과 같이 $\theta_{0}$를 x축으로 $\theta_{1}$을 y축으로 두고, 비용함수의 값을 z축에 놓는다. 그래프 위의 점은 $\theta$ 파라미터 값에 대한 가설함수의 비용함수 결과값이다. 우리는 이미 최적의 가설함수를 이루는 파라미터 값은 비용함수 그래프의 최소 값(minimum)에서의 파라미터 값과 같다는 것을 알고있다. 최소값을 가지는 비용함수의 파라미터 값을 구하는 한 가지 방법은 비용함수의 접선(derivative)을 도입하는데서 시작한다. 접선의 경사도는 그 점에서의 도함수이며, 이 값은 움직여야할 최적의 방향을 알려준다. 아래와 같이 매 스텝마다 학습속도(learning rate) $\alpha$만큼 접선 방향으로 하강한다. 
![](http://cfile24.uf.tistory.com/image/24127F3F5819E3A32C4209 "비용함수" "width:600px;height:300px;float:center;padding-left:10px;")
경사하강 알고리즘은 아래와 같다. 

수렴 할때 까지 반복:
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$
여기서 j=0,1 은 각 feature의 인덱스 번호와 같다. 위 식은 직관적으로 아래와 같이 생각 할 수 있다. 

수렴 할때 까지 반복:
$$\theta_j := \theta_j - \alpha [\text{Slope of tangent aka derivative in j dimension}]$$

##### 선형회귀에서의 경사하강법
특히 선형 회귀의 경우에 적용하면, 위 그라디언트 강하식의 새로운 형태를 유도 할 수 있다. 실제 비용함수와 가설함수를 아래 식과 같이 대체 할 수 있다. 
$$  \text{repeat until convergence: } \lbrace $$
$$ \theta_0 :=  \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}(h_\theta(x_{i}) - y_{i}) $$
$$ \theta_1 :=  \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m}\left((h_\theta(x_{i}) - y_{i}) x_{i}\right) $$
$$   \rbrace $$

여기서 m은 학습 데이터의 크기 이며, $\theta_{0}$는 $\theta_{1}$과 같이 동시에 바뀌는 상수이며, $x_{i}$, $y_{i}$는 학습데이터의 값이다. $\theta_{j}$에 따라 $\theta_{0}$과 $\theta_{1}$ 두 가지 경우로 나눠서 식이 세워지며, $\theta_{1}$에서는 도함수에 의해 $x_{i}$가 곱해진다. 요약하면, 가설함수를 설정한 다음 경사하강법의 식을 반복하는 과정을 통해 점점 더 정확한 가설함수를 얻을 수 있다. 

선형회귀에서의 경사하강법 시각적 예시

링크의 영상(https://www.youtube.com/watch?v=WnqQrPNYz5Q)에서 가설함수가 어떻게 오차를 줄이면서 성능을 증가 시키는지 시각적으로 볼 수 있다. 




















