---
title:  Week 9 Anomaly Detection and Recommender Systems
category: CourseraMachineLearning
excerpt: |
  Anomaly에서는 데이터 셋 ${x^{(1)}, x^{(2)},\dots,x^{(m)}}$이 주어지고, 새로운 데이터 $x_{test}$가 입력되었을 때, 이 데이터가 정상인지 이상인지 판단하고자 한다. 
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  
  ref: [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning "Coursera ML")
  
  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")
  
feature_image: "../assets/images/coursera_ML/title.png"
image: "./assets/images/coursera_ML/title.png"
---

### Anomaly Detection

##### Problem Motivation

![](http://cfile9.uf.tistory.com/image/237959475866E70F16B17B "Problem Motivation" "width:600px;height:300px;float:center;padding-left:10px;")

Anomaly에서는 데이터 셋 ${x^{(1)}, x^{(2)},\dots,x^{(m)}}$이 주어지고, 새로운 데이터 $x_{test}$가 입력되었을 때, 이 데이터가 정상인지 이상인지 판단하고자 한다. 

![](http://cfile26.uf.tistory.com/image/237D5E475866E71014AE63 "Problem Motivation" "width:600px;height:300px;float:center;padding-left:10px;")

이를 위해 주어진 데이터가 "이상"일 확률을 계산하는 새로운 모델 p(x)를 정의한다. 또한 문턱값 ϵ(입실론)을 사용하여 데이터가 이상인지 정상인지를 판단하는 기준으로 삼는다.    
보통 "사기"를 찾아내는데 Anomaly Detection을 사용한다. 

- $x^{(i)} =$사용자 i의 피쳐들
- 데이터 기반으로 정의한 Model p(x)  
- p(x)<ϵ 를 사용하여 비정상적인 사용자를 판별

만약 Anomaly Detection 결과에서 매우 많은 비율로 "비정상"이 나온다면, 문턱값 ϵ 을 줄인다. 

##### Gaussian Distribution

가우시안 분포는 가장 유명한 종 형태의 곡선으로 함수 $\mathcal{N}(\mu,\sigma^2)$로 나타낸다. x∈ℝ라고 할 때, 평균은 $\mu$, 분산은 $\sigma^2$이라고 하면 $x \sim \mathcal{N}(\mu, \sigma^2)$ 과 같이 표현 할 수 있다.  ~(tilde)는 distributed as로 읽는다. 평균 값 $\mu$는 곡선의 중심을 나타내고 표준편차 $\sigma$는 곡선의 너비를 나타낸다.

![](http://cfile8.uf.tistory.com/image/217561475866E710181477 "Gaussian Distribution" "width:600px;height:400px;float:center;padding-left:10px;")

함수를 풀어 쓰면 다음과 같다. 

$\large p(x;\mu,\sigma^2) = \dfrac{1}{\sigma\sqrt{(2\pi)}}e^{-\dfrac{1}{2}(\dfrac{x - \mu}{\sigma})^2}$

데이터셋으로부터 $\mu$를 추정할 수 있다. 

$\mu = \dfrac{1}{m}\displaystyle \sum_{i=1}^m x^{(i)}$

또 다른 파라미터인 $\sigma^2$ 역시 다음의 식으로 구할 수 있다.   

$\sigma^2 = \dfrac{1}{m}\displaystyle \sum_{i=1}^m(x^{(i)} - \mu)^2$

![](http://cfile30.uf.tistory.com/image/227966475866E711162EE8 "Gaussian Distribution" "width:600px;height:300px;float:center;padding-left:10px;")

##### Algorithm

실수값 벡터로 표현되는 학습 데이터 셋 $\lbrace x^{(1)},\dots,x^{(m)}\rbrace$이 주어진다.  

$p(x) = p(x_1;\mu_1,\sigma_1^2)p(x_2;\mu_2,\sigma^2_2)\cdots p(x_n;\mu_n,\sigma^2_n)$

통계에서는 이를 각 데이터 x의 피쳐에 대한 "독립 추정"이라고 한다. 좀 더 간단하게 다음과 같이 표현한다. 

$= \displaystyle \prod^n_{j=1} p(x_j;\mu_j,\sigma_j^2)$

즉 데이터의 각 피쳐에 대한 Anomaly Detection을 계산한 다음, 이를 전부 곱해서 x가 이상인지 판별한다. 

The Algorithm
![](http://cfile29.uf.tistory.com/image/21651F475866E7111D722F "Algorithm+" "width:600px;height:300px;float:center;padding-left:10px;")

비정상의 기준으로 판별할 수 있는 피쳐 $x_j$를 결정한다. 

각 피쳐에 대해 $\mu_1,\dots,\mu_n,\sigma_1^2,\dots,\sigma_n^2$를 구한다. 

$$\mu_j = \dfrac{1}{m}\displaystyle \sum_{i=1}^m x_j^{(i)}$$

$$\sigma^2_j = \dfrac{1}{m}\displaystyle \sum_{i=1}^m(x_j^{(i)} - \mu_j)^2$$

이를 곱해서 p(x)를 구한다. 

$p(x) = \displaystyle \prod^n_{j=1} p(x_j;\mu_j,\sigma_j^2) = \prod\limits^n_{j=1} \dfrac{1}{\sqrt{2\pi}\sigma_j}exp(-\dfrac{(x_j - \mu_j)^2}{2\sigma^2_j})$

p(x)<ϵ 라면 "이상"으로 판별한다. 

##### Developing and Evaluating an Anomaly Detection System

학습 알고리즘을 평가하기 위해 "정상", "비정상"으로 레이블링된 데이터가 필요하다. (정상: y=0, 비정상: y=1)
이 때 데이터 중 "정상"이 매우 많은 비율을 차지한다.  
p(x)를 학습 할 때 "정상"으로 레이블링된 데이터만 사용한다. 그리고 validation, test에서 "비정상"데이터를 섞어서 사용하다. 예를 들어, 데이터 셋 중 0.2%가 비정상으로 레이블링 되었다고 하자. 데이터 셋 중 60%를 train에 사용하고,  "비정상"데이터 0.1%를 포함하여 20%를 validation에 그리고 나머지  "비정상"데이터 0.1%를 포함하여 20%를 test에 사용한다. 즉 데이터를 60/20/20으로 나눠 train/validation/test 에 사용하고, 비정상 데이터를 50/50으로 나눠 validation/test에 사용한다. 

###### Algorithm evaluation

$\lbrace x^{(1)},\dots,x^{(m)} \rbrace$에 대해 모델 p(x)를 피팅한다. validation/test데이터 x에 대해 다음과 같이 추정한다. 
If p(x) < ϵ (anomaly), then y=1
If p(x) ≥ ϵ (normal), then y=0
가능한 평가 시스템

- True positive, false positive, false negative, true negative.
- Precision/recall
- $F_1$ score

cross-validation 셋을 통해 ϵ 값을 결정한다. 

###### Anomaly Detection vs. Supervised Learning

언제 anomaly detection을 사용하고 또 어떤 상황에서 supervised learning 사용할까?
다음과 같은 상황에서 anomaly detection을 사용한다. 

- 학습 데이터 중 매우 적은 수(0-20)가 y=1로 레이블링 되어 있고 나머지 대부분이 y=0인 경우
- "비정상"의 유형이 매우 다양하게 나타날 때, 그래서 미래의 비정상 결과의 형태를 예측하기 어려운 경우, 학습데이터에 "비정상"의 모든 유형이 포함 되어 있지 않음    

다음과 같은 상황에서 supervised learning을 사용한다.

- 학습데이터가 균등하게 레이블링 되어 있는 경우
- 충분히 많은 "비정상"학습 데이터를 가지고 있어 그 형태를 예측 할 수 있는 경우, 일반적으로 "비정상"으로 판별 될 때 그 데이터는 학습데이터와 비슷한 형태를 보임  

###### Choosing What Features to Use

![](http://cfile28.uf.tistory.com/image/257968475866E712165686 "Features" "width:600px;height:300px;float:center;padding-left:10px;")

피쳐는 anomaly detection에서 매우 큰 영향을 끼친다. 
데이터의 히스토그램을 그려 그 모양이 종 형태를 보이는 지를 확인함으로써 피쳐가 가우시안 분포인지를 알 수 있다. 
피쳐가 가우시안 분포를 보이지 않는 경우 몇몇 Transform을 사용한다. 

- log(x)
- log(x+1)
- log(x+c) c: 상수
- $\sqrt{x}$
- $x^{1/3}$

위의 함수들을 통해 피쳐가 가우시안 분포가 되도록 바꿀 수 있다. 
supervised learning애서 사용한 에러 분석 과정과 비슷한 방법을 통해 주어진 모델의 정확도를 검증한다.
anomaly detection 모델 함수의 목적은 정상인 데이터에 대해 p(x)가 큰 값을 가지고, 비정상인 데이터에서는 작은 값을 가지게 하는 것이다. 
일반적으로 발생하는 문제 중 하나는 두 경우 모두 p(x)가 작은 값을 가지는 경우다.  이 경우 정상인 데이터에 대해 더 높은 확률을 가지는 새로운 피쳐를 찾아야 한다. 일반적으로 비정상 데이터에 대해 예외적으로 큰 값 또는 작은 값을 가지는 피쳐를 사용한다.  

### Recommender Systems

##### Problem Formulation
![](http://cfile21.uf.tistory.com/image/2604A24F5866E90906A83D "Recommendations" "width:600px;height:300px;float:center;padding-left:10px;")

추천(Recommendation)은 기계학습에서 현재 가장 활발히 사용되고있다. 

고객에게 영화를 추천한다고 할때, 다음과 같은 정의를 사용한다. 

- $n_u =$ 이용자의 수 
- $n_m =$ 영화의 수
- $r(i,j) = 1$ 이용자 j가 영화 i를 평가한 경우 
- $y(i,j) =$ 이용자 j가 영화 i에 준 점수, $r(i,j) = 1$인 경우에만 정의 됨 

##### Content Based Recommendations

![](http://cfile6.uf.tistory.com/image/217B194F5866E9090A39ED "Recommendations" "width:600px;height:300px;float:center;padding-left:10px;")

영화의 장르에 기반하여 추천한다고 할 때,  영화가 얼마나 로맨스 장르인지, 액션 장르인지를 [0~1]의 범위에서 나타내는 두가지 피쳐 $x_1$, $x_2$를 설정 할 수 있다. 
학습하기위한 한가지 방법은 각 사용자에 대해 선형회귀를 사용하는 것이다. 각 유저 j에 대해 파라미터 $\theta^{(j)} \in \mathbb{R}^3$를 학습한 다음, 영화 i에 대한 유저 j의 별점을 다음과 같이 $(\theta^{(j)})^Tx^{(i)}$ 예측할 수 있다. 

- $\theta^{(j)} =$ 사용자 j의 파라미터 벡터
- $x^{(i)} =$ 영화 i의 피쳐 백터

영화 i에 대한 유저 j의 별점: $(\theta^{(j)})^Tx^{(i)}$ 

- $m^{(j)} =$ 사용자 j가 평가한 영화의 수

다음과 같은 과정을 통해 $\theta^{(j)}$를 학습한다. 

$min_{\theta^{(j)}} = \dfrac{1}{2}\displaystyle \sum_{i:r(i,j)=1} ((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)})^2 + \dfrac{\lambda}{2} \sum_{k=1}^n(\theta_k^{(j)})^2$

선형회귀와 매우 비슷한 형태를 보인다. 첫번째 항은 $r(i,j) = 1$인 경우에 대해 모든 합연산을 수행한다. 다른 모든 사용자에 대해서는 다음과 같이 구한다. 

$min_{\theta^{(1)},\dots,\theta^{(n_u)}} = \dfrac{1}{2}\displaystyle \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} ((\theta^{(j)})^T(x^{(i)}) - y^{(i,j)})^2 + \dfrac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n(\theta_k^{(j)})^2$

간단하게 모든 사용자에 대해 합연산을 수행한다. 위의 식을 비용함수로 삼아 Gradient Descent를 통해 파라미터 값을 업데이트 한다. 
선형회귀와 다른 점은 상수 $\dfrac{1}{m}$을 제거한 것이다. 사실 학습결과에는 큰 상관이 없다.   

##### Collaborative Filtering

![](http://cfile23.uf.tistory.com/image/252E34435866E98F211354 "Collaborative" "width:600px;height:300px;float:center;padding-left:10px;")

하지만 로맨스나 액션의 정도를 나타내는 피쳐를 찾기란 매우 어렵다. 즉 $x^{(i)} =$를 구하기가 매우 어렵다. feature finder를 사용하여 이를 나타낼 수 있다. 
사용자가 다른 장르에 대해 어느 정도 좋아하는 지를 파악하고, 이를 사용하여 선호도를 표현하는 벡터를 얻을 수 있다. 
매개변수로 부터 피쳐를 구하기 위해 위의 방법과 비슷하게, 제곱오차함수와 정규화를 사용해서 식을 세운다. 

$min_{x^{(1)},\dots,x^{(n_m)}} \dfrac{1}{2} \displaystyle \sum_{i=1}^{n_m} \sum_{j:r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2$

피쳐를 예즉하기위해 무작위로 파라미터를 초기화한 다음 위의 피쳐를 구하는 것과 파라미터를 구하는 두 과정을 반복한다. 실제 이러한 방법을 사용하더라도 피쳐는 꽤나 좋은 방향으로 수렴한다. 

##### Collaborative Filtering Algorithm
조금 더 학습 속도를 높이기 위해, 피쳐와 파라미터에 대한 학습을 동시에 수행 할 수 있다. 

$J(x,\theta) = \dfrac{1}{2} \displaystyle \sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)} - y^{(i,j)})^2 + \dfrac{\lambda}{2}\sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 + \dfrac{\lambda}{2}\sum_{j=1}^{n_u} \sum_{k=1}^{n} (\theta_k^{(j)})^2$

식이 매우 복잡하게 보이나, 단지 위에서 소개한 두 비용함수를 더한 것 뿐이다.  
x∈ℝn, θ∈ℝn로 차수가 같기 때문에 바이어스 x0=1을 포함시키지 않고 학습한다.
전체 알고리즘 과정은 다음과 같다. 

1. $x^{(i)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)}$를 무작위 값으로 초기화 한다. 이러한 과정을 통해 대칭성을 깰 수 있고, $x^{(i)},...,x^{(n_m)}$가 서로 다은 다양한 값을 가지게 한다. 
2. Gradient Descent나 다른 최적화 기법을 사용하여 비용함수  $J(x^{(i)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})$가 최소가 되는 피쳐와 파라미터를 학습한다.  ex) 모든 j에 대해 $x_k^{(i)} := x_k^{(i)} - \alpha\left (\displaystyle \sum_{j:r(i,j)=1}{((\theta^{(j)})^T x^{(i)} - y^{(i,j)}) \theta_k^{(j)}} + \lambda x_k^{(i)} \right)$
3. 사용자 파라미터 θ와 영화 피쳐 x를 곱해($\theta^Tx$) 별점을 예측한다. 

##### Vectorization: Low Rank Matrix Factorization

![](http://cfile6.uf.tistory.com/image/24796E4F5866E90A0B3077 "Low Rank Matrix Factorization" "width:600px;height:250px;float:center;padding-left:10px;")

각 영화에 대한 피쳐를 나타내는 행렬 X, 그 피쳐에 대한 파라미터를 나타내는 행렬 Θ가 주어 진다고 하면 모든 영화에대한 사용자의 평가는 다음과 같이 행렬 Y로 표현 할 수 있다. 
$Y = X\Theta^T$
두 영화 i, j가 얼마나 비슷한지를 판별하기 위해 다음과 같이 피쳐간 거리를 측정한다. $||x^{(i)} - x^{(j)}||$ 이 값이 작은 경우 두 영화는 서로 비슷한 특성을 지닌다고 볼 수 있다. 

##### Implementation Detail: Mean Normalization
 
![](http://cfile30.uf.tistory.com/image/2701694F5866E90907F353 "Mean Normalization" "width:600px;height:150px;float:center;padding-left:10px;") 

위에서 소개한 평점 알고리즘을 사용하는 경우, 기존에 영화를 한번도 보지 않는 새로운 사용자에 대해서는 제대로도니 추천을 하기가 어렵다. 특히 최적화 과정을 통해 모든 파라미터 값은 0으로 수렴하고, 모든 영화에 대한 평점이 0이 된다. 당연히 이러한 결과는 잘못된 것이다. 이러한 문제를 평균에 기반한 정규화를 통해 해결할 수 있다. 
i행 영화에 대한 j열 사용자의 평점으로 구성된 Y 행렬을 사용한다. 다음과 같은 평균값 백터를 정의 할 수 있다. 

$\mu = [\mu_1, \mu_2, \dots , \mu_{n_m}]$, $\mu_i = \frac{\sum_{j:r(i,j)=1}{Y_{i,j}}}{\sum_{j}{r(i,j)}}$

이는 i번째 영화에 대한 기존 사용자가 매긴 평점의 평균이다. Y에서 $\mu$를 빼서 Y를 정규화(Y') 한다.
예시: Y와 $\mu$가 아래와 같을 때    

$Y = \begin{bmatrix} 5 & 5 & 0 & 0 \newline 4 & ? & ? & 0 \newline 0 & 0 & 5 & 4 \newline 0 & 0 & 5 & 0 \newline \end{bmatrix}, \quad \mu = \begin{bmatrix} 2.5 \newline 2 \newline 2.25 \newline 1.25 \newline \end{bmatrix}$

정규화(Y')은 아래와 같다 

$Y' = \begin{bmatrix} 2.5 & 2.5 & -2.5 & -2.5 \newline 2 & ? & ? & -2 \newline -.2.25 & -2.25 & 2.75 & 1.75 \newline -1.25 & -1.25 & 3.75 & -1.25 \end{bmatrix}$ 

이러한 정규화를 포함하도록 위 비용함수를 아래와 같이 수정해야 한다.

$(\theta^{(j)})^T x^{(i)} + \mu_i$

따라서 새로운 사용자의 초기 값은 0이 아니라 평균값 $\mu$가 되므로, 조금 더 정확한 예측을 할 수 있다.  