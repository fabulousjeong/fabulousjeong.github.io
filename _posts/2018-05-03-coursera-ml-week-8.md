---
title:  Week 8 Clustering
category: CourseraMachineLearning
excerpt: |
  Unsupervised Learning(자율학습)은 Supervised learning과 달리 레이블링 되어 있지 않는 데이터를 사용한다. 즉 예상 결과를 나타내는 y벡터 없이 x데이터만 사용한다. 클러스터링은 다음과 같은 학습에 유용하다.
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  
  ref: [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning "Coursera ML")
  
  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")
  
feature_image: "../assets/images/coursera_ML/title.png"
image: "./assets/images/coursera_ML/title.png"
---

### ML: Clustering

##### Unsupervised Learning: Introduction

Unsupervised Learning(자율학습)은 Supervised learning과 달리 레이블링 되어 있지 않는 데이터를 사용한다. 즉 예상 결과를 나타내는 y벡터 없이 x데이터만 사용한다. 클러스터링은 다음과 같은 학습에 유용하다.

![](http://cfile8.uf.tistory.com/image/262E4D49585D33062057EB "Unsupervised Learning" "width:600px;height:350px;float:center;padding-left:10px;")

- 시장 세분화
- 소셜 네트워크 분석
- 컴퓨터 클러스터 조직
- 천문데이터 분석

##### K-Means Algorithm

![](http://cfile30.uf.tistory.com/image/24358D49585D330719BB79 "K-Means Algorithm" "width:600px;height:350px;float:center;padding-left:10px;")

출처: http://dendroid.sk/2011/05/09/k-means-clustering/

K-Means 알고리즘은 공통 부분집합으로 데이터를 자동으로 그룹핑하는 가장 유명하고 널리 사용되는 알고리즘이다. 

1. 무작위로 데에터 셋 중 두점을 cluster centroids로 정한다. (a)
2. Cluster assignment: 모든 데이터에 대해 가까운 cluster centroid를 할당한다. (b)
3. Move Centroid: 각각 두 cluster centroid에 대한 그룹 내 데이터들의 평균을 각각 구한다. 구한 평균을 새로운 cluster centroid로 삼는다. (c)
4. 적절한 그룹핑이 될 때까지 (2), (3)의 과정을 반복한다.

주요 파라미터는 다음과 같다. 

- K 클러스터 개수 
- 학습 데이터 셋 ${x^{(1)}, x^{(2)}, \dots,x^{(m)}}$
- 여기서 $x^{(i)} \in \mathbb{R}^n$

여기서는 기존의 x0=1 (바이어스)를 사용하지 않는 것에 주의하자. 

알고리즘 
``` python
Randomly initialize K cluster centroids mu(1), mu(2), ..., mu(K) 

Repeat: 
    for i = 1 to m: 
        c(i):= index (from 1 to K) of cluster centroid closest to x(i) 
    for k = 1 to K: 
        mu(k):= average (mean) of points assigned to cluster k
```

첫번째 반복문은 "클러스터 할당" 단계다. 여기서는 각 데이터 x(i)에 대한 centroid의 인덱스 벡터 c를 만든다. 위의 단계를 좀 더 수학적으로 표현하면 아래와 같다. 

$c^{(i)} = argmin_k\ ||x^{(i)} - \mu_k||^2$

각 $c^{(i)}$는 각 데이터 x(i)의 최소 거리에 위치하는 centroid의 인덱스와 같다. 관습적으로 제곱항을 사용하는데, 제곱 함수의 특징인 큰 기울기는 최적화 과정에서 빠르게 수렴하는데 도움을 준다. 이것이 관습적으로 사용하는 이유다. 또한 유클리디안 거리에서의 제곱근 보다 계산량이 적은 것도 사용하는 이유 중 하나다. 

제곱을 하지 않은 경우:

$$||x^{(i)} - \mu_k|| = ||\quad\sqrt{(x_1^i - \mu_{1(k)})^2 + (x_2^i - \mu_{2(k)})^2 + (x_3^i - \mu_{3(k)})^2 + ...}\quad||$$

제곱을 한 경우: 

$$||x^{(i)} - \mu_k||^2 = ||\quad(x_1^i - \mu_{1(k)})^2 + (x_2^i - \mu_{2(k)})^2 + (x_3^i - \mu_{3(k)})^2 + ...\quad||$$

따라서 제곱을 하는데는 빠르게 수렴하는 것과 계산량을 줄이는 두가지 목적이 있다. 

두번째 반복문은 "Centroid 이동" 단계로, 각 그룹의 평균으로 Centroid 를 이동시킨다. 공식으로 표현하면 아래와 같다. 

$\mu_k = \dfrac{1}{n}[x^{(k_1)} + x^{(k_2)} + \dots + x^{(k_n)}] \in \mathbb{R}^n$

여기서 $x^{(k_1)} + x^{(k_2)} + \dots + x^{(k_n)}$은 각 그룹 $m\mu_k$에 속한 학습 데이터들이다. 

만약 Centroid에 할당된 데이터 셋이 없다면 무작위로 새로운 Centroid의 위치를 정할 수 있다, 혹은 그 Centroid를 삭제 시킬 수도 있다. 

수 차례 반복문 수행 후에 알고리즘이 수렴하면, 그이후 수행되는 반복문은 더 이상 클러스터링에 영향을 주지 않는다.    

아래와 같이 실제 분리, 그룹에 대한 경계가 모호한 경우에도 K-means 알고리즘은 적절한 K개의 그룹으로 분리해준다.   

![](http://cfile24.uf.tistory.com/image/2571F64A585D33E22DB533 "K-Means Algorithm" "width:600px;height:200px;float:center;padding-left:10px;")

##### Optimization Objective

알고리즘에 사용한 몇몇 파라미터를 다시 한 번 살펴보자

- $c^{(i)}$ = x(i)가 속한 클러스터의 인덱스
- $\mu_{k}$ = 클러스터 Centroid k (μk∈ℝn)
- $\mu_{c^{(i)}}$ =  x(i)가 속한 클러스터 Centroid 


위의 변수를 사용하여 비용함수를 아래와 같이 표현 할 수 있다. 

$$J(c^{(i)},\dots,c^{(m)},\mu_1,\dots,\mu_K) = \dfrac{1}{m}\sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$$

위 비용함수를 최소화 하는 파라미터를 찾는것이 최적화 과정의 목적이다. 

$min_{c,\mu}\ J(c,\mu)$

즉, 모든 학습데이터에 대해 평균 거리가 최소가 되는 $\mu$와 $c^{(i)}$ 찾아야한다.

위의 비용함수는 종종 학습 데이터의 Distortion(왜곡)이라 불린다. 

클러스터 할당 단계에서의 목적은 다음과 같다.

Minimize J(...) with $c^{(1)},\dots,c^{(m)}$ (holding $\mu_1,\dots,\mu_K$ fixed)

그리고, Centroid 이동 단계에서의 목적은 아래와 같다. 

Minimize J(...) with $\mu_1,\dots,\mu_K$

K-means에서는 비용함수가 증가하는 일은 발생하지 않는다. 항상 감소한다. 

##### Random Initialization

cluster centroid를 무작위로 지정하는데 아래의 방법을 권한다. 
$K \le m$ 유지, 이는 클러스터의 수는 학습 데이터의 수보다 작음을 의미한다. 
학습 데이터 중 K개를 무작위로 선택한다. 
위에서 선택한 데이터를 $\mu_1,\dots,\mu_K$로 설정한다. 
K-means는 경우에 따라 아래와 같은 local optima 현상이 발생 할 수 있다. 

![](http://cfile10.uf.tistory.com/image/257D434A585D33E3203E21 "Random Initialization" "width:600px;height:350px;float:center;padding-left:10px;")

이러한 일이 발생하지 않도록 하기위해 여러 무작위 시작점에 대해서 K-means를 반복한다. $K\le10$ 인 경우 위의 아래 방법을 적용하길 권한다. 

``` python
for i = 1 to 100:
   randomly initialize k-means
   run k-means to get 'c' and 'm'
   compute the cost function (distortion) J(c,m)
pick the clustering that gave us the lowest cost
```

##### Choosing the Number of Clusters

K를 선택하는 것(총 몇개의 클러스터로 분류할 지)은 어려운 일이다. 
![](http://cfile29.uf.tistory.com/image/217AF24A585D33E32310E0 "Choosing the Number of Clusters" "width:600px;height:200px;float:center;padding-left:10px;")

Elbow method: 클러스터 당 cost J를 나타내는 그래프를 그린다. 비용함수는 클러스터 수가 증가함에 따라 처음에는 가파르게 감소하다 어느 순간 천천히 감소할 것이다. 이경우 천천히 감소하기 시작하는 지점을 K로 선택한다. 하지만 그래프가 점진적으로 감소하는 경우 이러한 변곡점이 명확하지 않은 경우도 있다. 알고리즘이 local optima에 빠지는 경우를 제외하면, K가 증가함에 따라 비용함수 J는 항상 감소한다. 
K를 선택하는 다른 방법은 학습 목적에 부합하는 수 만큼 그룹하는 것이다. 즉 전체 데이터를 10개 클러스터로 분류 하고자 한다면 K=10으로 설정하는 것이 가장 좋다. ex) 옷 분류시 M, L, XL 로 분류한다면 k=3으로 설정.    

### ML: Dimensionality Reduction

##### Motivation I: Data Compression

- 많은 중복되는 데이터가 있을 때, 피쳐의 수를 줄이는 것이 좋다. 
- 이를 위해 두 correlated 피쳐를 뽑고, 그려서 전체 피쳐를 나타내는 새로룬 선을 생성한다. 그리고 모든 피쳐를 이 선 위에 배치한다. 

이러한 방법으로 피쳐의 수를 줄이면 메모리 사용량을 줄일수 있으며, 학습 속도를 높일 수 있다.   

Note: 차원감소(Dimensionality Reduction)은 학습 데이터의 수를 줄이는 것이 아닌 피쳐의 수를 줄이는 것이다. 

##### Motivation II: Visualization

3차원 이상의 피쳐를 가진 데이터를 시각화 하기는 어렵다. 데이터를 그리기 위해 3차원 이하로 피쳐 수를 줄일 수 있다. 
데이터의 다른 피쳐를 잘 요약하여 표현 할 수 있는 새로운 피쳐 $z_1, z_2$(또는 추가로 $z_3$)를 찾아야 한다. 
Example: 한 국가의 경제 시스템에 관한 여러 피쳐를 "경제 활동"이라는 하나의 피쳐로 표현 할 수 있다. 

### Principal Component Analysis Problem Formulation

가장 유명한 차원 감소 알고리즘은 "Principal Component Analysis (PCA)"다. 

##### Problem formulation

두 피쳐($x_1, x_2$)가 있을 때, 이를 효과적으로 표현하는 새로운 선을 찾아야 한다. 즉 이전 피쳐를 새로운 라인에 대응하는 피쳐 위로 배치해야한다. 피쳐를 세개에서 두개로 줄이는 경우에 대해서도 면 위로 배치하는 것을 통해 위와 동일하게 적용할 수 있다.
PCA의 목적은 선 위로 투사(Projection)되는 피쳐의 위치와 이전 피쳐간의 평균 거리를 최소화 하는 것이다. 이는 Projection error라 한다. 

2D에서 1D로 차원 감소: 데이터가 선 위로 투사될 때 Projection error를 최소화 하는 방향($u^{(1)} \in \mathbb{R}^n$)을 찾는다. 
더 일반적인 경우에 대해서는 아래와 같다. 

n-D에서 k-D로 차원 감소: Projection error를 최소화 하는 k개 벡터($u^{(1)}, u^{(2)}, \dots, u^{(k)}$)을 찾는다. 

![](http://cfile1.uf.tistory.com/image/256C6D4B585D39C5333BF0 "pca" "width:600px;height:300px;float:center;padding-left:10px;")

![](http://cfile23.uf.tistory.com/image/2618514B585D39C5085C77 "pca" "width:600px;height:250px;float:center;padding-left:10px;")

##### PCA for linear regression

- 왼쪽그림과 같이 선형회귀에서는 예측함수의 선과 데이터 사이의 제곱오차를 최소화했다. 이 때 오차를 표현함에 있어 수직거리(y값의 차이)를 사용하였다.
- 반면 PCA에서는 데이터의 최단거리(최단 직교거리)를 최소화 한다.

 보다 일반적으로 선형 회귀에서는 학습데이터 x와 파리미터 $\Theta$를 통해 결과값 y를 추정하였다. PCA에서는 여러 피쳐 $x_1, x_2, \dots, x_n$에 대해 이를 가장 잘 나타내는 공통 피쳐를 찾는다. 여기서는 결과에 대한 예측이 없으며, 피쳐에 파라미터 $\Theta$를 적용하지도 않는다. 
 
### Principal Component Analysis Algorithm

PCA 알고리즘은 실행하기 전에 몇가지 전처리 과정이 있다.

##### Data preprocessing

- 학습 데이터 셋:  $x(1), x(2), \dots, x(m)$
- 전처리(feature scaling/mean normalization) $\mu_j = \dfrac{1}{m}\sum^m_{i=1}x_j^{(i)}$
- 각 $x_j^{(i)}$를 $x_j^{(i)-\mu_j}$로 바꿈
- 각 피쳐의 범위가 다를 때(ex: $x_1$=집의 크기, $x_2$=방의 수), 피쳐 스케일링은 값의 범위를 비슷하게 해준다. 먼저, 평균을 뺀 다음, 피쳐의 범위 크기 만큼 나눠 준다.  $x_j^{(i)} = \dfrac{x_j^{(i)} - \mu_j}{s_j}$

다음의 식을 통해 2D에서 1D로 줄어드는 것에 대한 의미를 알 수 있다. 

$\Sigma = \dfrac{1}{m}\sum^m_{i=1}(x^{(i)})(x^{(i)})^T$

z 값은 모두 실수이며, $u^{(1)}$위에 투영된 피쳐 값이다. 

PCA에서는 두 가지 작업을 한다. 먼저 $u^{(1)},\dots,u^{(k)}$를 구하고, 투영 값 $z_1, z_2, \dots, z_m$를 구한다. 

###### Compute "covariance matrix"

$\Sigma = \dfrac{1}{m}\sum^m_{i=1}(x^{(i)})(x^{(i)})^T$
Octave프로그램에서는 벡터화해서 아래과 같이 구한다. 
``` python
Sigma = (1/m) * X' * X;
```
 위의 공분산행렬을 Sigma로 나타낸다.(이 기호는 합을 뜻하기도 하지만 여기서는 다른 의미로 사용된다.) $x^{(i)}$는 n x 1 벡터며, $(x^{(i)})^T$는 1 x n 벡터다. 이 두 벡터의 곱은 n x n 행렬이 되며 이는 $\Sigma$의 크기가 된다. 

###### Compute "Eigenvectors" of covariance matrix $\Sigma$

``` python
[U,S,V] = svd(Sigma);
```
 svd는 Octave에 내장된 'singular value decomposition'함수다. 여기서 PCA에 사용하는 값은 U 행렬로 이는 $u^{(1)},\dots,u^{(n)}$를 나타낸다. 이것의 위에서 구하려고 했던 벡터와 일치한다. 


###### Take the first k columns of the U matrix and compute z

U에서 위에서 부터 k개 만큼 열을 뽑는 것을 U reduce라 한다. 그 결과 n x k 크기의 행렬을 얻는다. 이를 사용하여 z를 구한다. 
$z^{(i)} = Ureduce^T \cdot x^{(i)}$
$Ureduce^T$는 k x n 차원이고 x(i)는 n x 1 차원이다. 따라서 $Ureduce^T \cdot x^{(i)}$는 k x 1 차원이다. 즉 n에서 k로 차원이 줄었다!! Octave 코드를 정리하면 아래와 같다.
``` python
Sigma = (1/m) * X' * X; % compute the covariance matrix
[U,S,V] = svd(Sigma);   % compute our projected directions
Ureduce = U(:,1:k);     % take the first k directions
Z = X * Ureduce;        % compute the projected data points
```

![](http://cfile26.uf.tistory.com/image/2501C14B585D39C61D9C23 "pca" "width:600px;height:350px;float:center;padding-left:10px;")

##### Reconstruction from Compress Representation

![](http://cfile2.uf.tistory.com/image/25489F49585D3A3C2872B5 "pca" "width:600px;height:350px;float:center;padding-left:10px;")

PCA 사용 시, 예전 데이터 즉 PCA 적용 이전의 피쳐 개수로 복원하려면 어떻게 해야 할까? 1D에서 2D로 다시 복원: $z \in \mathbb{R} \rightarrow x \in \mathbb{R}^2$

식으로 나타내면 다음과 같다: $x_{approx}^{(1)} = U_{reduce} \cdot z^{(1)}$

완전한 복원이 아닌 오른쪽과 같은 대략적인 복원임에 주의하자. 

Note: U 행렬에는 Unitary 행렬에서 나타나는 특별한 특성이 있다. 한 가지 특징은 아래와 같다. 

"*"가 conjugate transpose일 때 $U^{-1} = U^∗$

U에 속한 값은 모두 실수이므로 이는 아래와 같이 나타낼 수 있다. 

$U^{-1} = U^T$, 즉 $U^{-1}$ 계산하기 위해 시간을 쓸 필요가 없다.

### Choose the Number of Principal components

주성분(principal components)의 수 k를 어떻게 결정할까? k가 줄이려고 하는 차원임을 상기해보자. 
k를 결정하는 한가지 방법은 아래 공식을 사용하는 것이다. 

- projection error의 제곱 평균이 주어짐: 
$$\dfrac{1}{m}\sum^m_{i=1}||x^{(i)} - x_{approx}^{(i)}||^2$$
- 전체 데이터의 variation을 구함: 
$$\dfrac{1}{m}\sum^m_{i=1}||x^{(i)}||^2$$
- 다음 값을 만족하는 k를 선택: 
$$\dfrac{\dfrac{1}{m}\sum^m_{i=1}||x^{(i)} - x_{approx}^{(i)}||^2}{\dfrac{1}{m}\sum^m_{i=1}||x^{(i)}||^2} \leq 0.01$$
 
오차를 분산으로 나눈 값이 1%이하이므로 99%의 데이터가 유지 된다고 볼 수 있다. 

##### Algorithm for choosing k

1. Try PCA with k=1,2,...
2. Compute $U_{reduce}, z, x$
3. 위의 식을 적용 99%가 유지 되는지 확인 하고, 그렇지 않다면 k를 증가 시킨다. 

하지만 위의 과정은 매우 비효율적이다. Octave에서는 보다 간단하게 구할 수 있다. 
``` python
[U,S,V] = svd(Sigma)
```
위에서 행렬 S를 얻을 수 있는데, S를 사용해서 값이 99%보장 되는지 판단 할 수 있다. 

$\dfrac{\sum_{i=1}^kS_{ii}}{\sum_{i=1}^nS_{ii}} \geq 0.99$

##### Advice for Applying PCA
PCA의 주된 사용처는 supervised learning에서 학습 속도를 높이는 것이다. 
엄청나게 많은 피쳐를 가지고 있는 데이터가 있다고 할 때($x^{(1)},\dots,x^{(m)} \in \mathbb{R}^{10000}$), PCA를 사용해서 피쳐 수를 줄일 수 있다($x^{(1)},\dots,x^{(m)} \in \mathbb{R}^{1000}$).
PCA를 사용하여 차원감소를 할 때 학습 데이터만 사용함을 주의하자. Cross-validation, 테스트 데이터는 위에서 구한 U를 사용하여 z(i)를 맵핑한다.

사용 분야

- Compression: 데이터 저장 공간 축소. 학습 속도 향상
- 데이터 시각화: k를 2 또는 3으로 설정

PCA 사용의 안 좋은 예: Overfitting을 해결하기위해 사용. PCA를 사용하여 피쳐 수를 줄일 수 있기 때문에 Overfitting을 해결하는데 얼핏 도움이 되는것 처럼 보인다. 하지만 결과 값 y에 대한 고려가 전혀 없으므로 그리 추천하는 방법이 되지 않는다. 차라리 정규화를 사용하는 것이 더 효과적이다.
항상 PCA를 사용할 필요는 없다. 먼저 PCA없이 학습 알고리즘을 적용한 다음 필요하다면 PCA를 사용하자. 

 

 