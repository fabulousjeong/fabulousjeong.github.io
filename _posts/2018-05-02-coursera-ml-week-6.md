---
title: Week 6 Machine Learning Deciding
category: CourseraMachineLearning
excerpt: |
  다음과 같은 방법으로 예측에서 생긴 오류를 해결할 수 있다. 학습 데이터를 더 많이 사용하기, 변수의 수를 줄이기 
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  
  ref: [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning "Coursera ML")
  
  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")
  
feature_image: "../assets/images/coursera_ML/title.png"
image: "./assets/images/coursera_ML/title.png"
---

### ML: Advance for Applying Machine Learning Deciding What to Try Next

다음과 같은 방법으로 예측에서 생긴 오류를 해결할 수 있다. 

- 학습 데이터를 더 많이 사용하기
- 변수의 수를 줄이기 
- 변수의 수를 늘리기 
- 다항($x^2, x^3 ...$) 변수를 사용하기
- $\lambda$의 값을 키우거나 줄이기

무작위로 위의 방법을 적용해서는 안된다. 아래에서 위의 방법을 적용하기 위한 여러 기법에 대해 설명한다. 

##### Evaluating a Hypothesis
![](http://cfile8.uf.tistory.com/image/2773AD3B586CF2A30C8CA1 "Evaluating a Hypothesis" "width:600px;height:200px;float:center;padding-left:10px;")

학습 데이터에 대해 가설함수의 오차가 매우 작더라도, 과적합(overfitting)에 의해 그 결과가 여전히 정확하지 않을 수 있다. test셋을 사용하여 과적합이 발생했는지 확인 할 수 있다. 학습 데이터 셋이 주어졌을 때, 이 데이터 셋을 training set과 test set 두 세트로 나눌 수 있다. 두 set을 사용하는 학습 방법은 다음과 같다.

1. training set을 사용하여 $J_{train}(\Theta)$를 최소화 하는 $\Theta$를 계산
2. test set 오차를 계산 $J_{test}(\Theta)$

##### The test set error

1. 선형회귀의 경우 
$J_{test}(\Theta) = \dfrac{1}{2m_{test}} \sum_{i=1}^{m_{test}}(h_\Theta(x^{(i)}_{test}) - y^{(i)}_{test})^2$
2. 분류/미분류에 대해 
$err(h_\Theta(x),y) = \begin{matrix} 1 & \mbox{if } h_\Theta(x) \geq 0.5\ and\ y = 0\ or\ h_\Theta(x) < 0.5\ and\ y = 1\newline 0 & \mbox otherwise \end{matrix}$

위의 식은 분류/미분류에 기반한 0 또는 1의 이진 결과를 출력한다. test error의 평균은 아래와 같다. 

$$\text{Test Error} = \dfrac{1}{m_{test}} \sum^{m_{test}}_{i=1} err(h_\Theta(x^{(i)}_{test}), y^{(i)}_{test})$$

위의 식은 test데이터에서 미분류의 비율을 나타낸다. 

### Model Selection and Train/Validation/Test Sets

training set에서만 잘 동작하는 알고리즘은 좋은 가설함수라고 볼 수 없다.
가설함수의 오차는 일반적으로 training set이 다른 데이터 셋보다 더 작다.
가설함수의 모델을 선택하기 위해, 여러 차수의 다항식에 대해 오차가 어느 정도인지 살펴봐야 한다.

##### Without the Validation Set(note: this is a bad method - do not use it)

2. 1. "training set"을 사용하여 각각 여러 차수의 가설함수에대해 $\Theta$ 값을 최적화 한다. 
2. "test set"을 사용하여 최적 다항 차수 d를 찾는다. 
3. 위에서 찾은 d차수의 가설함수 $J_{test}(\Theta^{(d)})$를 사용하여 오차를 구한다. 

그런데 위에서 d를 찾을 때 test 데이터를 사용하였다. 따라서 실제 모델을 사용할 경우 오차가 더 클 수 있다.

##### Use of CV set

이러한 문제를 해결하기위해 Cross Validation Set이라 부르는 세번째 데이터 셋을 도입한다. Cross Validation Set을 사용하여 d를 구한다. 따라서 test set은 더 정확한 결과를 줄 것이다. 
아래의 예는 데이터 셋을 셋으로 나누는 방법 중 하나를 설명한다. 

- Training Set: 60%
- Cross Validation Set: 20%
- Test Set: 20%

이제 3가지 데이터 셋을 사용하여 3가지 오차를 계산 할 수 있다.

##### With the Validation Set (note: this method presumes we do not also use the CV set for regularization)
1. "training set"을 사용하여 각각 여러 차수의 가설함수에대해 $\Theta$ 값을 최적화 한다. 
2. "validation set"을 사용하여 최적 다항 차수 d를 찾는다. 
3. 위에서 찾은 d차수의 가설함수 $J_{test}(\Theta^{(d)})$를 사용하여 오차를 구한다. 

여기서는 d를 구할 때 test set을 사용하지 않는다.

### Diagnosing Bias vs. Variance
![](http://cfile6.uf.tistory.com/image/213A7439586CF3502100F1 "Diagnosing Bias vs. Variance" "width:600px;height:200px;float:center;padding-left:10px;")

이번 장에서는 가설함수에서 다항식의 차수 d와 underfitting, overfitting 사이의 관계에 대해 알아보자. 
- Bias나 variance가 안 좋은 예측에 어떤 영향을 미치는지 구분할 필요가 있다. 
- 높은 Bias는 underfitting을, 높은 variance는 overfitting을 유발한다. 둘 사이에서 촤적의 값을 찾아야 한다. 

Training error는 차수 d가 커질수록 줄어드는 경향이 있다. 반면, cross validation error는 d가 커질수록 줄어들다가 특정 시점 이후에는 다시 증가하는 convex 곡선 형태를 띈다.   

High Bias(underfitting): $J_{train}(\Theta)$, $J_{CV}(\Theta)$ 둘 다 높은 값, 이 때 $J_{CV}(\Theta) \approx J_{train}(\Theta)$

High Variance(overfitting): $J_{train}(\Theta)$는 낮으나 $J_{CV}(\Theta)$는 높은 값을 가짐

아래와 같이 나타낼 수 있다.  

![](http://cfile8.uf.tistory.com/image/2110323A586CF33D090552 "cross validation error" "width:600px;height:200px;float:center;padding-left:10px;")

### Regularization and Bias/Variance

차수 d가 Bias/Variance에 어떻게 영향을 미치는지 살펴보았다. 이번 장에서는 정규화 파라미터 $\lambda$가 Bias/Variance에 어떤 영향을 주는지 살펴보자.

- Large $\lambda$: High Bias(underfitting)
- Intermediate $\lambda$: Just right
- Small $\lambda$: High variance(overfitting)

큰 $lambda$ 값은 $\Theta$를 굉장히 작은 값을 가지게 만들며, 따라서 결과 함수의 선을 단순하게 만들어 underfitting의 원인이 된다. 
$\lambda$와 Bias/Variance 사이의 관계는 아래와 같다.

Low $\lambda$: $J_{train}(\Theta)$는 낮으나 $J_{CV}(\Theta)$는 높은 값을 가짐 High variance(overfitting)

Intermediate $\lambda$: $J_{train}(\Theta)$와 $J_{CV}(\Theta)$가 적절히 작은 값을 가짐

Small $\lambda$: $J_{train}(\Theta)$, $J_{CV}(\Theta)$ 둘 다 높은 값(HighBias/underfitting)

아래 그림은 위에서 $\lambda$와 가설함수 사이의 관계를 표현한다. 

![](http://cfile30.uf.tistory.com/image/235AB433586CF36F032D26 "Regularization" "width:600px;height:200px;float:center;padding-left:10px;")

적절한 $\lambda$를 선택하는 방법은 아래와 같다.

1. $\lambda$의 리스트를 작성한다. (λ∈{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24})
2. 계산하기위해 $\lambda$를 선택한다. 
3. 가설함수의 차수와 같은 학습모델을 설정한다. 
4. $\Theta$를 학습하기위한 모델을 선택한다. 
5. $J_{train}(\Theta)$와 $\lambda$를 최적화 하는 $\Theta$를 찾는다.
6. $\lambda=0$인 경우에 대해 train error를 구한다. 
7. $\lambda=0$인 경우에 대해 validation error를 구한다. 
8. 전체 모델과 $\lambda$에 대해 가장 작은 train error, validation error를 가지는 조합을 찾는다. 
9. 시각적 도움이 필요할 경우 위의 두 그래프를 그린다. 
10. 최적의 호한에 해당하는 $\lambda$, $\Theta$를 사용하여 $J_{test}(\Theta)$를 구해 제대로 학습되었는지 확인한다. 
11. 다음 장에서 다룰 학습곡선을 통해 가장 적절한 차수 d와 $\lambda$를 선택할 수 있다.

##### Learning Curves

학습 데이터 셋이 3개인 경우 항상 0의 오차를 가지는(데이터 셋의 3점을 모두 지나는) 3차 함수를 만들 수 있다. 
학습 데이터 셋이 커질 경우, 3차 함수에서의 오차는 커진다. 특정 학습 데이터 사이즈 보다 큰 경우 오차는 점점 수렴한다. 

##### With high bias

Low training set size: $J_{train}(\Theta)$는 낮고 $J_{CV}(\Theta)$는 높은 값을 가지게 함

Large training set size: $J_{train}(\Theta)$, $J_{CV}(\Theta)$ 둘 다 높은 값, 이 때 $J_{CV}(\Theta) \approx J_{train}(\Theta)$ 

High bias의 경우 학습데이터 셋을 늘리는 것은 크게 도움 되지 않는다.  High variance의 경우 학습 데이터 셋의 크기와 관련이 있다.

![](http://cfile30.uf.tistory.com/image/235AB433586CF36F032D26 "high bias" "width:600px;height:200px;float:center;padding-left:10px;")

##### With high variance

Low training set size: $J_{train}(\Theta)$는 작아 지며 $J_{CV}(\Theta)$는 높은 값을 가짐

Large training set size: $J_{train}(\Theta)$은 데이터 셋이 커질때 마다 증가하며 $J_{CV}(\Theta)$은 수렴하기 전까지 점점 작아진다. $J_{train}(\Theta)$<$J_{CV}(\Theta)$이지만 둘 사이의 오차가 점점 줄어든다는 점은 여전히 중요하다. High variance의 경우 학습데이터 셋을 늘리는 것은 도움이 될 수 있다.

![](http://cfile29.uf.tistory.com/image/21175033586CF3BB307839 "high variance" "width:600px;height:300px;float:center;padding-left:10px;")


### Deciding What to Do Next Revisited

어떤 모델을 사용할지는 아래와 같이 세분화 할 수 있다. 

- 학습데이터 늘리기 - High Variance 수정
- 피쳐 수 감소 - High Variance 수정
- 피쳐 수 추가 - High Bias 수정
- 피쳐 차수 증가 - High Bias 수정
- $\lambda$ 감소 - High Bias 수정
- $\lambda$ 증가 - High Variance 수정

##### Diagnosing Neural Networks

![](http://cfile3.uf.tistory.com/image/227AA535586CF3FD1B3B69 "Diagnosing Neural Networks" "width:600px;height:300px;float:center;padding-left:10px;")

- 작은 수의 파라미터를 가지는 신경망은 언더피팅이 되기 쉽다. 또한 계산량이 적다. 
- 많은 수의 파라미터를 가지는 경우 오버피팅 되는 경향이 있다. 또한 계산량이 많다. 이 경우 정규화 항을 사용($\lambda$ 증가)하여 오버피팅을 해결할 수 있다. 

 하나의 히든 레이어를 가지는 신경망으로 출발하는 것을 추천한다. 교차검증(Cross Validation) 셋을 사용하여 여러 개의 신경망에 대해 점차 학습 할 수 있다. 
 
### Model Selection

적절한 가설함수의 차수 M을 선택한 다음, 모델의 $\Theta$ 파라미터를 어떻게 구할 수 있을까? 이를 해결하기 위한 여러 방법이 있다. 

- 더 많은 학습 데이터 구하기(어려움)
- 오버피팅 없이 주어지 데이터에 맞는 모델 구하기(어려움)
- 정규화를 통한 오버피팅 감소

##### Bais : 근사 오차 (기대 값과 최적 값의 차이) 

- High Bias = UnderFitting (BU)
- $J_{train}(\Theta)$와 $J_{CV}(\Theta)$ 둘 다 큰 값을 가지고  $J_{train}(\Theta)$≈$J_{CV}(\Theta)$

##### Variance: 한정적 데이터로 인한 추정 오차 

- High Variance = OverFitting (VO)
- $J_{train}(\Theta)$은 매우 작은 값을 가짐, $J_{CV}(\Theta)$≫$J_{train}(\Theta)$

##### 직관적인 Bias-Variance 트레이드 오프

- 복잡한 모델 => 데이터에 민감함 => 데이터 X의 변화에 영향을 많이 받음 => High Variance, Low Bias
- 단순한 모델 => 덜 민감함 => 데이터 X의 변화에 따른 값의 차이가 적음 => Low Variance, High Bias

학습에서 가장 중요한 목표중 하나: Bias-Variance 트레이드 오프에 따른 적합한 모델 찾기

##### 정규화 효과

- 작은 값의 $\lambda$는 모델이 작은 노이즈를 미세하게 조정되어 High Variance를 유발함 => OverFitting
- 큰 값의 $\lambda$는 파라미터($\Theta$)를 0에 가까운 값으로 만들어 High Bais를 유발함 => UnderFititng

##### 모델 복잡도 효과

- 작은 차수의 모델은(복잡도가 작은 모델) Low Variance, High Bias 특징을 띈다. 이경우 단조로운 모델이 생성됨.
- 높은 차수의 모델은(복잡도가 큰 모델) 학습데이터에 대해서는 정확도가 높지만 테스트 데이터에는 잘 맞지 않음. 이 경우 학습데이터에 대해서는 Low Bias특징을 보이나 High Variance 특징이 있음

실제 선택하는 모델은 위 둘 사이의 모델이 됨, 그 모델은 일반적인 테스트 데이터에 대해서도 높은 정확도를 보이며, 학습데이터에도 잘 맞음

##### 학습 모델을 선택하는 경험론적 방법

- 많은 학습데이터를 사용하는 것은 High Variance에는 좋으나 High Bias 해결에는 큰 도움이 되지 않는다. 
- 적은 양의 피쳐를 사용하는 것은 High Variance에는 좋으나 High Bias 해결에는 큰 도움이 되지 않는다. 
- 피쳐 수를 추가 하는 것은 High Bias에는 좋으나 High Variance 해결에는 큰 도움이 되지 않는다. 
- 다항식의 차수를 늘리거나 인터렉션 피쳐를 추가하는 것은 High Bias에는 좋으나 High Variance 해결에는 큰 도움이 되지 않는다. 
- Gradient Descent를 사용할 때, $\lambda$를 감소시켜 High Bias를 고칠수 있고, $\lambda$를 증가시켜 High Variance를 해결할 수 있다. 
- 신경망을 사용하는 경우, 작은 크기의 신경망은 언더피팅을 유발하여, 큰 사이즈의 신경망은 오버피팅을 유발한다. 적절한 크기를 찾기 위해 네트워크 크기에 대한 교차검증을 수행한다. 
- ※ 딥러닝은 매우 큰 사이즈의 신경망이며 오버피팅(High Variance)이 발생하나, 엄청난 수의 학습데이터로 이를 해결한다.   

### ML: Machine Learning System Design

##### Prioritizing What to Work On

머신러닝 문제를 해결할 몇 가지 방법이 있다.

- 많은 데이터 수집(ex: honeypot 프로젝트)
- 정교한 피쳐 학습(ex: 스팸메일에서 이메일 헤더 정보 활용)
- 다양한 방법으로 학습 데이터에 대한 알고리즘 학습(스팸에서 맞춤법 검사)

어떠한 옵션이 도움이 되는지 판단하는 것은 여전히 어렵다. 

##### Error Analysis

머신러닝 문제를 해결하기 위해 아래의 접근법을 추천한다. 

- 간단한 알고리즘으로 시작해서 빨리 구현한 다음, 일단 테스트 해보기  
- 러닝커브를 그려보고, 데이터를 더 수집할지 피쳐 수를 늘릴지 판단 해보기
- 에러분석: 교차점증에서의 에러를 직접 수집한 다음 경향성을 파악

오차의 수치적인 값을 구하는 것이 중요하다. 그렇지 않으면 알고리즘의 성능을 판단하기가 어렵다. 입력값을 사용하기 전에 프리프로세싱 해야할 필요가 있다. 예를 들어 단어을 입력으로 사용한다면, 한 가지 단어에 대한 여러 형(fail/failing/failed)을 형태소 분석기 같은 프로그램을 사용하여 한 가지 형으로 바꿀 필요가 있다.    

##### Error Metrics for Skewed Classes

때론, 오류가 감소하는 것이 실제 알고리즘의 성능이 좋아지는 것이라 보기 어려운 경우도 있다. 

- Ex: 암 진단시 전체 중 0.5%가 암을 가지고 있다고 예측 될 때, 학습 알고리즘이 1% 오차가 나왔다. 이 경우 암환자가 0이라고 가정하면 오차는 0.5%가 된다. 오차는 더 작지만 이것이 더 좋은 알고리즘이 아니라는 것은 자명하다. 

이렇듯 전체 학습 데이터 셋에서 특정 클래스가 매우 적은 경우를 Skewed classes라 한다. 또는 특정 한 클래스가 매우 많은 경우도 이에 해당한다. 

이러한 경우 Precision/Recall 기법을 사용한다.  

- Predicted: 1, Actual: 1 --- True positive
- Predicted: 0, Actual: 0 --- True negative
- Predicted: 0, Actual, 1 --- False negative
- Predicted: 1, Actual: 0 --- False positive

Precision: 암 환자라 진단한 모든 환자(y=1) 중 실제 암 환자의 비율

$\dfrac{\text{True Positives}}{\text{Total number of predicted positives}} = \dfrac{\text{True Positives}}{\text{True Positives}+\text{False positives}}$

Recall: 실제 암환자 중 암환자라고 진단한 환자의 비율

$\dfrac{\text{True Positives}}{\text{Total number of actual positives}}= \dfrac{\text{True Positives}}{\text{True Positives}+\text{False negatives}}$

이 두 기준을 활용하면 우리가 세운 분류기(알고리즘)이 얼마나 잘 동작하는지 측정하라 수 있다. 이 두 기준 모두 높아야 한다. 

한 가지 예로 모든 환자에 대해 암이 걸리지 않았다고(y=0) 판단하면, recall은 $\dfrac{0}{0 + f} = 0$이므로, 오차가 작다고 하더라고, 이것이 안 좋은 결과라는 것을 판단 할 수 있다. 

![](http://cfile2.uf.tistory.com/image/21773935586CF52F239660 "Precision/Recall" "width:600px;height:300px;float:center;padding-left:10px;")

주의 1: 모두 0이라고 판단하는 알고리즘의 경우 Precision은 분모가 0으로 나눠지므로 정의 되지 않는다. 

주의 2: Precision/Recall을 수동으로 계산 할 시 실수를 하기가 쉽다. 엑셀과 같은 프로그램을 사용하기를 권한다. 

##### Trading Off Precision and Recall

로지스틱 회귀를 사용해서 두가지 클래스에 대한 확실한 예측을 해야 할 경우가 있다. 한 가지 방법은 문턱값을 높히는 것이다.

- Predict 1 if: $h_\theta(x) \geq 0.7$
- Predict 0 if: $h_\theta(x) < 0.7$ 

이 경우 70%이상의 확률일 때 암이라고 판단한다. 
이러한 경우 Precision값은 높고, Recall은 낮다. 
반면 다음과 같이 문턱 값이 낮은 경우도 있다.

- Predict 1 if: $h_\theta(x) \geq 0.3$
- Predict 0 if: $h_\theta(x) < 0.3$ 

이러한 경우 안전한 예측 결과를 얻을 수 있으며, Recall 값은 높고, Precision 값은 낮다. 문턱 값이 클 수록 Precision은 높아지며 Recall은 작아진다. 문턱 값이 작아질 수록 Precision은 작아지고 Recall은 커진다. 두 판단기준을 하나로 합치기 위해 F 값을 도입한다. 
한 가지 방법으로 평균값을 사용 할 수 있다. 

$\dfrac{P+R}{2}$

하지만 이렇게 사용 할 경우 잘 동작하지 않는다. y=0이라고 판단 할 경우 R이 0임에도 불구하고 높은 값을 가지기 때문이다. 또한 모두 1이라고 판단 할 경우 P는 0이지만, 매우 큰 R값을 가지게 된다. 

F 값을 계산하는 더 나은 방법은 아래와 같다. 

$\text{F Score} = 2\dfrac{PR}{P + R}$

이 경우 F score가 큰 값을 가지려면 P와 R 모두 커야한다.

Test 데이터에 편향되지 않도록 F score 역시 Validation Set을 통해 계산한다. 

### Data for Machine Learning

![](http://cfile2.uf.tistory.com/image/22235734586CF5B12CDB14 "Data for Machine Learning" "width:600px;height:300px;float:center;padding-left:10px;")

학습하기 위해 얼마나 많은 데이터가 필요할까? 그리 좋지 않은 알고리즘이라도 충분한 데이터가 주어진다면, 데이터가 부족한 우수한 알고리즘 보다 더 나을 결과를 보일 수 있다. 빅데이터의 이론적 해석: 바이어스가 낮은 알고리즘(피쳐의 수가 많고, 히든 레이어가 많은 복잡한 알고리즘)의 경우, 사용하는 학습 데이터가 많을 수록 오버피팅이 줄어 들어 테스트 셋의 정확도가 올라간다. 