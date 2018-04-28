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
  
feature_image: "https://cdn.periscopix.co.uk/blog/_800xAUTO_crop_center-center_80/Robot-graduating.png"
image: "https://cdn.periscopix.co.uk/blog/_800xAUTO_crop_center-center_80/Robot-graduating.png"
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
































