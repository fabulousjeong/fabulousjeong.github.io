---
title: Rules of Machine Learning [Part1]
category: RulesOfMachineLearning
excerpt: |
  크게 4장으로 구성되어있으며, 머신러닝을 활용한 프로젝트 개발시 주의점 위주로 설명하고 있습니다. 


feature_text: |
  ## Martin Zinkevich - Rules of Machine Learning
  Google 개발자인 Martin Zinkevich가 정리한 문서를 소개

  ref: [http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf "martin")

feature_image: "https://picsum.photos/2560/600/?image=1073"
image: "https://picsum.photos/2560/600/?image=1073"
comment: true
---


Google 개발자인 Martin Zinkevich가 정리한 문서를 소개하겠습니다.  

크게 4장으로 구성되어있으며, 머신러닝을 활용한 프로젝트 개발시 주의점 위주로 설명하고 있습니다. 

각 장은 다음과 같습니다. Before Machine Learning, Your First Pipeline, Feature Engineering, Slowed Growth, Optimized Refinement, and Complex Model입니다. 본 포스팅에서는 Before Machine Learning, Your First Pipeline을 먼저 살펴 보겠습니다. 

#### 용어정리 

머신러닝에서 자주 등장하는 용어를 아래에 정리 했습니다.  

Instance: 예측하려 하는 것, 예를 들어 "고양이에 관한 것" 이거나 "고양이와 관련이 없는 것"으로 분류되는 웹페이지.

Label: 머신 러닝에 의해 생성 되었거나, 학습 데이터로 부터 제공된 예측에 대한 답, 예를들어 웹페이지의 Label은 "고양이에 관한 것"이 될 수 있음 

Feature: 인스턴스의 속성, 예를 들어 웹페이지에 "고양이"라는 단어가 포함 되어 있는지 여부

Feature Column : 연관된 피쳐의 집합

Example: (Feature가 포함된) Instance 와 label.

Model: 예측 작업의 통계적 표현, 모델을 학습 시킨 다음, 그 모델을 사용하여 예측함

Metric: 관심을 가지고 있는 수, 직접적으로 최적화 되거나 그렇지 않을 수 있음. 가령 정답과의 차이(Loss)를 예로 들 수 있음. 

Objective: 알고리즘에서 최적화 하려는 Metric.

Pipeline: 머신 러닝 알고리즘을 둘러싼 기반 구조(infrastructure), 데이터 수집 및 학습 데이터 생성, 하나이상의 모델 학습, 모델 Exporting



#### Overview

좋은 결과를 내려면?
머신러닝 전문가 보다는, 엔지니어적 관점에서 머신러닝을 다뤄라. 
실제 맞닥뜨릴 문제 중 대부분은 엔지니어링 문제다. 머신 러닝 전문가의 자원이 있더라고, 좋은 결과를 유발 하는 요소의 대부분은 좋은 머신러닝 알고리즘 보다는, 좋은 Feature에서 나온다. 따라서 기본 접근법은 아래와 같다. 
1. 파이프라인이 확실한 앤드 투 앤드(end to end)시스템을 이루는지 확인 할 것
2. 합리적인 Objective를 정하는 것으로 부터 시작 할 것
3. 간단한 방법을 통해 상식적인 Feature를 추가 할 것
4. 파이프 라인이 견고하게 유지 되는지 확인 할 것
위 접근법을 통해 오랜 시간동안 돈을 많이 벌 수 있으며, 많은 사람들을 행복하게 해 줄 것이다. 더 이상 진척이 없는 경우에만 위 접근 방법을 벗어나도 된다. 복잡성을 추가하면 릴리즈가 더 느려진다. 간단한 방법을 먼저 적용해 본 다음, 최신 머신러닝 기법을 사용하길 권한다. 

#### Before Machine Learning

##### Rule #1: Don’t be afraid to launch a product without machine learning.

Rule #1: 머신러닝 없이 제품(서비스)을 출시하는 것을 겁내지 마라.

 머신러닝은 멋지지만 데이터가 필요하다. 이론적으로는 다른 문제를 풀기위해 사용되는 데이터를 가져와서, 당신이 개발할 제품을 약간 수정 할 수는 있다. 하지만 기본적인 추론(heuristics) 정도의 성능만도 못 할 수 있다. 기계학습을 통해 100% 정도 성능 향상을 시킬 수 있다고 생각한다면, 휴리스틱을 통해서도 50%정도 성능을 향상 시킬 수 있을 것이다. 가령 앱 순위를 메기는 경우 설치 비율이나 설치 수를 사용 할 수 있을 것이다. 스팸 메세지 탐지의 경우 이전에 스팸을 보낸 사용자를 필처링 하면된다. 휴리스틱한 방법을 사용하는 것에 대해 겁내지 마라. 연락처 순위를 메기는 경우 가장 최근에 많이 사용한 순으로 정렬하면 된다. 제품(서비스)에 머신러닝이 절대적으로 필요하지 않는 경우, 적절한 데이터가 모일 때까지는 머신러닝을 사용할 필요는 없다. 

##### Rule #2: First, design and implement metrics.

Rule #2: 먼저 Metric을 설계하고 구현하라. 

 머신러닝 시스템을 본격적으로 적용하기 전에. 현재 시스템에서 가능한 많이 분석(Track)하라. 여기에는 아래와 같이 이유가 있다. 
 
1.시스템의 사용자로 부터 권한을 얻기가 더 쉽다.
2.미래에 어떤 부분에 대해 우려 되는 점이 있다면, 지금부터 기록하여 변화를 보는 것이 더 낫다.
3.메트릭 계측에 대한 부분을 생각하며 시스템을 설계하면 더 나은 구현을 할 수 있다.  앞으로 더 나은 것이 될 것이다. 특히, 우리 모두는 나중에 메트릭을 계측을 위해 log를 뒤져가며 개발 하는 것을 원하지 않는다!
4.어떤 것들이 변하는지, 그대로 유지하는지 알아 차리기 쉽다. 
5.
 메트릭 수집에 대한 보다 자유로운(liberal) 방식을 통해, 시스템에 대한 빅피쳐를 그릴 수 있다. 문제에 주목하고, 그것을 추적하는 메트릭을 추가하라. 릴리즈에서 몇 가지 양적 변화대해 관심이 있다면, 그것을 추적하기 위해 메트릭을 추가하라!

##### Rule #3: Choose machine learning over a complex heuristic. 

Rule #3: 복잡한 휴리스틱 방법보다 머신러닝을 선택하라.   
 간단한 휴리스틱 방법으로 제품을 릴리즈 할 수 있다. 하지만 복잡한 휴리스틱 방법을 사용하면 유지보수하기는 어렵다. 데이터와 기본 아이디어가 있다면 기계 학습을 통해 구현하자. 대부분의 소프트웨어 엔지니어링 작업에서와 마찬가지로 지속적인 업데이트를 할것이며, 일반적으로 머신러닝 모델이 업데이트 및 유지 관리가 더 쉽다.(Rule #16 참조)
