---
title: Rules of Machine Learning [Part3]
category: RulesOfMachineLearning
excerpt: |
  학습 데이터 통해 학습 모델을 만들어 동작하는 앤드 투 앤드 시스템을 갖추면 두 번째 단계가 시작된다. 두번째 단계에서는 다양한 피쳐를 적용하여 정교한 매트릭을 만들어 목표에 맞게 모델을 발전 시켜야 한다. 


feature_text: |
  ## Martin Zinkevich - Rules of Machine Learning
  Google 개발자인 Martin Zinkevich가 정리한 문서를 소개

  ref: [http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf "martin")

feature_image: "https://picsum.photos/2560/600/?image=1073"
image: "https://picsum.photos/2560/600/?image=1073"
comment: true
---

### ML Phase II: Feature Engineering

학습 데이터 통해 학습 모델을 만들어 동작하는 앤드 투 앤드 시스템을 갖추면 두 번째 단계가 시작된다. 두번째 단계에서는 다양한 피쳐를 적용하여 정교한 매트릭을 만들어 목표에 맞게 모델을 발전 시켜야 한다. 



##### Rule #16: Plan to launch and iterate.

Rule #16: 공개과 반복합습에 대한 계획을 세우라. 

 지금 공개한 모델은 마지막 모델이 아니다. 이번 공개 모델에서의 복잡성으로 인해 다음 모델 출시가 지연 되는 것을 고려하라. 새 모델을 출시하는데는 다음과 같은 이유가 있다. 
1. 새로운 피쳐를 도입
2. 기존 모델과 통합 및 정규화
3. 목표에 맞게 튜닝

##### Rule #17: Start with directly observed and reported features as opposed to learned features.   

Rule #17: 미리 학습된 피쳐보다 직접적으로 관찰되고 기록되는 피쳐부터 시작하라. 
미리 학습된 피쳐보다 눈에 보이는 피쳐를 기준으로 모델을 작성하면 시스템의 목표에 더 부합하는 접근법을 적용 할 수 있다. 

##### Rule #18: Explore with features of content that generalize across contexts.

Rule #18: 일반적인 컨텍스트에 맞는 컨텐츠의 피쳐를 탐색하라. 

 머신러닝은 종종 매우 큰 서비스의 일부분으로 동작한다. 서비스에서 제공하는 컨텐츠의 피쳐(ex. 유튜브의 조회수)는 머신러닝에서도 좋은 피쳐가 된다. 

##### Rule #19: Use very specific features when you can.

Rule #19: 가능한 매우 구체적인 피쳐를 사용하라. 

 수 많은 데이터를 통해 몇가지 복잡한 피쳐보다 매우 많은 양의 간단한 피쳐를 사용할 수 있다. 많은 피쳐를 다루는 것을 두려워 할 필요는 없다. 정규화 과정을 통해 중요하지 않은 피쳐를 없앨 수 있기 때문이다.  

##### Rule #20: Combine and modify existing features to create new features in human understandable ways.

Rule #20: 기존 피쳐를 통합, 가공하여 의미있는 새로운 피쳐를 생성하라.

 이산화(Discretization) 과정을 통해 연속된 피쳐에서 여러 개별 피쳐를 만들 수 있다. 피쳐 열의 외적을 통해서도 기존 피쳐를 통합하여 새로운 피쳐를 만들 수 있다. 
 
##### Rule #21: The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have.

Rule #21: 선형 모델에서 피쳐(weight) 수는 대략 데이터의 양에 비례한다.

 모델의 복잡한 정도는 기본적으로 데이터의 양에 비례한다. 데이터가 충분하다면 복잡한 모델로 디자인해도 되지만, 그렇지 않은 경우 단순한 모델로 디자인하라. 



##### Rule #22: Clean up features you are no longer using.

Rule #22: 더 이상 사용하지 않는 피쳐는 정리하라. 

 필요한 겨웅 언제든 피쳐를 다시 도입할 수 있다. 그러니 사용하지 않는 피쳐의 경우 정리하라. 이러한 피쳐가 남아 있을 경우 속도 및 성능 저하를 야기 할 수 있다. 



#### Human Analysis of the System
다음 장으로 넘어가기 전에, 지금까지와는 약간 다른 관점으로 시스템을 향상시키는 법을 살펴 보자. 
 
##### Rule #23: You are not a typical end user.

Rule #23: 당신은 일반적인 사용자가 아니다. 

 주어진 시스템은 실제 사용자에 대한 실험을 통해 추가 테스트를 시행해야한다. 여기에는 두 가지 이유가 있는데, 시스템을 직접 만든 개발자는 시스템이 어떻게 동작하는지 알고 있기 때문에 제대로 테스트를 하지 못한다. 그리고 개발자의 임금을 생각할 때 테스트에 시간을 솓는 것은 큰 낭비다.     

##### Rule #24: Measure the delta between models.

Rule #24: 모델 간의 차이를 측정하라. 

 모델을 시스템에 넣어 테스트 하기 전에, 기존 모델과의 차이를 측정하라. 크게 차이나지 않는다면 굳이 추가적인 실험을 수행할 필요는 없다.   

##### Rule #25: When choosing models, utilitarian performance trumps predictive power.

Rule #25: 모델을 선택할 때 실제 사용성에 대한 성능이 모델의 예측이 얼마나 정확한가 보다 더 중요하다  

 모델의 예측 정확도를 높히는 것도 좋지만, 가장 중요한 것은 그 예측으로 무엇을 할 것인가다. 이 두가지는 일치해야하며 그렇지 않은 경우 모델의 정확도가 좋다고 하더라도, 시스템의 사용성을 떨어진다. 이러한 경우가 자주 발생한다면, 모델의 목표나, matric을 다시 설정해야한다. 

##### Rule #26: Look for patterns in the measured errors, and create new features.

Rule #26: 측정한 오류의 패턴을 찾고 새로운 피쳐를 생성하라. 

 학습데이터가 잘못된 경우도 있다. 머신러닝에서 이러한 잘못된 입력을 수정하는 기능을 제공하면, 학습 과정에서 잘못된 입력(false positive , false negative)에 다른 오류를 줄일 수 있다. 또한 시스템에서 잘못된 피쳐를 입력하는 경우에도 이를 무시 할 수 있다. 

##### Rule #27: Try to quantify observed undesirable behavior. 

Rule #27: 관찰된 잘못된 행동을 계량하라. 

 개발시 matric에 포함되지 않는 피쳐에 좌절하는 경우도 있다. 이러한 경우 피쳐를 계량화 하기 위해 뭐든지 다 해야한다. 피쳐를 계량화 한다면 이를 matric으로 사용 할 수 있다. 일반적으로 "먼저 측정한 다음, 최적화"한다. 

##### Rule #28: Be aware that identical shortterm behavior does not imply identical longtermbehavior. 

Rule #28: 단기적으로 발생하는 현상이 장기적으로 발생하는 현상과 같지 않다는 것을 유념하라. 

 시간에 따라 학습데이터가 달라지며, 계속하여 입력데이터가 누적될 시 장기적인 동작은 처음과 상당히 차이난다. 장기적으로 어떻게 동작하는 지를 예측하는 방법은 모델이 실제 동작할 때 데이터를 획득하여, 이것으로 다시 학습하는 방법 밖에 없다. 하지만 이를 수행하기란 매우 어렵다.   


#### Training Serving Skew

학습과 그것을 실제 서비스하는 것은 다르다. 파이프 라인의 차이 또는 입력데이터의 차이로 인해 학습시의 성능이 실제 서비스에서는 제대로 나오지 않는 경우도 있다. 이번 장에서는 이것에 대해 다룬다. 

##### Rule #29: The best way to make sure that you train like you serve is to save the set of features used at serving time, and then pipe those features to a log to use them at training time.

Rule #29: 학습과 서비스의 성능을 일치 시키기 위한 가장 좋은 방법은 서비스에서 사용된 피쳐를 다음 학습에 반영하는 것이다. 

 서비스 시 남긴 로그를 통해 주요 피쳐를 뽑은 다음, 이를 다음 학습에 반영한다면 성능을 비약적으로 높일 수 있다. 

##### Rule #30: Importance weight sampled data, don’t arbitrarily drop it!

Rule #30: 샘플링된 데이터의 주요 가중치 값을 임의로 삭제하지 말라

 학습데이터가 많아지는 경우 특정 샘플만 사용하고 나머지는 수시하는 경우가 있다. 이는 명백한 실수이며, 학습에서 데이터를 삭제할 시 많은 문제가 발생할 수 있다. 이런 데이터는 삭제 할 수 있지만, 모델의 피쳐를 유지하기 위해 가중치 값은 적용하는 것이 좋다. 

##### Rule #31: Beware that if you join data from a table at training and serving time, the data in the table may change.

Rule #31: 학습이나 서비스 중 데이터에 접근하는 경우 기존 데이터의 변형을 일으킨다. 

 피쳐가 저장되어 있는 테이블을 수정하는 경우 학습이나 서비스 중에 피쳐의 특성이 변형 될 수 있다. 이를 피하는 방법은 서비스 시에만 천천히 기록하는 것인데 그럼에도 불구하고 위의 문제가 완전히 해결되지는 않는다.  

##### Rule #32: Reuse code between your training pipeline and your serving pipeline whenever possible.

Rule #32: 가능하다면 학습에 사용한 코드를 서비스에도 재사용하라. 

 배치(Batch)를 통해 입력을 묶어서 일괄적으로 학습을 진행하는데, 서비스에서는 각 입력에 대해 처리해야한다. 그러나 코드를 재사용 할 수 있는 몇 가지 방법이 있다. 가령 두 입력을 모두 처리 할 수 있는 시스템 고유의 개체를 만들 수 있다. 또한 둘 사이에 다른 프로그래밍 언어를 사용하지 않도록 주의 해야한다.   



##### Rule #33: If you produce a model based on the data until January 5th, test the model on the data from January 6th and after.

Rule #33: 1월 5일 까지의 데이터 기반 모델을 사용한다면, 1월 6일 이후의 데이터로 테스트한다.

 우리가 학습한 데이터 이후 데이터를 통해 모델의 성능을 측정한다. 물론 이런 경우 학습 시보다 성능은 떨어지겠지만, 심각할 정도로 떨어진다면 모델에 문제가 있음을 알아차려야 한다.      

##### Rule #34: In binary classification for filtering (such as spam detection or determining interesting emails), make small shortterm sacrifices in performance for very clean data.

Rule #34: 스팸처리와 같은 필터링을 위한 이진 분류의 경우 데이터를 깔끔하게 하기위해 성능적인 부분에서 단기적으로 양보해야한다.  

 스팸처리기의 경우 실제 Negative 데이터는 매우 적다. 학습시에는 성능이 조금 떨어지더라도 Negative 데이터를 많이 넣어 알고리즘의 Negative 데이터에 대한 이해를 높여야한다. 이렇게 한다면 실제 서비스시 더 나은 성능을 기대할 수 있다. 

 
##### Rule #35: Beware of the inherent skew in ranking problems.

Rule #35: 순위 결정 문제에서의 근본적인 왜곡을 주의하라. 

 순위 결정 문제에서 알고리즘이 변경되면, 알고리즘이 나중에 학습할 데이터까지 변경되는 효과가 난다. 왜냐하면 순위 결정 알고리즘은 이전 순위를 기반으로 학습되기 때문이다. 따라서 이러한 skew를 염두하고 모델을 설계해야한다.

##### Rule #36: Avoid feedback loops with positional features.

Rule #36: Positional features로 피트백 루프를 방지하라. 

 웹 서비스에서 콘텐츠의 위치는 사용자와의 상호작용에 영향을 미친다. 윕 페이지의 앱이나 서비스에 대한 피쳐를 추가함으로써 이를 모델에 반영 할 수 있다. 이때 이러한 위치에 관한 피쳐는 서비스에서만 나타나므로 이를 분리하여 처리하는 것이 중요하다. 

##### Rule #37: Measure Training/Serving Skew.

Rule #37: 학습/서비스의 왜곡을 측정하라. 

 왜곡을 유발하는 몇 가지 원인을 측정하라. 
 - 예외(holdout) 데이터 입력과 학습데이터간 성능 차이
 - 다음 날의 데이터 입력과 홀드아웃 데이터의 성능 차이, 정규화로 튜닝 할 수 있지만 한쪽에 치우쳐지지 않도록 주의해야 한다.
 - 다음 날의 데이터와 실제 데이터의 성능차이. 입력 데이터를 반영하여 학습을 할 수 있다. 이러한 경우 성능이 동일하게 나와야하며, 만약 다르다면 프로그래밍적인 오류가 발생한것을 나타낸다.  
