---
title: Rules of Machine Learning [Part2]
category: RulesOfMachineLearning
excerpt: |
  첫번째 파이프라인에서는 인프라시스템에 중점을 둔다. 지금 당장 머신러닝에 대해 생각하고 싶지만 잠시 뒤로 미루자. 처음 파이프 라인을 잘 설정하지 못한다면 어떤일들이 발생하는지 알기 어렵다. 


feature_text: |
  ## Martin Zinkevich - Rules of Machine Learning
  Google 개발자인 Martin Zinkevich가 정리한 문서를 소개

  ref: [http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf "martin")

feature_image: "https://picsum.photos/2560/600/?image=1073"
image: "https://picsum.photos/2560/600/?image=1073"
comment: true
---

### ML Phase I: Your First Pipeline

첫번째 파이프라인에서는 인프라시스템에 중점을 둔다. 지금 당장 머신러닝에 대해 생각하고 싶지만 잠시 뒤로 미루자. 처음 파이프 라인을 잘 설정하지 못한다면 어떤일들이 발생하는지 알기 어렵다. 

##### Rule #4: Keep the first model simple and get the infrastructure right.

Rule #4: 첫번째 모델은 간단하게 구현하고, 올바른 인프라시스템을 구축하라.

첫번째 모델은 제품에 있어 가장 큰 향상을 가져오므로, 굳이 멋지게 만들 필요는 없다. 간단하게 만든다고 하더라도 여러분이 생각하는 것보다 인므로 구축에서 많은 이슈가 발생할 것이다. 멋진 머신러닝 시스템을 사용하기 전에 다음 사항을 결정하라. 

1. 학습 알고리즘에 대한 예제(데이터)를 얻는 방법
2. 여러분이 개발할 시스템에서 좋고, 나쁜 것에 대한 의미(기준)
3. 모델을 응용프로그램에 통합하는 방법, 모델을 실시간으로 적용하거나, 미리 계산한 결과를 사용할 수 있다. 

간단한 피쳐를 사용하는 것은 아래 사항을 보다 쉽게 보장한다. 

1. 올바른 머신러닝 알고리즘
2. 모델이 합리적인 가중치를 가지게 학습 
3. 서버에서 올바른 모델이 되게 학습 
4. 
위 세가지가 안정적으로 동작하는 시스템을 갖추면, 작업 중 대분분을 수행한 것과 다름없다. 간단한 모델은 조금 더 복잡한 모델을 테스트 하는데 사용할 수 있는 기본 메트릭 및 기본 동작방식을 제공한다. 시스템이 복잡해 지는 것을 피하기 위해 머신러닝의 우선 순위를 낮추는게 좋다. 

Rule #5: Test the infrastructure independently from the machine learning.    
Rule #5: 머신러닝과 별개로 인프라시스템을 테스트하라. 
인프라시스템이 테스트 가능한지, 머신러닝 부분이 캡슐화 되어 따로 테스트 가능하지 확인해야한다. 
1. 알고리즘이 데이터를 제대로 가져오는지 테스트 하라. 피쳐가 제대로 채워졌는지, 개인정보 보호가 있는 부분은 수동으로 입력하여 테스트 되는지 확인해야한다. 
2. 트레이닝 알고리즘을 제외하고 모델을 테스트하라. 실제 서비스 환경에서의 결과와 학습 환경에서의 결과가 동일한지 확인하라. 데이터를 분석하고, 이해하는 것이 중요하다. 

##### Rule #6: Be careful about dropped data when copying pipelines.    

Rule #6: 파이프라인 복사 시 누락 된 데이터에 주의하라.

종종 기존 파이프라인을 복사하여 파이프 라인을 구축한다. 이러한 과정에서 기존 파이프라인은 새로운 파이프라인에 필요한 데이터를 누락 시킬 수 있다. 예를들어 "Google Plus What’s Hot" 파이프라인은 오래된 게시물을 삭제한다. 하지만 이 파이프라인을 기반으로 "Google Plus Stream"을 구현하였다. 하지만 위 파이프라인에서는 오래된 게시물도 표시해야한다. 이런식으로 기존 파이프라인을 복사 할 때는 현재 파이프라인에 맞게 수정해야한다. 



##### Rule #7: Turn heuristics into features, or handle them externally.

Rule #7: 휴리스틱 부분을 피쳐로 활용하거나, 외부에서 다뤄라

 기존 방법에서 머신러닝으로 바꿀때 휴리스틱 방법론에서 사용한 경험을 적용하면 큰 도움이 된다. 더 원활히 적용할 수 있으며, 기존 휴리스틱 방법론에는 시스템 및 솔루션에 대한 많은 직관이 담겨 있다.   



#### Monitoring

##### Rule #8: Know the freshness requirements of your system.     

Rule #8: 시스템의 업데이트 주기에 대해 알고 있어야 한다.

 현재 시스템의 성능이 떨어지는 주기에 대해 파악하고 있어야 한다. 이러한 정보는 모니터링에서 어느 부분은 우선으로 업데이트 할지 정하는데 도움을 준다. 만약 업데이트 하지 않았을 경우 하루에 10%씩 성능이 떨어 진다면, 그 부분에 대해서는 지속적인 모니터링이 필요하다.      

##### Rule #9: Detect problems before exporting models.

Rule #9: 모델을 추출하기 전에 문제점을 파악하라. 
 많은 머신러닝 시스템에서 별도로 모델을 개발한 다음, 그 것을 추출하여 서비스에 적용한다. 추출된 모델에서 이슈가 발생할 경우, 그 문제가 모델에서 발생한 것인지, 서비스에서 발생한 것인지, 추출 과정에서 발생한 것인지 파악하기가 어렵다. 따라서 모델을 추출하기 전에 그 성능에 대한 충분한 검증을 거쳐야 한다.

##### Rule #10: Watch for silent failures.

Rule #10: 감지하기 어려운 오류를 살펴보라

 이러한 문제는 다른 시스템 보다 머신러닝에서 주로 발생한다. 데이터가 더 이상 업데이트 되지 않으면 성능이 서서히 떨어진다. 몇 개월 뒤 문제를 발견하고 업데이트하여 성능을 다시 올릴 수 있지만, 문제를 빨리 발견하기는 어렵다. 
 
##### Rule #11: Give feature column owners and documentation.

Rule #11: 피쳐의 작성자와 문서를 남겨라 

 거대 시스템의 경우 많은 피쳐가 정의 되어 있다. 누가 그 것을 만들고 관리하는지 파악해야한다. 피쳐명에서 그 특성을 유추 할 수 있지만, 세부적인 의미를 설명해 줄 담당자 및 문서를 남겨야 한다.   

#### Your First Objective

머신러닝에서는 하나의 목표를 정하고, 그것을 최적화 하는 과정을 거친다. 반면 시스템에서 살펴봐야 하는 수치들을 Matric이라한다. 우리는 이 둘을 잘 구문 해야하며, 때론 매트릭이 그리 중요하지 않는 경우도 있다. 

##### Rule #12: Don’t overthink which objective you choose to directly optimize.

Rule #12: 직접 최적화하기로 한 목표를 선택하는 것에 대해 크게 생각하지 마라.

 시스템에는 여러 매트릭이 있으며, 우리가 설정한 목표를 달성하기 위해 이것을 최적화한다. 이를 위해 궁극적으로 필요한 모든 매트릭을 미리 설정할 필요는 없다. 점진적으로 늘리거나 수정해 나가며 시스템을 구성하라. 

##### Rule #13: Choose a simple, observable and attributable metric for your first objective.

Rule #13: 간단하고, 관찰가능하며 관련성이 높은 매트릭을 선택하라. 
 처음부터 목표에 필요한 모든 매트릭을 알 수는 없다. 측정하기 쉬운 항목부터 학습하며, 각종 항목을 추가하라. 
  
##### Rule #14: Starting with an interpretable model makes debugging easier.

Rule #14: 해석가능한 모델로 시작함으로써 디버깅을 쉽게 할 수 있다. 
 
선형회귀, 로지스틱 회귀 등의 방법은 확률 모델에 기반하므로, 예상 값을 쉽게 해석 할 수 있다. 이러한 모델은 다른 복잡한 모델에 비해 디버그 하기가 쉽다. 

##### Rule #15: Separate Spam Filtering and Quality Ranking in a Policy Layer.  

Rule #15: Policy Layer에서 스팸 필터 및 품질에 대한 평가를 분리하라. 

 데이터에서 스팸을 걸러내는 것과 품질을 평가하는 것은 별개의 문제다. 스팸을 걸러낼 때 품질 평가를 사용하면 안된다. 둘은 분리된 알고리즘을 사용하여 수행해야한다. 
