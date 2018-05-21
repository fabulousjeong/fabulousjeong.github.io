---
title: Rules of Machine Learning [Part4]
category: RulesOfMachineLearning
excerpt: |
   두번째 단계가 끝나가면 새로운 방법에 대해 직접 모색해야한다. 우선 학습에 더 이상 발전이 없으며, 서비스가 되면서 측정항목들 사이에 충돌이 발생하는 경우도 있다. 모델을 발전시키려면 더욱 정교하게 머신러닝을 설계해야한다. 


feature_text: |
  ## Martin Zinkevich - Rules of Machine Learning
  Google 개발자인 Martin Zinkevich가 정리한 문서를 소개

  ref: [http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf "martin")

feature_image: "https://picsum.photos/2560/600/?image=1073"
image: "https://picsum.photos/2560/600/?image=1073"
comment: true
---

### ML Phase III: Slowed Growth, Optimization Refinement, and Complex Models

 두번째 단계가 끝나가면 새로운 방법에 대해 직접 모색해야한다. 우선 학습에 더 이상 발전이 없으며, 서비스가 되면서 측정항목들 사이에 충돌이 발생하는 경우도 있다. 모델을 발전시키려면 더욱 정교하게 머신러닝을 설계해야한다. 



##### Rule #38: Don’t waste time on new features if unaligned objectives have become the issue.

Rule #38: 곧  이슈가 될 정의되지 않은 목표가 있다면, 새로운 피쳐에 대해 더이상 시간낭비 하지 말라.

 더이상 학습 모델이 발전되지 않고 안정적인 상태가 된다면, 이를 해결하기 위해 기존 서비스의 목표에 더욱 부합하게끔 학습 목표를 바꿀 새로운 이슈를 찾을 것이다. 기존의 피쳐로 최적화 할 수 있지만, 전혀 새로운 피쳐를 도입할 수 도있다. 

  
##### Rule #39: Launch decisions are a proxy for longterm product goals.

Rule #39: 공개에 대한 결정은 장기 서비스 목표에 대한 프록시(Proxy)와 같다.

 출시후 향후 시스템이 얼마나 좋은지를 예측할 수 있다. 서비스 런칭 후에는 1일 활성 사용자(DAU)나 광고 등을 신경써야한다. 이것 역시 측정 할 수 있고, 매트릭으로 나타낼 수 있다. 이 매트릭이 나빠지지 않게끔 서비스를 개선해야한다. 이를 학습하여 서비스의 성공 여부를 예측 할 수 있지만, 이는 컴퓨터 비전이나, 자연어 처리 만큼 어렵다. 

##### Rule #40: Keep ensembles simple.

Rule #40: 앙상블을 단순하게 하라. 

 여러 모델을 결합하여 시스템을 구성할 수 있다. 이를 앙상블 모델이라 한다. 한 모델의 출력이 다른 학습 모델의 입력으로 들어가서 피쳐를 공유 할 수도 있다. 이를 함부로 결합하면 좋지 않은 결과가 나올 수 있다. 앙상블 모델에서는 부분 모델의 정확도가 올라가는 경우에도, 이것이 전체 정확도를 떨어뜨리지 않도록 주의 해야한다. 

##### Rule #41: When performance plateaus, look for qualitatively new sources of information to add rather than refining existing signals.

Rule #41: 성능이 더이상 나아지지 않으면, 기존 정보를 개선하는 것 보다 질적으로 새로운 정보를 찾아라. 

 기존 정보를 개선하는 방법으로 몇 분기 동안 1%이상 개선된 결과를 보지 못했다면, 이제는 근본적으로 다른 피쳐를 도입하고, 이를 반영하게끔 모델을 수정해야한다. 하지만 새로운 피쳐를 도입하면 모델의 복잡성도 증가하므로, 이에 따른 장단점을 고려해야한다. 


##### Rule #42: Don’t expect diversity, personalization, or relevance to be as correlated with popularity as you think they are.

Rule #42: 다양성, 개인화, 연관성이 항상 당신이 생각한대로 서비스의 인기로 이어지지 않는다. 

 서비스의 인기를 측정하는 것은 다양성, 개인화, 연관성이 아닌 시스템에서 사용자의 클릭수, 머무는 시간, 공유 횟수 등이다. 하지만 이것이 중요하지 않다는 의미는 아니다. 장기적인 관점에서 사용자의 인기와 더불어 다양성, 개인화, 관련성을 기반으로 서비스 목표를 수정 할 수 있다.   


