---
title:  Week 10 Learning with Large Datasets
category: CourseraMachineLearning
excerpt: |
  학습모델의 데이터 개수가 작아 High Variance현상이 발생할 때, 데이터 셋을 늘리는 것은 큰 도움이 된다. 반면 High Bias현상에 대해서는 그리 큰 도움이 되지는 않는다. 
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  
  ref: [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning "Coursera ML")
  
  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")
  
feature_image: "https://i.imgur.com/jds8Wur.png"
image: "https://i.imgur.com/jds8Wur.png"
---

### Anomaly Detection


![](http://cfile4.uf.tistory.com/image/2241C9345866EFD026DF91 "Anomaly Detection" "width:600px;height:300px;float:center;padding-left:10px;")

![](http://cfile10.uf.tistory.com/image/232878345866EFD030B590 "Anomaly Detection" "width:600px;height:200px;float:center;padding-left:10px;")

학습모델의 데이터 개수가 작아 High Variance현상이 발생할 때, 데이터 셋을 늘리는 것은 큰 도움이 된다. 반면 High Bias현상에 대해서는 그리 큰 도움이 되지는 않는다. 
학습 모델을 세우다 보면 m=100,000,000과 같이 큰 데이터 셋을 사용 할 수 있다. 이러한 경우 gradient descent를 수행 할 때 1억번의 합연산을 수행해야 할 것이다. 이러한 연산은 매우 많은 메모리를 차지하며 시간 또한 오래 걸린다. 아래에 이를 피하는 방법을 소개한다. 

![](http://cfile22.uf.tistory.com/image/2242FB345866EFD1269290 "Anomaly Detection" "width:600px;height:200px;float:center;padding-left:10px;")

### Stochastic Gradient Descent       

Stochastic gradient descent (확률적 경사 하강법)은 기존(Full Batch) gradient descent의 대안으로 데이터 셋이 큰 경우 보다 효과적이고 확장성있는 동작을 보여준다. 

Stochastic gradient descent는 약간 다른 방식으로 표현되지만 비슷한 동작을 한다. 

$$cost(\theta,(x^{(i)}, y^{(i)})) = \dfrac{1}{2}(h_{\theta}(x^{(i)}) - y^{(i)})^2$$

유일하게 다른 부분은 상수 m 대신에 1/2를 썼다는 것이다.

$$J_{train}(\theta) = \dfrac{1}{m} \displaystyle \sum_{i=1}^m cost(\theta, (x^{(i)}, y^{(i)}))$$

$J_{train}$은 이제 각 학습 데이터에서 구한 모든 비용 함수의 평균을 나타낸다. 

알고리즘은 다음과 같은 과정으로 동작한다. 

1. 데이터 셋을 무작위로 섞는다. 
2. $i = 1\dots m$ 에 대해 
$$\Theta_j := \Theta_j - \alpha (h_{\Theta}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}_j$$

위 알고리즘은 한 번에 한 개 데이터에 대해서만 계산한다. 이렇게 해면 m개 학습 데이터에 대한 gradient descent계산하지않고, 한 개 데이터에 대한 gradient descent를 수행 할 수 있다. 따라서 글로벌한 최소값에 수렴하지 않으며, 무작위로 움직인다, 하지만 이를 충분히 많이 반복하면 글로벌한 최소 값에 충분히 근접한 결과를 보인다. 

일반적으로 Stochastic gradient descent 수행시 1~10회 반복 함으로써 글로벌한 최소 값에 근접하도록 한다. 

![](http://cfile28.uf.tistory.com/image/244045345866EFD126625A "Stochastic Gradient Descent" "width:600px;height:200px;float:center;padding-left:10px;")

##### Mini-Batch Gradient Descent

Mini Batch gradient descent는 때때로 Stochastic gradient descent보다 빠른 수렴 속도를 보여준다. Full Batch gradient descent에서 사용하는 전체 대에터 셋 개수 m과 Stochastic gradient descent에서 하나의 데이터만 사용 했던 것과는 달리, 여기서는 b개 만큼의 데이터 셋을 사용한다. 보통 b의 범위는 2-100 이다. 

예를 들어 b=10, m=1000이라고 하면, 
다음 $i = 1,11,21,31,\dots,991$에 대해

$$\theta_j := \theta_j - \alpha \dfrac{1}{10} \displaystyle \sum_{k=i}^{i+9} (h_\theta(x^{(k)}) - y^{(k)})x_j^{(k)}$$

을 반복한다. 10개 정도 데이터 셋에 대해서는 쉽게 합연산을 수행 할 수 있다. 연산에서 얻는 또 다른 이점은 벡터를 사용하여 쉽게 표현 할 수 있다는 것이다. 

##### Stochastic Gradient Descent Convergence

Stochastic gradient descent에서 learning rate α는 어떻게 설정할까?  또한 Stochastic gradient descent가 클로벌 최소값에 근접하도록 하려면 어떻게 해야 할까?
한가지 방법은 1000여개 정도 데이터를 사용했을 때 가설함수의 평균을 그려보는 것이다. gradient descent를 반복하면서 비용함수를 계산하고 저장 할 수 있다.  learning rate가 작은 경우, Stochastic gradient descent로 좋은 결과를 얻어 낼 수 있다. 
Stochastic gradient descent 적용시 learning rate가 크면 무작위로 진동하며 이동하다 글로벌 최소값을 지나칠 수 있기 때문에, learning rate를 줄여야한다. 알고리즘의 성능을 그리기 위해 데이터의 수를 늘리면 위에서 그렸던 선이 더 부드러워 진다. 반면 데이터의 수가 적은 경우 선에 노이즈가 많이 생겨 경향성을 찾기가 어려워 진다. 
글로벌 최소값을 찾기위한 한가지 좋은 방법은 learning rate를 서서히 줄이는 것이다. 가령 다음과 같은 식을 사용할 수 있다. 

$$\alpha = \dfrac{const1}{iterationNumber + const2}$$

하지만 보통 파라미터 수를 늘리는 것을 기피 하므로, 이러한 방법이 자주 사용되지는 않는다.  

### Online Learning

웹 사이트에서 사용자의 지속적인 활동에 기반하여, 업데이트 되는 (x,y)에 대한 학습을 수행 할 수 있다. 사용자의 입력이나 활동을 통해 x와 그에대한 예측 y를 수집한다. 
수집과 병행하여 파라미터 θ를 업데이트 할 수 있다. 이렇게 하면 새로운 데이터에 대해 계속해서 업데이트하므로 새로운 사용자에 대한 예측도 가능해진다.

##### Map Reduce and Data Parallelism

![](http://cfile6.uf.tistory.com/image/2528FE345866EFD22F5E47 "Online Learning" "width:600px;height:300px;float:center;padding-left:10px;")

gradient descent의 Batch를 나누고, 업데이트 되는 파라미러를 공유하여 여러 다른 디바이스(연산장치)를 통해 병렬적으로 학습을 수행할 수 있다. 
다바이스의 수 만큼 학습데이터 셋을 나눌 수 있다. 따라서 각 디바이스에서는 다음과 같이 p~q만큼의 데이터에 대해서만 학습을 수행한다. 

$$\displaystyle \sum_{i=p}^{q}(h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

MapReduce는 이러한 dispatched(mapped) 계산 결과를 취합하여 파라미터를 학습 시킨다.    

$$\Theta_j := \Theta_j - \alpha \dfrac{1}{z}(temp_j^{(1)} + temp_j^{(2)} + \cdots + temp_j^{(z)})$$

모든 $j = 0, \dots, n$에 대해 반복한다. 
위 과정은 단순히 각각 디바이스에서 계산된 결과를 취합하고, 평균은 구한 다음 learning rate를 곱해준 다음 theta를 업데이트 하는 것에 지나지 않는다. 
알고리즘이 학습 데이터셋의 합으로 표현 된다면 MapReduceable하다고 볼 수 있다. 선형 회귀나 로지스틱 회귀는 쉽게 병렬화를 적용 할 수 있다. 
신경망을 통해 학습하는 경우 여러 디바이스에 대해 forward propagation, back propagation 작업을 수행 할 수 있다. 각 디바이스에서 계산된 derivative를 master서버로 보내고, 이를 취합한 다음 각 디바이스로 보내 파라미터를 업데이트 한다.    
 
