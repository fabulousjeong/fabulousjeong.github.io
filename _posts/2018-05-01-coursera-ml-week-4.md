---
title: Week 4 Neural Networks
category: CourseraMachineLearning
excerpt: |
  신경망은 우리 자신의 두뇌가 어떻게 동작하는지를 대략적으로 모방한다. 신경망 알고리즘은 최근 컴퓨터 하드웨어의 발전으로 다시 관심을 끌고 있다. 
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  
  ref: [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning "Coursera ML")
  
  code: [https://github.com/fabulousjeong/ML-OctaveCode/](https://github.com/fabulousjeong/ML-OctaveCode/ "Code")
  
feature_image: "../assets/images/coursera_ML/title.png"
image: "./assets/images/coursera_ML/title.png"
---

### ML: Neural Networks: Representation

##### Non-linear Hypotheses

복잡한 데이터셋과 여러 변수에 대한 선형회귀를 수행하는 것은 매우 어렵다. 아래와 같이 세변수에 대해 이차원 가설함수를 세운다고 생각해보자. 

$$
\begin{align*}& g(\theta_0 + \theta_1x_1^2 + \theta_2x_1x_2 + \theta_3x_1x_3 \newline& + \theta_4x_2^2 + \theta_5x_2x_3 \newline& + \theta_6x_3^2 )\end{align*}
$$ 

위 식에서는 6가지 변수가 있다. 다음과 같은 '조합'함수에 의해 변수에 대해 얼마나 많은 항이 필요한지 계산할 수 있다. ($\frac{(n+r-1)!}{r!(n-1)!}$) http://www.mathsisfun.com/combinatorics/combinations-permutations.html 위 예의 경우 3가지 변수에 대해 2차항 까지 있으므로 다음과 같이 대입해 볼 수 있다. :$\frac{(3 + 2 - 1)!}{(2!\cdot (3-1)!)}$ 100개의 변수에 대해 2차항 가설함수를 세운다고 생각해보면 다음과 같이 $\frac{(100 + 2 - 1)!}{(2\cdot (100-1)!)} = 5050$개의 항(새로운 변수)가 생긴다. 일반적으로 가설함수가 이차함수인 경우에 대해서는 $\mathcal{O}(n^2/2)$ 복잡도를 가지며, 삼차함수인 경우에는 $\mathcal{O}(n^3)$ 복잡도를 가진다. 이렇게 이차, 삼차 함수에 대해서는 변수가 늘어남에 따라 다루는 항이 엄청나게 많이 늘어남을 볼 수 있다. 예를 들어 50x50 크기의 흑백 자동차 이미지를 분류한다고 하자. 변수는 픽셀의 수인 2500개이며 이를 각 픽셀마다 비교한다. 이제 여기에 대해 이차 가설함수를 세워보자. 이차항에 대해 $\mathcal{O}(n^2/2)$ 복잡도를 가지므로 전체 항의 수는 다음과 같다. $2500^2 / 2 = 3,125,000$ 이는 매우 비 효율적이다. 많은 변수를 다루는 경우 신경망학습(Neural network)은 좋은 대안이 된다.

![](http://cfile2.uf.tistory.com/image/2563CF44586CE97D1D82BE "Non-linear Hypotheses" "width:600px;height:300px;float:center;padding-left:10px;")


### Neuron and the Brain 

![](http://cfile23.uf.tistory.com/image/24731244586CE97E0E069C "Neuron and the Brain" "width:600px;height:300px;float:center;padding-left:10px;")

신경망은 우리 자신의 두뇌가 어떻게 동작하는지를 대략적으로 모방하였다. 이는 최근 컴퓨터 하드웨어의 발전으로 다시 관심을 끌고 있다. 우리의 뇌는 여러 다른 문제를 다루는 경우에도 오직 하나의 학습 알고리즘만 사용한다고 알려져 있다. 과학자들은 동물의 뇌에서 귀와 청각 피질 사이의 연결을 절단한 다음 시신경과 청각 피질을 연결했을 때, 청각피질에서 시각적 신호를 읽는 것을 발견했다. 이를 신경이식성(neuroplasticity)이라고하며 여기에 대해 많은 예시와 실험적 증거들이 있다.


### Model Representation I
![](http://cfile10.uf.tistory.com/image/23245B3D586CE87B1D5304 "Model Representation I" "width:600px;height:400px;float:center;padding-left:10px;")

ref:webspace.ship.edu

지금 부터는 신경망을 이용하여 어떻게 가설함수를 표현하는지를 설명한다. 간단히 말하자면, 뉴런은 기본적으로 입력(수상돌기)을 출력(축색돌기)로 전달하는 전기 입력(스파이크)를 다루는 계산 유닛이다. 


실제 이 강의의 모델에서 수상돌기는 입력 변수 $x_1\cdots x_n$와 유사하며, 출력은 가설함수의 결과와 같다. 이 모델에서 $x_0$는 종종 바이어스 유닛이라 불리며, 항상 1의 값을 가진다. 신경망의 분류학습에서 로지스틱 함수는 앞서 다룬 것과 같다. $\frac{1}{1 + e^{-\theta^Tx}}$. 신경망 학습에서는 이 함수를 시그모이드(sigmoid) 로지스틱 활성화(Activation) 함수라 부른다. "theta" 파라미터는 "weight"라 부르기도 한다. 시각적으로 간단히 아래와 같이 표현한다. 
$\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline \end{bmatrix}\rightarrow\begin{bmatrix}\ \ \ \newline \end{bmatrix}\rightarrow h_\theta(x)$  

![](http://cfile28.uf.tistory.com/image/217AE843586CE9D93334CF "Model Representation I2" "width:600px;height:350px;float:center;padding-left:10px;")

입력 노드(레이어 1)는 다른 노드(레이어 2)로 들어가 가설함수의 결과로 출력 된다. 첫번째 레이어를 입력레이어(input layer)라 부르며 최종 결과값이 계산되는 마지막 레이어를 출력레이어(output layer)라 부른다. 입력 레이어와 출력레이어 사이에 있는 레이어들은 히든레이어(hidden layer)라 부른다. 히든 레이어의 노드를 $a^2_0 \cdots a^2_n$과 같이 레이블링 하며 Activation units이라 부른다. 

$$
\begin{align*}& a_i^{(j)} = \text{"activation" of unit $i$ in layer $j$} \newline& \Theta^{(j)} = \text{matrix of weights controlling function mapping from layer $j$ to layer $j+1$}\end{align*}$$

하나의 히든레이어만 다룬다하면 아래와 같이 나타낼 수 있다. 

$$
\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \newline a_3^{(2)} \newline \end{bmatrix}\rightarrow h_\theta(x)
$$

각 Activation 노드의 값은 다음과 같은 과정으로 구해진다. 

$$
\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
$$

 위 식은 $\Theta^{(j)}$로 이루어진 3x4 행렬을 사용하여 위 과정을 수행 할 수 있다. 행렬의 크기는 아래와 같이 구한다. 
$$\text{If network has s_j units in layer j and} s_{j+1} \text{units in layer j+1, then} \Theta^j \text{will be of dimension} s_{j+1} \times (s_j + 1)$$

+1을 하는 이유는 $x_0$나 $\Theta_0^{(j)}$와 같은 바이어스 항을 고려해야 하기 때문이다. 
예: 레이어 1이 2개의 입력 노드를 가지고 있고 레이어 2에 4개의 활성화 노드가 있을때 $\Theta^{(1)}$의 크기는 4x3이며 아래와 같이 계산된다. 

$s_j = 2$, $s_{j+1} = 4$, 따라서 $s_{j+1} \times (s_j + 1) = 4 \times 3$

### Model Representation II

이번장에서는 위 함수를 벡터화해서 살펴본다. 여기서 함수 g의 내부 파라미터들의 합을 표현하는 새로운 변수 $z_k^{(j)}$를 정의 한다. 이를 이용해 위 식을 아래와 같이 표현 할 수 있다.

$$\begin{align*}a_1^{(2)} = g(z_1^{(2)}) \newline a_2^{(2)} = g(z_2^{(2)}) \newline a_3^{(2)} = g(z_3^{(2)}) \newline \end{align*}$$

다시 설명하자면, 레이어 j=2 이고, 노드 k인 경우 변수 z는 아래와 같다. 

$z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n$

x와 $z^j$를 벡터로 표현하면 아래와 같다. 
$$
\begin{align*}x = \begin{bmatrix}x_0 \newline x_1 \newline\cdots \newline x_n\end{bmatrix} &z^{(j)} = \begin{bmatrix}z_1^{(j)} \newline z_2^{(j)} \newline\cdots \newline z_n^{(j)}\end{bmatrix}\end{align*}
$$

다음과 같이 $x = a^{(1)}$이라고 보면 위 식을 아래와 같이 쓸 수 있다.

$z^{(j)} = \Theta^{(j-1)}a^{(j-1)}$ 

곱하는 행렬 $\Theta^{(j-1)}$의 크기는 $s_j\times (n+1)$와 같으며 여기서 $s_j$는 활성화 노드의 수와 같고 n+1은 벡터 $a^{(j-1)}$의 길이와 같다. 아래와 같이 활성화노드 j를 얻을 수 있다. 

$a^{(j)} = g(z^{(j)})$

$a^{(j)}$를 계산한 다음 레이어 j에 바이어스 유닛 $a_0^{(j)}$를 더한다. 이때 바이어스 유닛은 1의 값을 가진다. 
최종 가설함수를 구하기 위해 먼저, 다음 레이어의 z백터를 구한다. 

$z^{(j+1)} = \Theta^{(j)}a^{(j)}$

위와 같이 theta 행렬과 활성화 노드를 곱해서 구할 수 있다. 

마지막 theta 행렬 $\Theta^{(j)}$는 한개의 열로 구성되어 있으므로 마지막 결과는 한개 값으로 나온다. 아래의 식과 같이 나타낼 수 있다. 

$h_\Theta(x) = a^{(j+1)} = g(z^{(j+1)})$

위 식에서 레이어 j와 레이어 j + 1 사이에서 앞에서 로지스틱 회귀에서했던 것과 똑같은 작업을 수행하고 있음에 주목하자. 
신경망 중간에 레이어를 추가하여, 복잡하고 비선형적인 가설함수를 멋지게 만들수 있다.  


### Examples and Intuitions I

![](http://cfile7.uf.tistory.com/image/24190D3C586CEB9223BD40 "Examples and Intuitions I" "width:600px;height:450px;float:center;padding-left:10px;")

$x_1$ 과 $x_2$가 1인 경우에만 True를 반환하는 and 연산을 신경망으로 구축한 예를 살펴보자.
함수의 그래프는 아래와 같다. 

$$
\begin{align*}\begin{bmatrix}x_0 \newline x_1 \newline x_2\end{bmatrix} \rightarrow\begin{bmatrix}g(z^{(2)})\end{bmatrix} \rightarrow h_\Theta(x)\end{align*}
$$

$x_0$는 바이어스 변수이며 항상 1이라는 것을 상기하자.  
theta 행렬를 아래와 같이 설정하자. 
$\Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20\end{bmatrix}$
아래의 식을 통해 위 행렬이 바르게 설정 되었음을 볼 수 있다. 

$$
\begin{align*}& h_\Theta(x) = g(-30 + 20x_1 + 20x_2) \newline \newline & x_1 = 0 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-30) \approx 0 \newline & x_1 = 0 \ \ and \ \ x_2 = 1 \ \ then \ \ g(-10) \approx 0 \newline & x_1 = 1 \ \ and \ \ x_2 = 0 \ \ then \ \ g(-10) \approx 0 \newline & x_1 = 1 \ \ and \ \ x_2 = 1 \ \ then \ \ g(10) \approx 1\end{align*}
$$ 

위 예를 통해 신경망을 이용하여 가장 간단한 논리 연산자를 구현하였다. 신경망을 이용하여 다른 모든 논리연산자를 구현할 수 있다.  


### Examples and Intuitions II

![](http://cfile27.uf.tistory.com/image/2119213C586CEB932225D7 "Examples and Intuitions II" "width:600px;height:300px;float:center;padding-left:10px;")

AND, NOR, OR 연산자의 $\Theta^{(1)}$ 행렬은 아래와 같다. 
$$
\begin{align*}AND:\newline\Theta^{(1)} &=\begin{bmatrix}-30 & 20 & 20\end{bmatrix} \newline NOR:\newline\Theta^{(1)} &= \begin{bmatrix}10 & -20 & -20\end{bmatrix} \newline OR:\newline\Theta^{(1)} &= \begin{bmatrix}-10 & 20 & 20\end{bmatrix} \newline\end{align*}
$$

위를 조합하여 XNOR 연산자를 구현 할 수 있다. ($x_1$ 과 $x_2$가 같은 값을 가지는 경우 True 반환)

$$
\begin{align*}\begin{bmatrix}x_0 \newline x_1 \newline x_2\end{bmatrix} \rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \end{bmatrix} \rightarrow\begin{bmatrix}a^{(3)}\end{bmatrix} \rightarrow h_\Theta(x)\end{align*}
$$

첫번째 레이어에서 두번째 레이어를 구하기 위해 AND, NOR 매트릭스로 구성된 $\Theta^{(1)}$을 사용한다. 

$\Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20 \newline 10 & -20 & -20\end{bmatrix}$

첫번째 레이어에서 두번째 레이어를 구하기 위해 OR 매트릭스로 구성된 $\Theta^{(2)}$을 사용한다.  

$\Theta^{(2)} =\begin{bmatrix}-10 & 20 & 20\end{bmatrix}$

모든 노드에 대한 계산은 아래와 같다. 
$$
\begin{align*}& a^{(2)} = g(\Theta^{(1)} \cdot x) \newline& a^{(3)} = g(\Theta^{(2)} \cdot a^{(2)}) \newline& h_\Theta(x) = a^{(3)}\end{align*}
$$
위 두 레이어를 사용한 신경망 예를 통해 XNOR 논리 연산을 수행 할 수 있다. 


### Multiclass Classification

![](http://cfile10.uf.tistory.com/image/21593834586CEC402C4243 "Multiclass Classification" "width:600px;height:250px;float:center;padding-left:10px;")

여러 항목에 대해 분류하기위해, 가설 함수를 여러 변수를 가지는 벡터 형태로 반환해야한다. 위의 예와 같이 네개 항목 중 하나로 분류해보자.

$$
\begin{align*}\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline\cdots \newline x_n\end{bmatrix} \rightarrow\begin{bmatrix}a_0^{(2)} \newline a_1^{(2)} \newline a_2^{(2)} \newline\cdots\end{bmatrix} \rightarrow\begin{bmatrix}a_0^{(3)} \newline a_1^{(3)} \newline a_2^{(3)} \newline\cdots\end{bmatrix} \rightarrow \cdots \rightarrow\begin{bmatrix}h_\Theta(x)_1 \newline h_\Theta(x)_2 \newline h_\Theta(x)_3 \newline h_\Theta(x)_4 \newline\end{bmatrix} \rightarrow\end{align*}
$$

마지막 노드 레이어에 g() 로지스틱 함수를 곱해 아래와 같은 최종 결과를 얻을 수 있다. 

$h_\Theta(x) =\begin{bmatrix}0 \newline 0 \newline 1 \newline 0 \newline\end{bmatrix}$

이는 세번째 항목에 해당된다. 

![](http://cfile27.uf.tistory.com/image/21028034586CEC40102AA4 "Multiclass Classification2" "width:600px;height:100px;float:center;padding-left:10px;")

가설함수의 결과는 위 그림과 같이 y중 하나로 표현된다.  