---
title: Week 1 Introduction
category: CourseraMachineLearning
excerpt: |
  머신러닝이란?
feature_text: |
  ## [Coursera] Machine Learning
  Andrew Ng의 Machine Learning 강의 정리
  ref: https://www.coursera.org/learn/machine-learning/home/week/1
  code: https://github.com/fabulousjeong/ML-OctaveCode/
feature_image: "https://cdn-images-1.medium.com/max/1600/1*_eGKYpUT6dFjOPLHVLDsFw.png"
image: "https://cdn-images-1.medium.com/max/1600/1*_eGKYpUT6dFjOPLHVLDsFw.png"
---
###ML: 서론
머신러닝이란?
머신러닝에 관한 2가지 정의가 있다. 첫 번째는 Arthur Samuel의 정의인 "구체적인 프로그래밍의 도움 없이 학습하는 능력을 컴퓨터에게 주는 학문"이다. 두 번째는 Tom Mitchell이 제안한 보다 현대적인 관점의 정의다. 여기서 머신러닝이란 "어떤 경험(experience) E와 관련된 작업(Task) T, 성능(performance)P가 있을 때 경험 E로 부터 작업 T의 성능 P를 증가시키는 학습 컴퓨터 프로그램"을 말한다.
예) 체스 게임
E = 체스 기사들의 체스 게임 경험
T = 체스 게임의 시행
P = 다음 게임에서 이길 확률
{% include figure.html image="https://raw.githubusercontent.com/mahmoudparsian/data-algorithms-book/master/misc/machine_learning.jpg" width="800" height="400" %}
ref: https://github.com/mahmoudparsian/data-algorithms-book/

위의 그림을 보면 머신러닝이란 주어진 데이터(입력, 결과)를 사용하여 알고리즘(예측모델)을 얻어내는 것 이라고 볼 수 있다. 일반적으로, 많은 머신러닝 문제는 지도학습(supervised learning)과 자율학습(unsupervised learning) 두 가지로 나뉜다.

#####지도학습
{% include figure.html image="https://raw.githubusercontent.com/mahmoudparsian/data-algorithms-book/master/misc/machine_learning.jpg" caption="Image with caption" width="300" height="800" %}
![](https://raw.githubusercontent.com/mahmoudparsian/data-algorithms-book/master/misc/machine_learning.jpg "" "width:600px;height:400px;float:center;padding-left:10px;")