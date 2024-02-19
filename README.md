# 딥러닝 기반 파이프 용접부 탐지 모델 개발
    
컴퓨터 비전 영역의 비파괴 검사로 촬영된 파이프 용접부 이미지의 이상탐지 및 분류 모델링을 진행했다.

이것을 고려하여 첫 번째, 약 20만 장의 이미지 데이터를 육안으로 분석하여 딥러닝 모델 학습을 위한 이미지를 정제했다. 그 다음 학습에 유의미한 전처리를 조사하고 적용하였다. 전처리 기법은 Normalization, Histogram Equalization, Median Bluring, Sobel Masking, Noise drop을 적용하였다.
두 번째, 정상 이미지수에 비해서 결함과 비정상 이미지의 수가 매우 적은편이다. 이에 대한 대책으로는 Virtual flaw(가상 결함)알고리즘을 통해 정상 이미지에 가상의 결함을 인공적으로 합성해서 데이터를 증강하여 사용하였다.
세 번째, 다양한 브랜치에 서로 다른 전처리 이미지를 활용하여 동일한 이미지를 서로 다른 측면에서 볼 수 있도록 멀티뷰러닝 모델 아키텍처를 설계하였다.
네 번째, 각 브랜치에는 Vanila CNN, VGG16, ResNet50, InceptionNet, mobileNet 등의 모델들을 활용할 수 있다.
다섯 번째, Optuna를 통한 하이퍼파라미터 튜닝과 MLflow를 통한 실험관리를 진행하였다.

- Multi-View Architecture
![image](https://github.com/ChangZero/mlflow-optuna-multi-view-learning/assets/97018869/25428e7e-e379-42ae-85da-b004eabfc876)
