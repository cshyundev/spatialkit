# cv_utils

**현재 버전:** 0.1.0-alpha  
**최근 업데이트:** 2024년 6월 1일  
**상태:** 개발 중 - 이 버전은 아직 개발 중이며 버그가 포함되어 있을 수 있다.

## 라이선스

`cv_utils`는 MIT 라이선스에 따라 자유롭게 이용 가능하다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조한다.

## 소개

`cv_utils`는 컴퓨터 비전 및 로보틱스 분야의 연구 및 개발을 지원하기 위해 설계된 개인 라이브러리다. 이 라이브러리는 3D 비전을 포함한 컴퓨터 비전 알고리즘의 개발 및 테스트에 필요한 다양한 기능과 함수를 제공한다. `cv_utils`는 사용자가 복잡한 데이터를 효과적으로 처리하고 분석할 수 있도록 돕는 도구와 기능을 포함하고 있다.

### 주요 기능 및 특징

- **프로토타이핑 및 연구를 위한 테스트 코드**: 컴퓨터 비전 알고리즘 개발 및 테스트 단계에서 자주 필요한 테스트 코드를 제공한다.
- **파이토치 지원**: 3D 데이터를 처리하고 분석하기 위한 PyTorch의 텐서를 다루는 함수와 클래스를 제공한다.
- **주요 라이브러리 통합**: Numpy, OpenCV, Scipy, PyTorch 같은 유명 라이브러리의 핵심 기능들을 통합하여 간소화된 사용법을 제공한다.

### 추천 대상

- **컴퓨터 비전 및 로보틱스 초심자**: 다른 라이브러리에 비해 코드가 간단하고 이해하기 쉬우며, 특히 3차원 태스크에서의 코드 레벨의 이해가 용이하다.
- **3D 비전 연구자**: 파이토치를 기반으로 한 코드와 테스트 코드를 제공하여, 딥러닝을 포함한 다양한 3D 비전 연구에서 프로그래밍 과정을 단축할 수 있다.

## 주의사항

- **실행 속도 및 효율 문제**: 몇 가지 기능은 기존 OpenCV나 다른 라이브러리보다 느릴 수 있으므로, 최적화 및 속도가 중요한 연구 및 개발 시 사용에 주의해야 한다.

## Getting Started (개발 모드)

### 필요 조건

- **파이썬 버전**: Python3 >= 3.8
- **의존성**: 설치 과정에서 필요한 주요 의존성은 자동으로 설치되나, PyTorch는 사용자의 환경에 따라 다른 버전이 필요할 수 있으므로 별도로 설치하는 것을 권장

### 설치 방법

1. 리포지토리 복제
   ```bash
   git clone https://github.com/cshyundev/cv_utils.git
   ```
2. 개발 모드로 설치
    ```bash
    cd cv_utils
    pip install -e .
    ```

### Conda를 이용한 설치
1. 새 Conda 가상 환경 생성 및 활성화
    ```bash
    conda create --name cv_utils python=3.8
    conda activate cv_utils
   ```
2. 리포지토리 복제 
   ```bash
   git clone https://github.com/cshyundev/cv_utils.git
   ```
2. 개발 모드로 설치
    ```bash
    cd cv_utils
    pip install -e .
    ```

## 제공하는 기능 및 테스트

### 1. 3D Pose 검증
에피폴라 라인(Epipolar line)을 이용해 MVS 데이터셋의 3D 포즈의 정확도를 검증한다.\
[3D Pose 검증 테스트 사용법](docs/tests/3D_pose_verification.md)

### 2. 카메라 캘리브레이션 정확도 테스트 (개발 예정)
카메라 캘리브레이션 결과를 재투영 오차(reprojection error)와 왜곡 보정을 통해 정량적, 정석적으로 평가한다.\
[캘리브레이션 테스트 사용법](docs/tests/homography_transformation.md)

### 3. 이미지 왜곡 보정 변환
카메라로 촬영된 이미지의 왜곡을 보정후 결과를 저장한다.\
[이미지 왜곡 보정 테스트 사용법](docs/tests/image_distortion_correction.md)

### 4. Fiducial Marker 인식률 및 Pose 정확도 테스트 (개발 예정)
Fiducial Marker를 인식하고 인식 결과의 Pose의 정확성을 평가한다.
