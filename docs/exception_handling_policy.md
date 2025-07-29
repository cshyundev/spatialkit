# CV Utils 예외 처리 정책 및 가이드라인

**최종 수정 날짜:** 2025년 7월 29일  
**버전:** 1.0.0

이 문서는 cv_utils 라이브러리의 예외 처리 정책과 사용 가이드라인을 정의합니다.

## 📋 예외 처리 정책

### 1. 기본 원칙

- **Fail Fast**: 잘못된 입력이나 상태를 빠르게 감지하여 즉시 예외 발생
- **명확한 메시지**: 사용자가 문제를 이해하고 해결할 수 있는 구체적인 오류 메시지 제공
- **계층적 구조**: 도메인별로 예외를 분류하여 선택적 예외 처리 가능
- **일관성**: 유사한 상황에서는 동일한 예외 타입 사용

### 2. 예외 vs Assert 사용 기준

| 상황 | 사용할 것 | 이유 |
|------|-----------|------|
| 사용자 입력 검증 | `raise Exception` | 프로덕션에서도 항상 검증 필요 |
| 타입/형태 검증 | `raise Exception` | 라이브러리 사용자를 위한 명확한 피드백 |
| 내부 논리 검증 | `assert` (개발 시에만) | 개발자 가정 확인용 |
| 외부 의존성 오류 | `raise Exception` | 사용자가 처리 가능한 오류 |

### 3. 로깅 vs 예외 사용 기준

| 상황 | 사용할 것 | 예시 |
|------|-----------|------|
| 치명적 오류 | `raise Exception` | 잘못된 입력, 계산 실패 |
| 경고성 메시지 | `logger.warning` | 성능 저하, 권장사항 |
| 정보성 메시지 | `logger.info` | 처리 완료, 진행 상황 |
| 디버그 정보 | `logger.debug` | 내부 상태, 중간 결과 |

## 🏗️ 예외 계층 구조

```
CVUtilsError (기본 예외)
├── MathError (수학 연산)
│   ├── InvalidDimensionError (차원 오류)
│   ├── InvalidShapeError (형태 불일치)
│   ├── IncompatibleTypeError (타입 불일치)
│   ├── NumericalError (수치 계산 오류)
│   └── SingularMatrixError (특이 행렬)
├── GeometryError (기하학 연산)
│   ├── ConversionError (좌표 변환 오류)
│   ├── InvalidCoordinateError (잘못된 좌표)
│   ├── ProjectionError (투영 오류)
│   └── CalibrationError (캘리브레이션 오류)
├── CameraError (카메라 관련)
│   ├── InvalidCameraParameterError (잘못된 카메라 파라미터)
│   ├── UnsupportedCameraTypeError (지원되지 않는 카메라 타입)
│   └── CameraModelError (카메라 모델 오류)
├── VisualizationError (시각화)
│   ├── RenderingError (렌더링 오류)
│   └── DisplayError (표시 오류)
├── IOError (입출력)
│   ├── FileNotFoundError (파일 없음)
│   ├── FileFormatError (잘못된 파일 형식)
│   └── ReadWriteError (읽기/쓰기 오류)
├── MarkerError (마커 관련)
│   ├── MarkerDetectionError (마커 탐지 실패)
│   └── InvalidMarkerTypeError (잘못된 마커 타입)
└── MVSError (다중 뷰 스테레오)
    ├── DatasetError (데이터셋 오류)
    └── ReconstructionError (재구성 오류)
```

## 📖 사용 가이드라인

### 1. 라이브러리 개발자용

#### Assert 제거 규칙
```python
# ❌ Before (잘못된 사용)
def qr(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Expected 2D matrix, got {x.shape}"
    return np.linalg.qr(x)

# ✅ After (올바른 사용)  
def qr(x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    if x.ndim != 2:
        raise InvalidDimensionError(f"QR decomposition requires 2D matrix, got {x.ndim}D array with shape {x.shape}")
    
    try:
        return np.linalg.qr(x)
    except Exception as e:
        raise NumericalError(f"QR decomposition failed: {e}") from e
```

#### 로깅 정리 규칙
```python
# ❌ Before (LOG_ERROR 남용)
def convert_coordinates(x):
    if x.shape[0] not in [2, 3]:
        LOG_ERROR(f"Invalid shape: {x.shape}")
        raise ValueError("Invalid input")

# ✅ After (명확한 예외)
def convert_coordinates(x):
    if x.shape[0] not in [2, 3]:
        raise InvalidCoordinateError(f"Expected 2D or 3D coordinates, got shape {x.shape}")
```

#### Docstring 업데이트 규칙
```python
def matrix_operation(x: ArrayLike) -> ArrayLike:
    """
    Perform matrix operation on input.

    Args:
        x (ArrayLike): Input matrix.

    Returns:
        ArrayLike: Result of operation.
        
    Raises:
        InvalidDimensionError: If input is not a 2D matrix.
        InvalidShapeError: If matrix is not square (for operations requiring square matrices).
        NumericalError: If computation fails due to numerical issues.
        IncompatibleTypeError: If input type is not supported.
        
    Example:
        >>> result = matrix_operation(np.eye(3))
        >>> result.shape
        (3, 3)
    """
```

#### 모듈별 리팩토링 패턴

**Operations 모듈 (ops/)**
```python
# Before
def qr(x: ArrayLike) -> ArrayLike:
    assert x.ndim == 2, f"Expected 2D matrix, got {x.shape}"
    return np.linalg.qr(x)

# After  
def qr(x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """QR decomposition with proper exception handling."""
    from ..exceptions import InvalidDimensionError, NumericalError
    
    if x.ndim != 2:
        raise InvalidDimensionError(
            f"QR decomposition requires 2D matrix, got {x.ndim}D array with shape {x.shape}. "
            f"Please ensure input is a 2D matrix."
        )
    
    try:
        if is_tensor(x):
            return torch.linalg.qr(x)
        return np.linalg.qr(x)
    except Exception as e:
        raise NumericalError(f"QR decomposition failed: {e}") from e
```

**Camera 모듈 (geom/camera.py)**
```python
# Before
def __init__(self, K, image_size):
    assert K.shape == (3, 3), f"Invalid K matrix shape: {K.shape}"
    self.K = K

# After
def __init__(self, K, image_size):
    """Initialize camera with proper validation.""" 
    from ..exceptions import InvalidCameraParameterError
    
    if not isinstance(K, np.ndarray) or K.shape != (3, 3):
        raise InvalidCameraParameterError(
            f"Camera matrix K must be 3x3 numpy array, got {type(K)} with shape {getattr(K, 'shape', 'unknown')}. "
            f"Please provide a valid 3x3 intrinsic matrix."
        )
    
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise InvalidCameraParameterError(
            f"Focal lengths must be positive, got fx={K[0,0]}, fy={K[1,1]}. "
            f"Please check your camera calibration."
        )
    
    self.K = K
```

**I/O 모듈 (utils/io.py)**
```python
# Before 
def read_image(path):
    if not os.path.exists(path):
        LOG_ERROR(f"File not found: {path}")
        return None

# After
def read_image(path):
    """Read image with proper exception handling."""
    from ..exceptions import FileNotFoundError, FileFormatError
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image file not found: {path}. "
            f"Please check the file path and ensure the file exists."
        )
    
    try:
        return cv2.imread(path)
    except Exception as e:
        raise FileFormatError(
            f"Failed to read image {path}: {e}. "
            f"Please ensure the file is a valid image format."
        ) from e
```

### 2. 라이브러리 사용자용

#### 예외 처리 패턴

**세밀한 예외 처리 (권장)**
```python
import cv_utils as cvu
from cv_utils.exceptions import InvalidDimensionError, NumericalError, MathError

try:
    result = cvu.umath.qr(matrix)
except InvalidDimensionError as e:
    print(f"입력 차원 오류: {e}")
    # 차원 수정 후 재시도
except NumericalError as e:
    print(f"수치 계산 실패: {e}")
    # 다른 알고리즘 시도
except MathError as e:
    print(f"수학 연산 오류: {e}")
    # 전반적인 수학 오류 처리
```

**카테고리별 예외 처리**
```python
try:
    # 여러 수학 연산
    result1 = cvu.umath.svd(matrix1)
    result2 = cvu.umath.inv(matrix2) 
    result3 = cvu.umath.solve(A, b)
except MathError as e:
    print(f"수학 연산 중 오류 발생: {e}")
    # 모든 수학 연산 오류를 일괄 처리
```

**라이브러리 전체 예외 처리**
```python
from cv_utils.exceptions import CVUtilsError

try:
    # cv_utils의 모든 기능 사용
    cam = cvu.camera.PerspectiveCamera(...)
    points = cvu.geom_utils.triangulate_points(...)
    pcd = cvu.vis3d.create_point_cloud(...)
except CVUtilsError as e:
    print(f"cv_utils 라이브러리 오류: {e}")
    # cv_utils 관련 모든 오류 처리
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        # 다른 라이브러리나 시스템 오류
```

##  예외 메시지 작성 가이드라인

### 좋은 예외 메시지의 특징
1. **구체적**: 무엇이 잘못되었는지 명확히 설명
2. **실행 가능**: 사용자가 어떻게 해결할지 알 수 있음
3. **컨텍스트 포함**: 관련 값이나 상태 정보 포함
4. **일관된 형식**: 유사한 상황에서 일관된 메시지 형식

### 예시
```python
# ❌ 나쁜 예
raise ValueError("Invalid input")

# ✅ 좋은 예  
raise InvalidDimensionError(
    f"QR decomposition requires 2D matrix, got {x.ndim}D array with shape {x.shape}. "
    f"Please ensure input is a 2D matrix."
)

# ✅ 더 좋은 예 (해결책 포함)
raise InvalidShapeError(
    f"Matrix multiplication requires compatible shapes: A({A.shape}) @ B({B.shape}). "
    f"Expected A.shape[-1] == B.shape[0], but got {A.shape[-1]} != {B.shape[0]}. "
    f"Consider reshaping or transposing one of the matrices."
)
```