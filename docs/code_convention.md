# 프로젝트 코딩 컨벤션 문서

**최종 수정 날짜:** 2025년 7월 29일  
**버전:** 0.2.1-alpha

이 문서는 프로젝트의 코딩 컨벤션을 정의한다.

## 네이밍 규칙

### 패키지
- 패키지 이름은 모두 소문자로 작성한다.
- 단어 간 구분은 언더스코어(_)를 사용하지 않고, 가능한 짧고 명확한 이름을 사용한다.

### 모듈
- 모듈 파일 이름은 모두 소문자로 작성하며, 단어 간 구분에는 언더스코어(_)를 사용한다.
- 모듈 이름은 해당 모듈의 기능을 명확히 나타내야 한다.

### 클래스
- 클래스 이름은 파스칼 케이스(PascalCase)를 사용한다.
- 클래스 이름은 해당 클래스의 역할과 기능을 명확히 나타내야 한다.
- 클래스 내부에서만 사용하는 변수 및 함수는 언더스코어(_)를 붙여 구분한다.
-  

### 변수
- 변수 이름은 스네이크 케이스(snake_case)를 사용한다.
- 변수 이름은 해당 변수의 역할과 기능을 명확히 나타내야 한다.
- 상수 이름은 모두 대문자로 작성하며, 단어 간 구분에는 언더스코어(_)를 사용한다.

### 함수
- 함수 이름은 스네이크 케이스(snake_case)를 사용한다.
- 함수 이름은 해당 함수의 역할과 기능을 명확히 나타내야 한다.

## 주석 스타일

### 함수
- 함수의 주석은 아래 예시에 맞춰 작성한다.
- 함수의 동작이 간단하거나 자명할 경우 생략할 수 있다.
- 주석의 위치는 함수의 몸체 상단에 위치한다.
#### 구성
1. 간단한 설명
2. 함수의 인자(Arguments)
3. 함수의 반환(Returns)
4. 자세한 동작 원리 (Option)
5. 함수 사용 예시 (Option)

```python
def function(...):
    """
        Brief description of what the function does.

        Args:
            param1 (type, size): Description of param1.
            param2 (type, size): Description of param2.
            ...
            paramN (type, size): Description of paramN.

        Returns:
            return_type (type, size): Description of the return value.

        Details:
            - Explanation of the first key detail.
            - Explanation of the second key detail.
            - ...
            - Explanation of the nth key detail.

        Example:
            example_code = function_name(param1_value, param2_value, ...)
    """
    ...
```
### 클래스
- 클래스의 주석은 아래 예시에 맞춰 작성한다.
- 주석의 위치는 `__init__` 함수 상단에 위치한다.
#### 구성
1. 간단한 설명
2. 클래스의 멤버 변수(Attributes)
3. 추상 메소드 (Option)

```python
class Class
    """
        Brief description of what the class does.

        Attributes:
            attr1 (type, size): Description of attr1.
            attr2 (type, size): Description of attr2.
            ...
            attrN (type, size): Description of attrN.

        Abstract Methods:
            abstract_method1: Description of abstract_method1.
            abstract_method1: Description of abstract_method2.
            ...
            abstract_methodN: Description of abstract_methodN.  
    """
    def __init__(self,...): 
```

### 모듈
- 모듈의 주석은 아래 예시에 맞춰 작성한다.
- 모듈의 위치는 파일 최상단에 위치한다.
#### 구성
1. 모듈 이름
2. 모듈에 대한 간단한 설명
3. 모듈의 주요 기능 및 지원하는 타입(Option)
4. 저자 정보
5. 라이센스 정보
6. 모듈 사용 예시
```python
"""
Module Name: module.py

Description:
Brief Description

Main Functions:
...


Author: name
Email: email@email.com
Version: x.x.x

License: License

Usage:
...
"""
```