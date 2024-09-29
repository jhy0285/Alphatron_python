# # # 파이썬에서 단순 값들의 오름차순, 내림차순 우선순위 커스텀이 필요할 경우, 가장 직관적이고 간단한 방법
# # import heapq
# #
# # #원본 데이터
# # data = [[2, 0, 10],[5, 0, 10], [1, 5, 5], [3, 5, 3], [3, 12, 2]]
# #
# # #우선순위 기준을 커스텀한 튜플을 만들기 위해 새로운 리스트 생성
# # custom_priority_data = [(-item[2], -item[0], item[1], item) for item in data]
# #
# # #heapify를 사용하여 커스텀 우선순위 큐 생성
# # heapq.heapify(custom_priority_data)
# #
# # #우선순위 큐에서 요소를 하나씩 꺼내 출력
# # while custom_priority_data:
# #     print(heapq.heappop(custom_priority_data)[3])
# #
# #
# # #이 경우 우선순위는 다음과 같다
# # # 1. 리스트의 세번째 값을 내림차순으로 => 즉 큰 값이 더 우선됨
# # # 2. 그다음 리스트의 첫번쨰 값을 내림차순으로 => 즉 큰 값이 더 우선됨
# # # 3. 그다음 리스트의 두번쨰 값을 오름차순으로 => 즉 작은 값이 더 우선됨
#
# import pandas as pd
# import numpy as np
#
# # Boston 데이터셋 URL로부터 데이터 로드
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
#
# # 데이터 전처리: 첫 번째 행과 두 번째 행을 각각 데이터와 타겟으로 나눔
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # Feature 데이터
# target = raw_df.values[1::2, 2]  # Target 데이터 (MEDV)
#
# # 데이터프레임 생성
# columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
# boston_df = pd.DataFrame(data, columns=columns)
#
# # 집값(MEDV) 컬럼 추가
# boston_df['MEDV'] = target
#
# # tax와 medv(집값)의 상관계수를 구하기
# correlation = boston_df['TAX'].corr(boston_df['MEDV'])
#
# # 결과 출력
# print(f"TAX와 MEDV의 상관계수: {correlation:.4f}")
import numpy as np

a=np.array([1,2,3,4])

print(a)