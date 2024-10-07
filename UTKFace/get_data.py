import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

def get_data(base_dir):
    images = []
    ages = []
    genders = []

    image_dir = base_dir

    # 디렉터리 내 파일들을 순회하며 전처리 수행
    for i in os.listdir(image_dir):
        split = i.split('_')
        ages.append(int(split[0]))
        genders.append(int(split[1]))

        image_path = os.path.join(image_dir, i)

        # 이미지 열고 처리
        with Image.open(image_path) as img:
            images.append(img.copy())
    
    # Pandas Series로 변환
    images = pd.Series(images, name='Images')
    ages = pd.Series(ages, name='Ages')
    genders = pd.Series(genders, name='Genders')

    # 데이터프레임으로 병합
    df = pd.concat([images, ages, genders], axis=1)

    # 나이 4세 미만 샘플 추출
    under4s = df[df['Ages'] < 4].sample(frac=0.3)
    df = df[df['Ages'] >= 4]  # 4세 이상의 데이터만 남김
    df = pd.concat([df, under4s], ignore_index=True)  # 다시 합침

    # 최소 2개 이상의 데이터가 있는 그룹만 사용
    df = df.groupby("Genders").filter(lambda x:len(x) > 1)
    df = df.groupby("Ages").filter(lambda x:len(x) > 1)

    # 나이 80세 미만으로 필터링
    df = df[df['Ages'] < 80]

    X = []
    y = []

    for i in range(len(df)):
        image = df['Images'].iloc[i]

        if isinstance(image, Image.Image):
            image = image.resize((200, 200), Image.LANCZOS).convert('RGB')  # 이미지 크기 변경
            ar = np.asarray(image)
            X.append(ar)  # 이미지 배열 추가
            age_gender = [int(df['Ages'].iloc[i]), int(df['Genders'].iloc[i])]
            y.append(age_gender)  # 나이와 성별 추가
        else:
            print(f"이미지가 올바르지 않음: {i}")
    
    X = np.array(X)
    print(f"X 배열 크기: {X.shape}")

    # 나이와 성별로 데이터 나누기
    y_age = df['Ages']
    y_gender = df['Genders']

    # 데이터 클래스 확인 (분포가 충분한지 체크)
    print("성별 데이터 분포:\n", y_gender.value_counts())
    print("나이 데이터 분포:\n", y_age.value_counts())
    
    # train/test 분리
    if len(y_age.unique()) > 1:  # 최소 2개 이상의 나이 그룹이 있을 때만 stratify 사용
        x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(X, y_age, test_size=0.2, stratify=y_age)
    else:
        x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(X, y_age, test_size=0.2)
        
    if len(y_gender.unique()) > 1:  # 최소 2개 이상의 성별 그룹이 있을 때만 stratify 사용
        x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(X, y_gender, test_size=0.2, stratify=y_gender)
    else:
        x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(X, y_gender, test_size=0.2)
    
    return (x_train_age, x_test_age, y_train_age, y_test_age), (x_train_gender, x_test_gender, y_train_gender, y_test_gender)