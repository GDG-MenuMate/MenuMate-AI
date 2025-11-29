import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from datetime import datetime, time
import redis
import uvicorn
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# LangChain 및 OpenAI 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# --- 1. Pydantic 모델 정의 (기존과 동일) ---

# 다이어트 정보 모델
class DietInfo(BaseModel):
    height: Optional[int] = Field(default=None, description="키 (cm)")
    weight: Optional[int] = Field(default=None, description="몸무게 (kg)")


# 가격 범위 모델
class PriceRangeInput(BaseModel):
    minPrice: Optional[int] = Field(default=None, description="최소 가격")
    maxPrice: Optional[int] = Field(default=None, description="최대 가격")


# 사용자 입력 모델
class UserInput(BaseModel):
    category: str = Field(description="카테고리: 'diet', 'vegan', 'low_sugar', 'muslim' 중 하나")
    diet: Optional[DietInfo] = Field(default=None, description="다이어트 정보 (category='diet'일 때만)")
    meals: List[str] = Field(description="추천받을 식사: ['breakfast', 'lunch', 'dinner'] 중 선택")
    priceRange: PriceRangeInput = Field(description="가격 범위")
    prompt: str = Field(description="사용자 요청사항 (예: '국수 말고 밥')")
    location: List[str] = Field(default=[], description="위치: ['science_campus', 'humanities_campus'] 중 선택")


# UserPreferences: 내부 처리용으로 변환된 모델
class UserPreferences(BaseModel):
    user_id: Optional[int] = None
    budget: Optional[int] = Field(default=10000, description="최대 예산 (원)")
    min_budget: Optional[int] = Field(default=None, description="최소 예산 (원)")
    preferred_categories: List[str] = Field(default=[], description="선호 카테고리 (사용 안 함. 'location'으로 대체됨)")
    meal_type: str = Field(default="점심", description="식사 시간대: 아침/점심/저녁 (시간 필터링에 사용)")
    target_meals: List[str] = Field(default=["lunch"], description="['morning', 'lunch', 'dinner']")
    user_prompt: str = Field(default="", description="사용자 세부 요청")

    # 영양 관련 선호도
    prefer_high_protein: bool = Field(default=False, description="고단백 선호 (비활성화)")
    prefer_low_calorie: bool = Field(default=False, description="저칼로리 선호 (유일하게 사용)")
    prefer_low_sodium: bool = Field(default=False, description="저나트륨 선호 (비활성화)")

    # 사용자의 '목표' 카테고리 (예: 'vegan', 'muslim')
    category: Optional[str] = Field(default=None, description="카테고리")
    height: Optional[int] = Field(default=None, description="키")
    weight: Optional[int] = Field(default=None, description="몸무게")
    location: List[str] = Field(default=[], description="위치 정보 (캠퍼스 필터링에 사용)")


# AI A와 B가 공통으로 사용하는 모델들 (기존과 동일)
class MenuCandidate(BaseModel):
    restaurant_name: str
    menu_name: str
    price: int
    base_score: float
    tags: List[str]


class PriceRange(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None


class RecommendationRequest(BaseModel):
    candidates: List[MenuCandidate]
    user_prompt: str
    price: Optional[PriceRange] = Field(default=None, description="사용자 가격 제한")
    target_meals: List[str] = Field(description="추천받고자 하는 끼니 목록")
    conversation_history: Optional[List[str]] = []


class RecommendedMenu(BaseModel):
    restaurant_name: str = Field(description="식당 이름")
    menu_name: str = Field(description="메뉴 이름")
    price: int = Field(description="메뉴의 실제 가격")
    justification: str = Field(description="이 메뉴를 추천하는 이유 (LLM이 작성)")
    new_score: float = Field(description="LLM이 재평가한 최종 점수 (0.0 ~ 1.0)")
    reason_hashtags: List[str] = Field(description="추천 이유를 요약하는 3-5개의 해시태그")


class FinalRecommendation(BaseModel):
    morning: Optional[RecommendedMenu] = Field(default=None, description="추천 아침 메뉴")
    lunch: Optional[RecommendedMenu] = Field(default=None, description="추천 점심 메뉴")
    dinner: Optional[RecommendedMenu] = Field(default=None, description="추천 저녁 메뉴")


# --- 2. 입력 변환 함수 (기존과 동일) ---

def convert_user_input(user_input: UserInput) -> UserPreferences:
    """
    (수정됨) 프론트엔드 JSON 형식을 내부 처리용 UserPreferences로 변환
    - 'location' 필드를 'preferred_categories'가 아닌 'location' 필드에 그대로 전달
    """
    # 식사 타입 변환
    meal_mapping = {
        "BREAKFAST": "morning",
        "LUNCH": "lunch",
        "DINNER": "dinner"
    }
    target_meals = [meal_mapping.get(m.upper(), m.lower()) for m in user_input.meals]

    meal_type_mapping = {
        "morning": "아침",
        "lunch": "점심",
        "dinner": "저녁"
    }
    meal_type = meal_type_mapping.get(target_meals[0], "점심") if target_meals else "점심"

    # 'DIET'일 때만 저칼로리 플래그 활성화
    prefer_low_calorie = False
    if user_input.category.upper() == "DIET":
        prefer_low_calorie = True

    prefer_high_protein = False
    prefer_low_sodium = False

    preferred_categories = []

    return UserPreferences(
        budget=user_input.priceRange.maxPrice,
        min_budget=user_input.priceRange.minPrice,
        preferred_categories=preferred_categories,  # <- 이제 사용 안 함
        meal_type=meal_type,
        target_meals=target_meals,
        user_prompt=user_input.prompt,
        prefer_high_protein=prefer_high_protein,
        prefer_low_calorie=prefer_low_calorie,
        prefer_low_sodium=prefer_low_sodium,
        category=user_input.category,  # <-- 'VEGAN', 'MUSLIM' 등 전달
        height=user_input.diet.height if user_input.diet else None,
        weight=user_input.diet.weight if user_input.diet else None,
        location=user_input.location  # <-- 'science_campus' 등을 그대로 전달
    )


# --- 3. 데이터 전처리 클래스 (기존과 동일) ---

class MenuDataPreprocessor:
    """
    (수정됨) '열량', '가격', '태그', '카테고리', '평점', '영업시간', '캠퍼스' 기반 전처리
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.menu_features = None
        self.menu_df = None
        self.db_engine = self._create_db_engine()

    def _create_db_engine(self):
        """
        .env 파일의 정보로 SQLAlchemy DB 엔진 생성
        """
        load_dotenv()
        db_user = os.environ.get("DB_USER")
        db_pass = os.environ.get("DB_PASSWORD")
        db_host = os.environ.get("DB_HOST")
        db_port = os.environ.get("DB_PORT")
        db_name = os.environ.get("DB_NAME")

        if not all([db_user, db_pass, db_host, db_port, db_name]):
            print("경고: DB 접속 정보(.env)가 불완전합니다. DB 로드에 실패할 수 있습니다.")
            return None

        DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

        try:
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ DB 연결 성공")
            return engine
        except Exception as e:
            print(f"❌ DB 엔진 생성 또는 연결 테스트 실패: {e}")
            return None

    def load_nutrition_data(self):
        """
        (디버깅 모드) 데이터 로드 및 전처리 과정의 에러를 추적합니다.
        """
        print("🔄 [1단계] DB 데이터 로드 시작...")

        if self.db_engine is None:
            print("❌ 오류: DB 엔진 없음. 샘플 데이터 사용.")
            df = self._get_sample_data()
        else:
            # 최종 쿼리 (카테고리 JOIN 제거됨)
            sql_query = """
            SELECT
                m.name AS "제품명",
                r.name AS "식당명",
                m.price AS "가격",
                m.calories AS "열량",
                m.tags AS "태그",          
                r.rating AS "평점",
                r.open_time AS "오픈시간",
                r.close_time AS "마감시간",
                r.campus AS "캠퍼스",
                '' AS "카테고리"
            FROM menus m
            JOIN restaurants r ON m.restaurants_id = r.restaurants_id
            """
            try:
                df = pd.read_sql(text(sql_query), self.db_engine)
                print(f"✅ DB 데이터 로드 성공! ({len(df)}개 메뉴 가져옴)")
            except Exception as e:
                print(f"❌ DB 로드 실패: {e}")
                df = self._get_sample_data()

        # --- 전처리 디버깅 시작 ---
        print("🔄 [2단계] 데이터 전처리 시작...")
        try:
            # 1. 시간 변환
            # (이미 time 객체일 수도 있고 문자열일 수도 있어서 안전하게 처리)
            def safe_time_convert(x):
                if pd.isna(x): return None
                if isinstance(x, time): return x  # 이미 시간 객체면 그대로
                try:
                    return pd.to_datetime(str(x), format='%H:%M:%S').time()
                except:
                    return None

            df['오픈시간'] = df['오픈시간'].apply(safe_time_convert)
            df['마감시간'] = df['마감시간'].apply(safe_time_convert)
            print("   - 시간 데이터 변환 완료")

            # 2. 평점 처리
            df['평점'] = pd.to_numeric(df['평점'], errors='coerce').fillna(3.0)
            print("   - 평점 데이터 처리 완료")

            # 3. 캠퍼스 처리 (리스트 -> 문자열)
            def clean_campus(x):
                if isinstance(x, list):
                    return str(x[0]) if len(x) > 0 else '정보없음'
                if pd.isna(x):
                    return '정보없음'
                return str(x)

            df['캠퍼스'] = df['캠퍼스'].apply(clean_campus)
            print("   - 캠퍼스 데이터 처리 완료")

            # 4. 열량 처리
            df['열량'] = pd.to_numeric(df['열량'], errors='coerce')
            df['열량'] = df['열량'].fillna(df['열량'].median())
            print("   - 열량 데이터 처리 완료")

            # 5. 태그 처리
            if '태그' in df.columns:
                # DB에서 문자열('밥,국')로 올 수도 있고 리스트(['밥','국'])로 올 수도 있음
                df['태그'] = df['태그'].apply(
                    lambda x: x if isinstance(x, list) else (str(x).split(',') if pd.notna(x) and x else [])
                )
            else:
                df['태그'] = [[] for _ in range(len(df))]
            print("   - 태그 데이터 처리 완료")

            # 6. 카테고리 채우기
            if '카테고리' in df.columns:
                df['카테고리'] = df['태그']
            print("   - 카테고리 데이터 처리 완료")

            # 7. 최종 저장
            self.menu_df = df
            print(f"✅ [완료] 최종 메뉴 데이터 준비됨: {len(df)}개")
            return df

        except Exception as e:
            print(f"❌ [치명적 오류] 전처리 중 에러 발생: {e}")
            import traceback
            traceback.print_exc()  # 자세한 에러 위치 출력
            return None

    def _get_sample_data(self):
        """
        DB 연결 실패 시 사용하는 샘플 데이터 (평점, 시간, 캠퍼스 추가)
        """
        print("샘플 데이터를 사용합니다...")
        return pd.DataFrame([
            {'제품명': '김치찌개', '식당명': '한식당 A', '가격': 8000, '열량': 480, '카테고리': '한식,인문계캠퍼스', '태그': '밥,국물,돼지고기', '평점': 4.2,
             '오픈시간': '10:00:00', '마감시간': '22:00:00', '캠퍼스': '인문계캠퍼스'},
            {'제품명': '비건 비빔밥', '식당명': '한식당 A', '가격': 8000, '열량': 550, '카테고리': '한식,인문계캠퍼스,채식', '태그': '밥,야채,비건', '평점': 4.2,
             '오픈시간': '10:00:00', '마감시간': '22:00:00', '캠퍼스': '인문계캠퍼스'},
            {'제품명': '제육볶음', '식당명': '한식당 B', '가격': 8500, '열량': 620, '카테고리': '한식,자연계캠퍼스', '태그': '밥,고기,돼지고기', '평점': 4.0,
             '오픈시간': '09:00:00', '마감시간': '21:00:00', '캠퍼스': '자연계캠퍼스'},
            {'제품명': '심야 라멘', '식당명': '일식당 C', '가격': 9000, '열량': 650, '카테고리': '일식,자연계캠퍼스', '태그': '면,국물', '평점': 4.5,
             '오픈시간': '18:00:00', '마감시간': '02:00:00', '캠퍼스': '자연계캠퍼스'},  # 야간 영업
            {'제품명': '치킨 케밥', '식당명': '케밥 전문점', '가격': 7000, '열량': 310, '카테고리': '기타,자연계캠퍼스,무슬림', '태그': '빵,가벼운,할랄',
             '평점': 4.8, '오픈시간': '11:00:00', '마감시간': '20:00:00', '캠퍼스': '자연계캠퍼스'},
            {'제품명': '저당 비건 샌드위치', '식당명': '샌드위치 전문점', '가격': 7500, '열량': 372, '카테고리': '양식,인문계캠퍼스,채식,저당', '태그': '샌드위치,비건',
             '평점': 4.1, '오픈시간': '08:00:00', '마감시간': '17:00:00', '캠퍼스': '인문계캠퍼스'},  # 아침 가능
        ])

    def extract_features(self):
        """
        '평점'을 수치형 피처에, '캠퍼스'를 더미 피처에 추가
        """
        if self.menu_df is None:
            print("오류: 메뉴 데이터가 로드되지 않았습니다.")
            return None

        df = self.menu_df.copy()

        numeric_features = ['열량', '가격', '평점']

        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
            else:
                print(f"경고: '{col}' 컬럼이 없어 피처 추출에서 제외됩니다.")

        numeric_features = [col for col in numeric_features if col in df.columns]

        if '열량' in df.columns:
            df['칼로리_레벨'] = pd.cut(df['열량'], bins=[-np.inf, 400, 600, np.inf], labels=[0, 1, 2], right=True)

        if '가격' in df.columns:
            df['가격_레벨'] = pd.cut(df['가격'], bins=[-np.inf, 7000, 9000, np.inf], labels=[0, 1, 2], right=True)

        # '캠퍼스' (단일 값) 원-핫 인코딩
        if '캠퍼스' in df.columns:
            campus_dummies = pd.get_dummies(df['캠퍼스'], prefix='캠퍼스')
            df = pd.concat([df, campus_dummies], axis=1)

        # '카테고리' (다중 값) 원-핫 인코딩
        if '카테고리' in df.columns:
            all_categories = set(cat for cats in df['카테고리'] for cat in cats if cat)
            for cat in all_categories:
                df[f'카테고리_{cat}'] = df['카테고리'].apply(lambda x: 1 if cat in x else 0)

        # '태그' (다중 값) 원-핫 인코딩
        if '태그' in df.columns:
            all_tags = set(tag for tags in df['태그'] for tag in tags if tag)
            for tag in all_tags:
                df[f'태그_{tag}'] = df['태그'].apply(lambda x: 1 if tag in x else 0)

        feature_cols = numeric_features
        feature_cols = [col for col in feature_cols if col in df.columns]

        if feature_cols:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            print("경고: 정규화할 수치형 피처가 없습니다.")

        tag_cols = [col for col in df.columns if col.startswith('태그_')]
        category_cols = [col for col in df.columns if col.startswith('카테고리_') or col.startswith('캠퍼스_')]
        level_cols = ['칼로리_레벨', '가격_레벨']
        level_cols = [col for col in level_cols if col in df.columns]

        all_feature_cols = feature_cols + tag_cols + category_cols + level_cols
        all_feature_cols = [col for col in all_feature_cols if col in df.columns]

        self.menu_features = df[all_feature_cols].values
        print(f"✅ 피처 추출 완료: {self.menu_features.shape[1] if isinstance(self.menu_features, np.ndarray) else 0}개 피처")
        return df

    def calculate_scores(self, user_preferences: UserPreferences):
        """
        (수정됨) '저당(LOW_SUGAR)' 필터 제거, 'VEGAN', 'MUSLIM' 필터 버그 수정
        """
        if self.menu_df is None:
            raise ValueError("메뉴 데이터가 로드되지 않았습니다.")

        df = self.menu_df.copy()
        scores = np.ones(len(df))  # 1점으로 시작

        # --- [1. 강력한 필터 로직 (Hard Filters)] ---

        # (A) [수정됨] VEGAN/MUSLIM 필터 (DB '카테고리' 컬럼 검사)
        if user_preferences.category and '카테고리' in df.columns:
            category_upper = user_preferences.category.upper()

            if category_upper == "VEGAN":  # <-- UserInput의 'vegan'
                is_not_vegan = df['카테고리'].apply(
                    lambda cats: '채식' not in cats and '비건' not in cats  # DB의 '채식'
                )
                scores[is_not_vegan] = 0.0

            if category_upper == "MUSLIM":  # <-- UserInput의 'muslim'
                is_not_muslim = df['카테고리'].apply(
                    lambda cats: '무슬림' not in cats and '할랄' not in cats  # DB의 '무슬림'
                )
                scores[is_not_muslim] = 0.0

                if '태그' in df.columns:
                    has_pork = df['태그'].apply(lambda tags: '돼지고기' in tags)
                    scores[has_pork] = 0.0

                    # [제거됨] LOW_SUGAR (저당) 필터
            # if category_upper == "LOW_SUGAR":
            #    ...

        # (B) Location(Campus) 필터 (DB '캠퍼스' 컬럼 검사)
        if user_preferences.location and '캠퍼스' in df.columns:
            location_map = {
                'science_campus': '자연계캠퍼스',
                'humanities_campus': '인문계캠퍼스'
            }
            target_campuses = [location_map.get(loc) for loc in user_preferences.location if location_map.get(loc)]

            if target_campuses:
                # [수정] "둘다" 케이스 고려 (v2.9 제안 방식)
                is_match = df['캠퍼스'].apply(
                    lambda menu_campus: (menu_campus in target_campuses) or (menu_campus == '둘다')
                )
                scores[~is_match] = 0.0  # 일치하지 않으면 제외

                # (참고: 만약 v2.8의 N:M 리스트 방식('이캠,문캠')을 쓰려면 아래 코드로 대체)
                # has_matching_campus = df['캠퍼스'].apply(
                #    lambda menu_campuses: any(target in menu_campuses for target in target_campuses)
                # )
                # scores[~has_matching_campus] = 0.0

        # (C) Operating Time 필터 (DB '오픈/마감시간' 검사)
        meal_time_map = {
            '아침': time(9, 0),  # 9:00 AM
            '점심': time(13, 0),  # 1:00 PM
            '저녁': time(19, 0)  # 7:00 PM
        }
        target_time = meal_time_map.get(user_preferences.meal_type)

        if target_time and '오픈시간' in df.columns and '마감시간' in df.columns:

            def is_open(row):
                open_t = row['오픈시간']
                close_t = row['마감시간']
                if pd.isna(open_t) or pd.isna(close_t):
                    return True
                if open_t <= close_t:
                    return open_t <= target_time <= close_t
                else:
                    return target_time >= open_t or target_time <= close_t

            is_not_open = ~df.apply(is_open, axis=1)
            scores[is_not_open] = 0.0

        # --- [2. 가산점 로직 (Soft Filters)] ---

        # (A) 예산 적합도 (하드 필터 + 가산점)
        if (user_preferences.budget is not None) and (user_preferences.budget > 0) and ('가격' in df.columns):
            within = df['가격'] <= user_preferences.budget
            scores[~within.values] = 0.0

            if user_preferences.min_budget:
                within_min = df['가격'] >= user_preferences.min_budget
                scores[~within_min.values] = 0.0

            if scores.sum() > 0:
                price_diff = (df['가격'] - user_preferences.budget).abs()
                price_score = 1 - (price_diff / max(user_preferences.budget, 1)).clip(0, 1)
                scores[scores > 0] += price_score[scores > 0] * 3.0

        # (B) Rating(평점) 가산점
        if '평점' in df.columns:
            rating_score = (df['평점'] / 5.0)
            scores[scores > 0] += rating_score[scores > 0] * 1.5

            # (C) 사용자 프롬프트 (태그)
        if user_preferences.user_prompt and '태그' in df.columns:
            prompt_lower = user_preferences.user_prompt.lower()
            negative_keywords = []
            positive_keywords = []

            if '말고' in prompt_lower or '빼고' in prompt_lower or '제외' in prompt_lower:
                words = prompt_lower.replace(',', ' ').split()
                for i, word in enumerate(words):
                    if word in ['말고', '빼고', '제외']:
                        if i > 0:
                            negative_keywords.append(words[i - 1])

            if '밥' in prompt_lower: positive_keywords.append('밥')
            if '고기' in prompt_lower: positive_keywords.append('고기')
            if '가벼운' in prompt_lower: positive_keywords.append('가벼운')
            if '든든한' in prompt_lower: positive_keywords.append('든든한')

            for neg_keyword in negative_keywords:
                has_negative = df['태그'].apply(
                    lambda tags: any(neg_keyword in tag for tag in tags) if isinstance(tags, list) else False
                )
                scores[has_negative] = 0.0

            for pos_keyword in positive_keywords:
                has_positive = df['태그'].apply(
                    lambda tags: any(pos_keyword in tag for tag in tags) if isinstance(tags, list) else False
                )
                scores[scores > 0] += has_positive.astype(float)[scores > 0] * 2.0

        # (D) 카테고리 매칭 (예: '한식', '일식') - UserInput에 이 필드가 없으므로 현재 비활성
        if user_preferences.preferred_categories and '카테고리' in df.columns:
            for category in user_preferences.preferred_categories:
                has_category = df['카테고리'].apply(
                    lambda cats: category in cats if isinstance(cats, list) else False
                )
                scores[scores > 0] += has_category.astype(float)[scores > 0] * 2.0

        # (E) 영양 선호도 (DIET)
        if user_preferences.prefer_low_calorie and '열량' in df.columns:
            if df['열량'].max() > df['열량'].min():
                calorie_score = 1 - ((df['열량'] - df['열량'].min()) / (df['열량'].max() - df['열량'].min()))
                scores[scores > 0] += calorie_score[scores > 0] * 1.5

        # (F) 식사 시간대 (태그)
        if '태그' in df.columns:
            meal_tags = {
                '아침': ['가벼운', '샌드위치', '샐러드'],
                '점심': ['든든한', '고단백', '영양균형', '밥', '고기'],
                '저녁': ['든든한', '국물', '따뜻한', '밥']
            }
            if user_preferences.meal_type in meal_tags:
                for tag in meal_tags[user_preferences.meal_type]:
                    tag_match = df['태그'].apply(lambda tags: tag in tags if isinstance(tags, list) else False)
                    scores[scores > 0] += tag_match.astype(float)[scores > 0] * 1.0

        # 정규화 (유지)
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores


# --- 4. LangChain 및 LLM 설정 (기존과 동일) ---

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
parser = PydanticOutputParser(pydantic_object=FinalRecommendation)
if not OPENAI_API_KEY:
    print("경고: OPENAI_API_KEY 환경 변수가 .env 파일에 설정되지 않았습니다. LLM 기능이 비활성화됩니다.")
    llm = None
else:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=OPENAI_API_KEY)
    except ImportError:
        print("langchain_openai가 설치되지 않았습니다. LLM 기능을 사용하려면 설치해주세요.")
        llm = None
    except Exception as e:
        print(f"OpenAI LLM 초기화 실패: {e}. (API 키를 확인하세요)")
        llm = None
prompt_template = """
당신은 고려대학교 근처 맛집 메뉴 추천 AI 'MenuMate'입니다.
사용자의 세부 요청과 AI A가 1차 필터링한 메뉴 후보 리스트를 받았습니다.

[후보 리스트] (이름, 가격, AI A 점수, 태그 순)
{candidates_str}

[사용자 세부 요청]
"{user_prompt}"

[사용자 가격 제한]
{price_str}

[사용자가 추천받길 원하는 끼니]
{target_meals_str}

[지시]
1. [후보 리스트] 중에서 [사용자 세부 요청]과 [사용자 가격 제한]을 모두 만족하는 메뉴를 고르세요.
2. [사용자가 추천받길 원하는 끼니] 목록({target_meals_str})에 있는 슬롯에만 추천 메뉴를 채워주세요.
3. 각 후보의 'price' 정보를 [출력 JSON]의 'price' 필드에 정확히 기입해주세요.
4. 사용자 요청 "{user_prompt}"을 반드시 고려하세요. (예: "국수 말고 밥" → 면 종류 메뉴는 제외)
5. 추천 메뉴를 선정한 후, [사용자 세부 요청]을 바탕으로 그 이유를 요약하는 1~3개의 'reason_hashtags'를 반드시 생성해주세요. (예: '#속편한', '#든든한', '#가성비')
6. 목록에 없는 다른 끼니 슬롯은 반드시 null로 설정해야 합니다.
7. 만약 사용자가 '점심'만 원했다면 'morning'과 'dinner'는 반드시 null이어야 합니다.

반드시 다음 JSON 형식으로만 응답해야 합니다 (다른 말은 절대 하지 마세요):
{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(
    template=prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
if llm:
    chain = prompt | llm | parser
else:
    chain = None
    print("LLM 체인 구성 실패. /recommend 엔드포인트가 작동하지 않을 수 있습니다.")

# --- 5. FastAPI 앱 및 캐싱 설정 (기존과 동일) ---

app = FastAPI(
    title="MenuMate AI - 통합 추천 모듈",
    description="영양성분 기반 후보 생성(A) 및 LLM 최종 추천(B) 통합 API"
)
preprocessor: Optional[MenuDataPreprocessor] = None
cache: Optional[redis.Redis] = None


@app.on_event("startup")
async def startup():
    """서버 시작 시 데이터 로드, 전처리, 캐시 연결"""
    global preprocessor, cache
    preprocessor = MenuDataPreprocessor()
    preprocessor.load_nutrition_data()
    preprocessor.extract_features()
    print("✅ (AI A) 데이터 전처리 모듈 준비 완료!")
    try:
        cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        cache.ping()
        print("✅ (AI B) Redis 캐시 서버에 연결되었습니다.")
    except redis.ConnectionError as e:
        print(f"경고: Redis 서버에 연결할 수 없습니다. 캐싱이 비활성화됩니다. (오류: {e})")
        cache = None
    print("✅ 통합 AI 모듈 준비 완료!")


@app.get("/")
async def root():
    return {
        "service": "MenuMate AI (Combined A+B)",
        "status": "running",
        "version": "2.6.1 (No LowSugar, Rating, Time, Campus)",
        "llm_ready": (chain is not None),
        "preprocessor_ready": (preprocessor is not None and preprocessor.menu_df is not None),
        "db_connected": (preprocessor is not None and preprocessor.db_engine is not None)
    }


# --- 6. 내부 로직 함수 (기존과 동일) ---

async def _generate_candidates(preferences: UserPreferences) -> RecommendationRequest:
    if preprocessor is None or preprocessor.menu_df is None:
        raise HTTPException(status_code=503, detail="데이터 전처리기가 준비되지 않았습니다.")
    try:
        scores = preprocessor.calculate_scores(preferences)
        df = preprocessor.menu_df.copy()
        df['base_score'] = scores
        df_filtered = df[df['base_score'] > 0]
        df_sorted = df_filtered.sort_values('base_score', ascending=False).head(20)
        candidates = []
        for _, row in df_sorted.iterrows():  # df -> df_sorted
            candidates.append(MenuCandidate(
                restaurant_name=row['식당명'],
                menu_name=row['제품명'],
                price=int(row['가격']),
                base_score=float(row['base_score']),
                tags=row['태그'] if isinstance(row['태그'], list) else []
            ))
        ai_b_request = RecommendationRequest(
            candidates=candidates,
            user_prompt=preferences.user_prompt or "맛있고 건강한 메뉴",
            price=PriceRange(
                min=preferences.min_budget,
                max=preferences.budget
            ) if (preferences.budget or preferences.min_budget) else None,
            target_meals=preferences.target_meals,
            conversation_history=[]
        )
        return ai_b_request
    except Exception as e:
        print(f"후보군 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"후보군 생성(AI A) 실패: {str(e)}")


async def _refine_recommendations(request: RecommendationRequest) -> FinalRecommendation:
    if chain is None:
        raise HTTPException(status_code=503, detail="LLM 체인이 준비되지 않았습니다. (.env 파일 또는 API 키 확인 필요)")
    cache_key = None
    if cache:
        try:
            cache_key = f"recommend:{hash(request.model_dump_json())}"
            cached_result = cache.get(cache_key)
            if cached_result:
                print("캐시된 결과를 반환합니다.")
                return FinalRecommendation.model_validate_json(cached_result)
        except Exception as e:
            print(f"캐시 조회 중 오류 발생: {e}")
    candidates_str = "\n".join(
        [f"- {c.restaurant_name} '{c.menu_name}' (가격: {c.price}원, 점수: {c.base_score:.2f}, 태그: {c.tags})"
         for c in request.candidates]
    )
    price_str = "제한 없음"
    if request.price:
        parts = []
        if request.price.min is not None:
            parts.append(f"{request.price.min}원 이상")
        if request.price.max is not None:
            parts.append(f"{request.price.max}원 이하")
        if parts:
            price_str = " ".join(parts)
    target_meals_str = ", ".join(request.target_meals)
    try:
        print(f"LLM 호출 시작: (User: {request.user_prompt}, Price: {price_str}, Meals: {target_meals_str})")
        result = await chain.ainvoke({
            "candidates_str": candidates_str,
            "user_prompt": request.user_prompt,
            "price_str": price_str,
            "target_meals_str": target_meals_str,
            "history": "\n".join(request.conversation_history or [])
        })
        if cache and cache_key:
            try:
                cache.set(cache_key, result.model_dump_json(), ex=3600)
                print("새 결과를 생성하고 캐시에 저장했습니다.")
            except Exception as e:
                print(f"캐시 저장 중 오류 발생: {e}")
        return result
    except Exception as e:
        print(f"LLM 파싱 또는 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"메뉴 추천(AI B) 실패: {str(e)}")


# --- 7. 통합 엔드포인트 (새 JSON 형식 사용) ---

@app.post("/recommend", response_model=FinalRecommendation)
async def get_recommendation(user_input: UserInput):
    try:
        print(
            f"입력 받음: category={user_input.category}, meals={user_input.meals}, prompt='{user_input.prompt}', location={user_input.location}")
        preferences = convert_user_input(user_input)
        print(
            f"변환 완료: meal_type={preferences.meal_type}, category={preferences.category}, location={preferences.location}")
        print(f"1단계: 후보군 생성 시작")
        candidate_request = await _generate_candidates(preferences)
        if not candidate_request.candidates:
            print("1단계 결과: 추천할 후보군이 없습니다. (필터 조건이 너무 많을 수 있습니다)")
            return FinalRecommendation(morning=None, lunch=None, dinner=None)
        print(f"1단계 완료: {len(candidate_request.candidates)}개 후보 생성")
        print("2단계: LLM 정제 시작")
        final_recommendation = await _refine_recommendations(candidate_request)
        print("2단계 완료: 최종 추천 생성")
        return final_recommendation
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"전체 추천 파이프라인 오류: {e}")
        raise HTTPException(status_code=500, detail=f"전체 추천 프로세스 실패: {str(e)}")


# --- 8. 기존 호환성 엔드포인트 (선택적) ---

@app.post("/recommend/full", response_model=FinalRecommendation)
async def get_full_recommendation(preferences: UserPreferences):
    try:
        print(
            f"1단계: 후보군 생성 시작 (User: {preferences.user_prompt}, Category: {preferences.category}, Location: {preferences.location})")
        candidate_request = await _generate_candidates(preferences)
        if not candidate_request.candidates:
            print("1단계 결과: 추천할 후보군이 없습니다.")
            return FinalRecommendation(morning=None, lunch=None, dinner=None)
        print(f"1단계 완료: {len(candidate_request.candidates)}개 후보 생성")
        print("2단계: LLM 정제 시작")
        final_recommendation = await _refine_recommendations(candidate_request)
        print("2단계 완료: 최종 추천 생성")
        return final_recommendation
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"전체 추천 파이프라인 오류: {e}")
        raise HTTPException(status_code=500, detail=f"전체 추천 프로세스 실패: {str(e)}")


# --- 9. 헬퍼 엔드포인트 ---

@app.get("/menu/all")
async def get_all_menus():
    """(수정됨) 전처리된 메뉴 리스트 조회 (디버깅용)"""
    if preprocessor is None or preprocessor.menu_df is None:
        raise HTTPException(status_code=503, detail="데이터가 로드되지 않음")

    df = preprocessor.menu_df.copy()
    df['태그'] = df['태그'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    df['카테고리'] = df['카테고리'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    df['오픈시간'] = df['오픈시간'].apply(lambda x: x.strftime('%H:%M:%S') if isinstance(x, time) else None)
    df['마감시간'] = df['마감시간'].apply(lambda x: x.strftime('%H:%M:%S') if isinstance(x, time) else None)
    return df.to_dict(orient='records')


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "preprocessor": preprocessor is not None,
        "llm": chain is not None,
        "cache": cache is not None,
        "db_connected": (preprocessor is not None and preprocessor.db_engine is not None)
    }


# --- 10. 서버 실행 ---

if __name__ == "__main__":
    print('서버실행')
    uvicorn.run(app, host="0.0.0.0", port=8000)