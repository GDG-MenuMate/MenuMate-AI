import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import redis
import uvicorn
from dotenv import load_dotenv  # <-- .env 로드 라이브러리
from sqlalchemy import create_engine, text  # <-- SQLAlchemy 임포트

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
    preferred_categories: List[str] = Field(default=[], description="선호 카테고리")
    meal_type: str = Field(default="점심", description="식사 시간대: 아침/점심/저녁")
    target_meals: List[str] = Field(default=["lunch"], description="['morning', 'lunch', 'dinner']")
    user_prompt: str = Field(default="", description="사용자 세부 요청")

    # [수정됨] 영양 관련 선호도 축소 (DB에 정보 없음)
    prefer_high_protein: bool = Field(default=False, description="고단백 선호 (비활성화)")
    prefer_low_calorie: bool = Field(default=False, description="저칼로리 선호 (유일하게 사용)")
    prefer_low_sodium: bool = Field(default=False, description="저나트륨 선호 (비활성화)")

    # [중요] 사용자의 '목표' 카테고리 (예: 'vegan', 'muslim')
    category: Optional[str] = Field(default=None, description="카테고리")
    height: Optional[int] = Field(default=None, description="키")
    weight: Optional[int] = Field(default=None, description="몸무게")
    location: List[str] = Field(default=[], description="위치 정보")


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
    - 'DIET' 카테고리는 'prefer_low_calorie' 플래그로 변환됩니다.
    - 'VEGAN', 'MUSLIM'은 'category' 필드에 그대로 전달됩니다.
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

    # [수정됨] 카테고리에 따른 영양 선호도 설정 (저칼로리만 남김)
    prefer_low_calorie = False

    if user_input.category.upper() == "DIET":
        prefer_low_calorie = True

    # [수정됨] 고단백, 저나트륨 로직은 DB에 정보가 없으므로 항상 False
    prefer_high_protein = False
    prefer_low_sodium = False

    # [선택적] 위치 정보를 선호 카테고리로 매핑 (예시)
    preferred_categories = []
    if "science_campus" in user_input.location:
        preferred_categories.append("자연계캠퍼스")
    if "humanities_campus" in user_input.location:
        preferred_categories.append("인문계캠퍼스")

    return UserPreferences(
        budget=user_input.priceRange.maxPrice,
        min_budget=user_input.priceRange.minPrice,
        preferred_categories=preferred_categories,  # <- DB 카테고리 필터링에 사용
        meal_type=meal_type,
        target_meals=target_meals,
        user_prompt=user_input.prompt,
        prefer_high_protein=prefer_high_protein,  # 항상 False
        prefer_low_calorie=prefer_low_calorie,  # 'DIET'일 때만 True
        prefer_low_sodium=prefer_low_sodium,  # 항상 False
        category=user_input.category,  # <-- 'VEGAN', 'MUSLIM' 등 필터링을 위해 전달
        height=user_input.diet.height if user_input.diet else None,
        weight=user_input.diet.weight if user_input.diet else None,
        location=user_input.location
    )


# --- 3. 데이터 전처리 클래스 (수정됨) ---

class MenuDataPreprocessor:
    """
    (수정됨) '열량', '가격', '태그', '카테고리' 기반 메뉴 데이터 전처리
    - 'tags', 'category'를 DB에서 직접 불러온다고 가정합니다.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.menu_features = None
        self.menu_df = None
        self.db_engine = self._create_db_engine()  # <-- DB 엔진 초기화

    def _create_db_engine(self):
        """
        .env 파일의 정보로 SQLAlchemy DB 엔진 생성
        """
        load_dotenv()  # <-- .env 파일 로드
        db_user = os.environ.get("DB_USER")
        db_pass = os.environ.get("DB_PASSWORD")
        db_host = os.environ.get("DB_HOST")
        db_port = os.environ.get("DB_PORT")
        db_name = os.environ.get("DB_NAME")

        if not all([db_user, db_pass, db_host, db_port, db_name]):
            print("경고: DB 접속 정보(.env)가 불완전합니다. DB 로드에 실패할 수 있습니다.")
            return None

        # [수정] 사용하는 DB에 맞게 연결 문자열(DATABASE_URL) 변경
        # (예: PostgreSQL)
        DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

        # (예: MySQL)
        # DATABASE_URL = f"mysql+mysqlclient://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

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
        (수정됨) SQLAlchemy와 pd.read_sql을 사용해 DB에서 직접 데이터를 로드합니다.
        """

        if self.db_engine is None:
            print("오류: DB 엔진이 초기화되지 않았습니다. 샘플 데이터로 대체합니다.")
            df = self._get_sample_data()  # <-- 실패 시 샘플 데이터 사용
        else:
            sql_query = """
            SELECT
                m.name AS "제품명",
                r.name AS "식당명",
                m.price AS "가격",
                m.calories AS "열량",
                m.tags AS "태그",         
                m.category AS "카테고리"  
            FROM menus m
            JOIN restaurants r ON m.restaurants_id = r.restaurants_id
            """

            try:
                print("DB에서 데이터 로드를 시도합니다...")
                df = pd.read_sql(text(sql_query), self.db_engine)

            except Exception as e:
                print(f"❌ DB 데이터 로드 실패: {e}. 샘플 데이터로 대체합니다.")
                df = self._get_sample_data()

        # --- (이하 로직은 기존과 동일) ---

        # '열량'이 없는 경우 (예: 0 또는 None) 중앙값으로 대체
        df['열량'] = pd.to_numeric(df['열량'], errors='coerce')
        df['열량'] = df['열량'].fillna(df['열량'].median())

        # DB에서 '태그'를 문자열(예: "밥,국물")로 가져왔다고 가정하고 리스트로 변환
        if '태그' in df.columns:
            df['태그'] = df['태그'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) and x else [])
        else:
            df['태그'] = [[] for _ in range(len(df))]
            print("경고: '태그' 컬럼이 DB에 없습니다.")

        if '카테고리' not in df.columns:
            df['카테고리'] = '기타'
            print("경고: '카테고리' 컬럼이 DB에 없습니다.")

        self.menu_df = df
        print(f"✅ 메뉴 데이터 처리 완료: {len(df)}개 (소스: {'DB' if self.db_engine else '샘플'})")
        return df

    def _get_sample_data(self):
        """
        DB 연결 실패 시 사용하는 하드코딩된 샘플 데이터
        (VEGAN, MUSLIM 테스트를 위해 샘플 데이터 수정)
        """
        print("샘플 데이터를 사용합니다...")
        return pd.DataFrame([
            {'제품명': '김치찌개', '식당명': '한식당 A', '가격': 8000, '열량': 480, '카테고리': '한식', '태그': '밥,국물,든든한,돼지고기'},  # '돼지고기' 태그
            {'제품명': '비건 비빔밥', '식당명': '한식당 A', '가격': 8000, '열량': 550, '카테고리': '한식', '태그': '밥,야채,건강한,비건'},  # '비건' 태그
            {'제품명': '제육볶음', '식당명': '한식당 B', '가격': 8500, '열량': 620, '카테고리': '한식', '태그': '밥,고기,든든한,매콤한,돼지고기'},
            # '돼지고기' 태그
            {'제품명': '냉면', '식당명': '한식당 C', '가격': 9000, '열량': 420, '카테고리': '한식', '태그': '면,시원한,여름'},
            {'제품명': '돈까스', '식당명': '일식당 A', '가격': 9000, '열량': 750, '카테고리': '일식', '태그': '밥,튀김,든든한,돼지고기'},  # '돼지고기' 태그
            {'제품명': '치킨 샐러드', '식당명': '샐러드 전문점', '가격': 7000, '열량': 310, '카테고리': '양식', '태그': '샐러드,가벼운,다이어트,고단백,할랄'},
            # '할랄' 태그
            {'제품명': '두부 샌드위치', '식당명': '샌드위치 전문점', '가격': 7500, '열량': 372, '카테고리': '양식', '태그': '샌드위치,가벼운,비건'},  # '비건' 태그
        ])

    def extract_features(self):
        """
        (수정됨) '열량', '가격', '태그', '카테고리' 기반 피처 추출
        """
        if self.menu_df is None:
            print("오류: 메뉴 데이터가 로드되지 않았습니다.")
            return None

        df = self.menu_df.copy()

        numeric_features = ['열량', '가격']

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

        if '카테고리' in df.columns:
            category_dummies = pd.get_dummies(df['카테고리'], prefix='카테고리')
            df = pd.concat([df, category_dummies], axis=1)

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
        category_cols = [col for col in df.columns if col.startswith('카테고리_')]
        level_cols = ['칼로리_레벨', '가격_레벨']
        level_cols = [col for col in level_cols if col in df.columns]

        all_feature_cols = feature_cols + tag_cols + category_cols + level_cols
        all_feature_cols = [col for col in all_feature_cols if col in df.columns]

        self.menu_features = df[all_feature_cols].values
        print(f"✅ 피처 추출 완료: {self.menu_features.shape[1] if isinstance(self.menu_features, np.ndarray) else 0}개 피처")
        return df

    def calculate_scores(self, user_preferences: UserPreferences):
        """
        (수정됨) VEGAN, MUSLIM 강력한 필터 조건 추가
        """
        if self.menu_df is None:
            raise ValueError("메뉴 데이터가 로드되지 않았습니다.")

        df = self.menu_df.copy()
        # [수정] 점수 계산을 1점으로 시작
        scores = np.ones(len(df))

        # --- [새로 추가된 강력한 필터 로직] ---
        # 이 필터들은 '가산점'이 아닌 '제외' 필터입니다.
        if user_preferences.category and '태그' in df.columns:
            category_upper = user_preferences.category.upper()

            if category_upper == "VEGAN":
                # '태그'에 '비건' 또는 'vegan'이 없는 메뉴를 찾습니다.
                is_not_vegan = df['태그'].apply(
                    lambda tags: '비건' not in tags and 'vegan' not in tags
                )
                scores[is_not_vegan] = 0.0  # 해당 메뉴 제외

            if category_upper == "MUSLIM":
                # '태그'에 '할랄'이 없거나, '돼지고기' 태그가 있는 메뉴를 찾습니다.
                is_not_muslim = df['태그'].apply(
                    lambda tags: ('할랄' not in tags) or ('돼지고기' in tags)
                )
                scores[is_not_muslim] = 0.0  # 해당 메뉴 제외
        # --- [여기까지] ---

        # 1. 예산 적합도 (하드 필터)
        if (user_preferences.budget is not None) and (user_preferences.budget > 0) and ('가격' in df.columns):
            within = df['가격'] <= user_preferences.budget
            scores[~within.values] = 0.0  # 예산 초과 메뉴 제외

            if user_preferences.min_budget:
                within_min = df['가격'] >= user_preferences.min_budget
                scores[~within_min.values] = 0.0  # 최소 예산 미만 메뉴 제외

            # [수정] 살아남은 메뉴(0점 아님)에만 예산 근접 가산점 부여
            if scores.sum() > 0:  # 살아남은 메뉴가 있을 때만
                price_diff = (df['가격'] - user_preferences.budget).abs()
                price_score = 1 - (price_diff / max(user_preferences.budget, 1)).clip(0, 1)
                scores[scores > 0] += price_score[scores > 0] * 3.0

        # 2. 사용자 프롬프트 키워드 매칭 (하드 필터 + 가산점)
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

            # 부정 키워드 (하드 필터)
            for neg_keyword in negative_keywords:
                has_negative = df['태그'].apply(
                    lambda tags: any(neg_keyword in tag for tag in tags) if isinstance(tags, list) else False
                )
                scores[has_negative] = 0.0

            # [수정] 긍정 키워드 (가산점) - 살아남은 메뉴에만
            for pos_keyword in positive_keywords:
                has_positive = df['태그'].apply(
                    lambda tags: any(pos_keyword in tag for tag in tags) if isinstance(tags, list) else False
                )
                scores[scores > 0] += has_positive.astype(float)[scores > 0] * 2.0

        # 3. 카테고리 매칭 (가산점)
        if user_preferences.preferred_categories and '카테고리' in df.columns:
            for category in user_preferences.preferred_categories:
                category_match = df['카테고리'].str.contains(category, case=False, na=False)
                # [수정] 살아남은 메뉴에만 가산점
                scores[scores > 0] += category_match.astype(float)[scores > 0] * 2.0

        # 4. 영양 선호도 (가산점 - DIET)

        # --- [제거됨] (단백질 점수 로직) ---

        # --- [유지] (저칼로리)
        if user_preferences.prefer_low_calorie and '열량' in df.columns:
            if df['열량'].max() > df['열량'].min():
                calorie_score = 1 - ((df['열량'] - df['열량'].min()) / (df['열량'].max() - df['열량'].min()))
                # [수정] 살아남은 메뉴에만 가산점
                scores[scores > 0] += calorie_score[scores > 0] * 1.5

        # --- [제거됨] (나트륨 점수 로직) ---

        # 5. 식사 시간대 적합도 (가산점)
        if '태그' in df.columns:
            meal_tags = {
                '아침': ['가벼운', '샌드위치', '샐러드'],
                '점심': ['든든한', '고단백', '영양균형', '밥', '고기'],
                '저녁': ['든든한', '국물', '따뜻한', '밥']
            }
            if user_preferences.meal_type in meal_tags:
                for tag in meal_tags[user_preferences.meal_type]:
                    tag_match = df['태그'].apply(lambda tags: tag in tags if isinstance(tags, list) else False)
                    # [수정] 살아남은 메뉴에만 가산점
                    scores[scores > 0] += tag_match.astype(float)[scores > 0] * 1.0

        # 정규화 (유지)
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores


# --- 4. LangChain 및 LLM 설정 (기존과 동일) ---

# .env 파일에서 환경 변수를 로드
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 1. Output Parser
parser = PydanticOutputParser(pydantic_object=FinalRecommendation)

# 2. LLM 모델: GPT-4o-mini 사용, temperature=0.6
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

# 3. 프롬프트 템플릿 (기존과 동일)
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

# 4. 프롬프트 완성
prompt = ChatPromptTemplate.from_template(
    template=prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 5. LangChain 체인 구성
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

# 전역 변수
preprocessor: Optional[MenuDataPreprocessor] = None
cache: Optional[redis.Redis] = None


@app.on_event("startup")
async def startup():
    """서버 시작 시 데이터 로드, 전처리, 캐시 연결"""
    global preprocessor, cache

    # 1. AI A 시작 로직
    preprocessor = MenuDataPreprocessor()
    preprocessor.load_nutrition_data()  # (수정됨) DB에서 로드 시도
    preprocessor.extract_features()
    print("✅ (AI A) 데이터 전처리 모듈 준비 완료!")

    # 2. AI B 시작 로직 (Redis 캐시)
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
        "version": "2.3.0 (Strong Filters, DB Load, With Tags)",
        "llm_ready": (chain is not None),
        "preprocessor_ready": (preprocessor is not None and preprocessor.menu_df is not None),
        "db_connected": (preprocessor is not None and preprocessor.db_engine is not None)
    }


# --- 6. 내부 로직 함수 (기존과 동일) ---

async def _generate_candidates(preferences: UserPreferences) -> RecommendationRequest:
    """
    (구 AI A 로직) 사용자 선호도에 맞는 후보군 리스트 생성
    (강력한 필터 로직이 포함된 calculate_scores 사용)
    """
    if preprocessor is None or preprocessor.menu_df is None:
        raise HTTPException(status_code=503, detail="데이터 전처리기가 준비되지 않았습니다.")

    try:
        # 1. 점수 계산 (수정된 calculate_scores 호출)
        scores = preprocessor.calculate_scores(preferences)

        # 2. 상위 후보 선택
        df = preprocessor.menu_df.copy()
        df['base_score'] = scores

        # [수정] 점수가 0보다 큰 (살아남은) 메뉴들 중에서 정렬
        df_filtered = df[df['base_score'] > 0]
        df_sorted = df_filtered.sort_values('base_score', ascending=False).head(20)

        # 3. AI B 형식으로 변환
        candidates = []
        for _, row in df_sorted.iterrows():  # df -> df_sorted
            candidates.append(MenuCandidate(
                restaurant_name=row['식당명'],
                menu_name=row['제품명'],
                price=int(row['가격']),
                base_score=float(row['base_score']),
                tags=row['태그'] if isinstance(row['태그'], list) else []
            ))

        # 4. AI B로 전달할 요청 객체 생성
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
    """
    (구 AI B 로직) LLM을 사용하여 후보군을 최종 추천으로 정제
    (기존 로직과 동일)
    """
    if chain is None:
        raise HTTPException(status_code=503, detail="LLM 체인이 준비되지 않았습니다. (.env 파일 또는 API 키 확인 필요)")

    # 1. 캐싱 (기존과 동일)
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

    # 2. LLM 입력값 가공 (기존과 동일)
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

    # 3. LLM 체인 호출 (기존과 동일)
    try:
        print(f"LLM 호출 시작: (User: {request.user_prompt}, Price: {price_str}, Meals: {target_meals_str})")
        result = await chain.ainvoke({
            "candidates_str": candidates_str,
            "user_prompt": request.user_prompt,
            "price_str": price_str,
            "target_meals_str": target_meals_str,
            "history": "\n".join(request.conversation_history or [])
        })

        # 4. 캐시에 결과 저장 (기존과 동일)
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
    """
    새로운 JSON 형식으로 사용자 입력을 받아 추천 결과 반환
    (DB에 'tags'가 있다고 가정한 로직으로 작동)
    """
    try:
        # 1. 입력 형식 변환
        print(f"입력 받음: category={user_input.category}, meals={user_input.meals}, prompt='{user_input.prompt}'")
        preferences = convert_user_input(user_input)
        print(
            f"변환 완료: target_meals={preferences.target_meals}, category={preferences.category}, low_calorie={preferences.prefer_low_calorie}")

        # 2. AI A 로직 호출 (강력 필터 적용)
        print(f"1단계: 후보군 생성 시작")
        candidate_request = await _generate_candidates(preferences)

        if not candidate_request.candidates:
            print("1단계 결과: 추천할 후보군이 없습니다. (필터 조건이 너무 많을 수 있습니다)")
            return FinalRecommendation(morning=None, lunch=None, dinner=None)

        print(f"1단계 완료: {len(candidate_request.candidates)}개 후보 생성")

        # 3. AI B 로직 호출 (LLM 정제)
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
    """
    기존 UserPreferences 형식을 사용하는 엔드포인트 (하위 호환성)
    """
    try:
        print(f"1단계: 후보군 생성 시작 (User: {preferences.user_prompt}, Category: {preferences.category})")
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
    # '태그'가 리스트이므로 문자열로 변환
    df['태그'] = df['태그'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')

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
    # 이 파일을 'main.py'로 저장했다면, 터미널에서 아래와 같이 실행합니다.
    # (실행 전 .env 파일에 OPENAI_API_KEY와 DB 접속 정보를 저장해야 합니다.)
    # uvicorn main:app --host "127.0.0.1" --port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)