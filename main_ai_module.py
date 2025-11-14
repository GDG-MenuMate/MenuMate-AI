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

# LangChain 및 OpenAI 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# --- 1. Pydantic 모델 정의 (AI A와 B의 모델 통합) ---

# AI A가 사용하던 모델 (사용자 입력)
class UserPreferences(BaseModel):
    """사용자 입력 (버튼/챗봇)"""
    user_id: Optional[int] = None
    budget: Optional[int] = Field(default=10000, description="예산 (원)")
    preferred_categories: List[str] = Field(default=[], description="선호 카테고리")
    allergens: List[str] = Field(default=[], description="알레르기 ['땅콩', '대두', ...]")
    meal_type: str = Field(default="점심", description="식사 시간대: 아침/점심/저녁")
    target_meals: List[str] = Field(default=["lunch"], description="['morning', 'lunch', 'dinner']")
    user_prompt: str = Field(default="", description="사용자 세부 요청 (예: '속편한 음식')")

    # 영양 관련 선호도
    prefer_high_protein: bool = Field(default=False, description="고단백 선호")
    prefer_low_calorie: bool = Field(default=False, description="저칼로리 선호")
    prefer_low_sodium: bool = Field(default=False, description="저나트륨 선호")


# AI A와 B가 공통으로 사용하던 모델 (중복 제거)
class MenuCandidate(BaseModel):
    """AI A -> AI B로 전달할 메뉴 후보"""
    restaurant_name: str
    menu_name: str
    price: int
    base_score: float
    tags: List[str]


class PriceRange(BaseModel):
    """가격 범위"""
    min: Optional[int] = None
    max: Optional[int] = None


# AI B가 AI A로부터 받던 요청 모델
class RecommendationRequest(BaseModel):
    """AI B 모듈(LLM)이 받을 요청 Body 전체"""
    candidates: List[MenuCandidate]
    user_prompt: str
    price: Optional[PriceRange] = Field(
        default=None, description="사용자 가격 제한 (예: {'max': 10000})"
    )
    target_meals: List[str] = Field(
        description="추천받고자 하는 끼니 목록 (예: ['lunch', 'dinner'])"
    )
    conversation_history: Optional[List[str]] = []  # (확장용) 대화 이력


# AI B가 FE로 응답하던 최종 모델
class RecommendedMenu(BaseModel):
    """FE의 Result 화면에 바인딩될 최종 메뉴 1개 정보"""
    restaurant_name: str = Field(description="식당 이름")
    menu_name: str = Field(description="메뉴 이름")
    price: int = Field(description="메뉴의 실제 가격")
    justification: str = Field(description="이 메뉴를 추천하는 이유 (LLM이 작성)")
    new_score: float = Field(description="LLM이 재평가한 최종 점수 (0.0 ~ 1.0)")
    reason_hashtags: List[str] = Field(
        description="추천 이유를 요약하는 3-5개의 해시태그 (예: ['#속편한', '#든든한', '#가성비'])"
    )


class FinalRecommendation(BaseModel):
    """최종 API 응답 형식"""
    morning: Optional[RecommendedMenu] = Field(
        default=None, description="추천 아침 메뉴 (해당 없으면 null)"
    )
    lunch: Optional[RecommendedMenu] = Field(
        default=None, description="추천 점심 메뉴 (해당 없으면 null)"
    )
    dinner: Optional[RecommendedMenu] = Field(
        default=None, description="추천 저녁 메뉴 (해당 없으면 null)"
    )


# --- 2. 데이터 전처리 클래스 (From AI A) ---

class MenuDataPreprocessor:
    """
    영양성분표 기반 메뉴 데이터 전처리
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.menu_features = None
        self.menu_df = None

    def load_nutrition_data(self, csv_path: str = None):
        """
        영양성분표 CSV 로드 또는 샘플 데이터 생성
        """
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            # 샘플 데이터 (예시)
            df = pd.DataFrame([
                # 영양성분표 1 (한식)
                {'제품명': '치킨 텐더바이트', '식당명': '브랜드 A', '중량': 16, '열량': 222, '단백질': 35, '포화지방': 2, '당류': 4, '나트륨': 222,
                 '가격': 6500, '카테고리': '양식', '태그': '고단백,가벼운'},
                {'제품명': '치킨 마리네이드 샐러드', '식당명': '브랜드 A', '중량': 74, '열량': 492, '단백질': 44, '포화지방': 6, '당류': 1, '나트륨': 492,
                 '가격': 8500, '카테고리': '양식', '태그': '고단백,샐러드'},
                {'제품명': '더블 치킨 타르타르', '식당명': '브랜드 A', '중량': 71, '열량': 615, '단백질': 71, '포화지방': 5, '당류': 10, '나트륨': 615,
                 '가격': 9000, '카테고리': '양식', '태그': '든든한,고단백'},
                {'제품명': '치킨 칠리 샐러드', '식당명': '브랜드 A', '중량': 88, '열량': 591, '단백질': 44, '포화지방': 7, '당류': 10, '나트륨': 591,
                 '가격': 8500, '카테고리': '양식', '태그': '샐러드,고단백'},
                {'제품명': '비프 칠리 샐러드', '식당명': '브랜드 A', '중량': 89, '열량': 731, '단백질': 51, '포화지방': 19, '당류': 9, '나트륨': 731,
                 '가격': 9500, '카테고리': '양식', '태그': '든든한,고단백'},

                # 영양성분표 2 (서브웨이 스타일)
                {'제품명': '스파이시 바비큐', '식당명': '샌드위치 전문점', '중량': 256, '열량': 374, '단백질': 25.2, '포화지방': 7.4, '당류': 15.0,
                 '나트륨': 903, '가격': 6500, '카테고리': '양식', '태그': '샌드위치,간편'},
                {'제품명': '스파이시 쉬림프', '식당명': '샌드위치 전문점', '중량': 213, '열량': 245, '단백질': 16.5, '포화지방': 0.9, '당류': 9.1,
                 '나트륨': 570, '가격': 7000, '카테고리': '양식', '태그': '샌드위치,가벼운,저칼로리'},
                {'제품명': '스파이시 이탈리안', '식당명': '샌드위치 전문점', '중량': 224, '열량': 464, '단백질': 20.7, '포화지방': 9.1, '당류': 8.7,
                 '나트륨': 1250, '가격': 6500, '카테고리': '양식', '태그': '샌드위치,든든한'},
                {'제품명': 'K-바비큐', '식당명': '샌드위치 전문점', '중량': 256, '열량': 372, '단백질': 25.6, '포화지방': 2.1, '당류': 14.7,
                 '나트륨': 899, '가격': 7500, '카테고리': '양식', '태그': '샌드위치,한국풍'},

                # 추가 샘플 (다양한 카테고리)
                {'제품명': '김치찌개', '식당명': '한식당 A', '중량': 350, '열량': 450, '단백질': 25, '포화지방': 8, '당류': 5, '나트륨': 1200,
                 '가격': 7000, '카테고리': '한식', '태그': '국물,따뜻한,든든한'},
                {'제품명': '비빔밥', '식당명': '한식당 A', '중량': 400, '열량': 550, '단백질': 20, '포화지방': 6, '당류': 12, '나트륨': 800,
                 '가격': 8000, '카테고리': '한식', '태그': '건강한,영양균형'},
                {'제품명': '제육볶음', '식당명': '한식당 B', '중량': 300, '열량': 620, '단백질': 35, '포화지방': 15, '당류': 18, '나트륨': 1500,
                 '가격': 8500, '카테고리': '한식', '태그': '든든한,고단백'},
                {'제품명': '냉면', '식당명': '한식당 C', '중량': 450, '열량': 420, '단백질': 15, '포화지방': 3, '당류': 10, '나트륨': 900,
                 '가격': 9000, '카테고리': '한식', '태그': '시원한,여름'},
                {'제품명': '돈까스', '식당명': '일식당 A', '중량': 350, '열량': 750, '단백질': 30, '포화지방': 20, '당류': 8, '나트륨': 950,
                 '가격': 9000, '카테고리': '일식', '태그': '튀김,든든한'},
                {'제품명': '라멘', '식당명': '일식당 B', '중량': 500, '열량': 650, '단백질': 28, '포화지방': 18, '당류': 6, '나트륨': 2000,
                 '가격': 9500, '카테고리': '일식', '태그': '국물,면요리'},
            ])

        df.columns = df.columns.str.strip()
        # '단백질' 컬럼이 '탄백질'로 오타가 난 경우 수정
        if '탄백질' in df.columns and '단백질' not in df.columns:
            df.rename(columns={'탄백질': '단백질'}, inplace=True)

        if '태그' in df.columns:
            df['태그'] = df['태그'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

        self.menu_df = df
        print(f"✅ 메뉴 데이터 로드 완료: {len(df)}개")
        return df

    def extract_features(self):
        """
        영양성분 기반 피처 추출 및 정규화
        """
        if self.menu_df is None:
            print("오류: 메뉴 데이터가 로드되지 않았습니다.")
            return None

        df = self.menu_df.copy()
        numeric_features = ['중량', '열량', '단백질', '포화지방', '당류', '나트륨', '가격']

        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
            else:
                print(f"경고: '{col}' 컬럼이 없어 피처 추출에서 제외됩니다.")

        # 유효한 피처만 남기기
        numeric_features = [col for col in numeric_features if col in df.columns]

        # 파생 피처 생성 (필요한 컬럼이 모두 있는지 확인)
        if '단백질' in df.columns and '중량' in df.columns and df['중량'].nunique() > 0:
            df['단백질_비율'] = df['단백질'] / df['중량'].replace(0, np.nan)
            df['단백질_비율'] = df['단백질_비율'].fillna(0)

        if '열량' in df.columns and '중량' in df.columns and df['중량'].nunique() > 0:
            df['칼로리_밀도'] = df['열량'] / df['중량'].replace(0, np.nan)
            df['칼로리_밀도'] = df['칼로리_밀도'].fillna(0)

        if '나트륨' in df.columns:
            df['나트륨_레벨'] = pd.cut(df['나트륨'], bins=[-np.inf, 600, 1200, np.inf], labels=[0, 1, 2], right=True)

        if '열량' in df.columns:
            df['칼로리_레벨'] = pd.cut(df['열량'], bins=[-np.inf, 400, 600, np.inf], labels=[0, 1, 2], right=True)

        if '가격' in df.columns:
            df['가격_레벨'] = pd.cut(df['가격'], bins=[-np.inf, 7000, 9000, np.inf], labels=[0, 1, 2], right=True)

        if '카테고리' in df.columns:
            category_dummies = pd.get_dummies(df['카테고리'], prefix='카테고리')
            df = pd.concat([df, category_dummies], axis=1)

        if '태그' in df.columns:
            all_tags = set(tag for tags in df['태그'] for tag in tags)
            for tag in all_tags:
                df[f'태그_{tag}'] = df['태그'].apply(lambda x: 1 if tag in x else 0)

        feature_cols = numeric_features + ['단백질_비율', '칼로리_밀도']
        feature_cols = [col for col in feature_cols if col in df.columns]

        if feature_cols:
            # StandardScaler로 정규화
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            print("경고: 정규화할 수치형 피처가 없습니다.")

        tag_cols = [col for col in df.columns if col.startswith('태그_')]
        category_cols = [col for col in df.columns if col.startswith('카테고리_')]
        level_cols = ['나트륨_레벨', '칼로리_레벨', '가격_레벨']
        level_cols = [col for col in level_cols if col in df.columns]  # 존재하는 레벨 컬럼만

        all_feature_cols = feature_cols + tag_cols + category_cols + level_cols
        # 존재하지 않는 컬럼 최종 제외
        all_feature_cols = [col for col in all_feature_cols if col in df.columns]

        self.menu_features = df[all_feature_cols].values
        print(f"✅ 피처 추출 완료: {self.menu_features.shape[1]}개 피처")
        return df

    def calculate_scores(self, user_preferences: UserPreferences):
        """
        사용자 선호도 기반 메뉴별 점수 계산
        """
        if self.menu_df is None:
            raise ValueError("메뉴 데이터가 로드되지 않았습니다.")

        df = self.menu_df.copy()
        scores = np.zeros(len(df))

        # 1. 예산 적합도 (가장 중요)
        if user_preferences.budget and '가격' in df.columns:
            price_diff = abs(df['가격'] - user_preferences.budget)
            price_score = 1 - (price_diff / user_preferences.budget).clip(0, 1)
            scores += price_score * 3.0  # 가중치 3.0

        # 2. 카테고리 매칭
        if user_preferences.preferred_categories and '카테고리' in df.columns:
            for category in user_preferences.preferred_categories:
                category_match = df['카테고리'].str.contains(category, case=False, na=False)
                scores += category_match.astype(float) * 2.0  # 가중치 2.0

        # 3. 영양 선호도
        if user_preferences.prefer_high_protein and '단백질' in df.columns and '중량' in df.columns:
            protein_ratio = df['단백질'] / df['중량'].replace(0, np.nan)
            protein_ratio = protein_ratio.fillna(0)
            if protein_ratio.max() > protein_ratio.min():
                protein_score = (protein_ratio - protein_ratio.min()) / (protein_ratio.max() - protein_ratio.min())
                scores += protein_score * 1.5

        if user_preferences.prefer_low_calorie and '열량' in df.columns:
            if df['열량'].max() > df['열량'].min():
                calorie_score = 1 - ((df['열량'] - df['열량'].min()) / (df['열량'].max() - df['열량'].min()))
                scores += calorie_score * 1.5

        if user_preferences.prefer_low_sodium and '나트륨' in df.columns:
            if df['나트륨'].max() > df['나트륨'].min():
                sodium_score = 1 - ((df['나트륨'] - df['나트륨'].min()) / (df['나트륨'].max() - df['나트륨'].min()))
                scores += sodium_score * 1.5

        # 4. 알레르기 필터링 (점수 0으로)
        if user_preferences.allergens:
            pass  # (로직 추가 필요)

        # 5. 식사 시간대 적합도
        if '태그' in df.columns:
            meal_tags = {
                '아침': ['가벼운', '샌드위치', '샐러드'],
                '점심': ['든든한', '고단백', '영양균형'],
                '저녁': ['든든한', '국물', '따뜻한']
            }
            if user_preferences.meal_type in meal_tags:
                for tag in meal_tags[user_preferences.meal_type]:
                    tag_match = df['태그'].apply(lambda tags: tag in tags if isinstance(tags, list) else False)
                    scores += tag_match.astype(float) * 1.0

        # 정규화 (0-1)
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores


# --- 3. LangChain 및 LLM 설정 (From AI B) ---

# OpenAI API 키 설정 (환경 변수에서 불러오는 것을 권장)
# os.environ["OPENAI_API_KEY"] = "sk-..."

# 1. Output Parser (출력 파서): LLM의 응답을 'FinalRecommendation' JSON으로 강제
parser = PydanticOutputParser(pydantic_object=FinalRecommendation)

# 2. LLM 모델: GPT-4 mini 사용, temperature=0.6 (랜덤한 답변)
# (실제 실행 시 API 키가 환경변수에 설정되어 있어야 합니다)
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
except ImportError:
    print("langchain_openai가 설치되지 않았습니다. LLM 기능을 사용하려면 설치해주세요.")
    llm = None
except Exception as e:  # API 키가 없는 경우 등
    print(f"OpenAI LLM 초기화 실패: {e}. LLM 기능이 비활성화됩니다.")
    llm = None

# 가격, 끼니 정보를 모두 포함하도록 LLM 지시서 수정
prompt_template = """
당신은 고려대학교 근처 맛집 메뉴 추천 AI 'MenuMate'입니다.
사용자의 세부 요청과 AI A가 1차 필터링한 메뉴 후보 리스트를 받았습니다.

[후보 리스트] (이름, 가격, AI A 점수, 태그 순)
{candidates_str}

[사용자 세부 요청] (예: 속편한, 든든한)
"{user_prompt}"

[사용자 가격 제한]
{price_str}

[사용자가 추천받길 원하는 끼니]
{target_meals_str}

[지시]
1. [후보 리스트] 중에서 [사용자 세부 요청]과 [사용자 가격 제한]을 모두 만족하는 메뉴를 고르세요.
2. [사용자가 추천받길 원하는 끼니] 목록({target_meals_str})에 있는 슬롯에만 추천 메뉴를 채워주세요.
3. 각 후보의 'price' 정보를 [출력 JSON]의 'price' 필드에 정확히 기입해주세요.
5. 추천 메뉴를 선정한 후, [사용자 세부 요청]을 바탕으로 그 이유를 요약하는 1~3개의 'reason_hashtags'를 반드시 생성해주세요. (예: '#속편한', '#든든한', '#가성비')
6. 목록에 없는 다른 끼니 슬롯은 반드시 null로 설정해야 합니다.
7. 만약 사용자가 '점심'만 원했다면 'morning'과 'dinner'는 반드시 null이어야 합니다.

반드시 다음 JSON 형식으로만 응답해야 합니다 (다른 말은 절대 하지 마세요):
{format_instructions}
"""

# 3. 프롬프트 템플릿 완성
prompt = ChatPromptTemplate.from_template(
    template=prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. LangChain '체인' 구성 (입력 -> 프롬프트 -> LLM -> JSON 파서)
if llm:
    chain = prompt | llm | parser
else:
    chain = None
    print("LLM 체인 구성 실패. /recommend/full 엔드포인트가 작동하지 않을 수 있습니다.")

# --- 4. FastAPI 앱 및 캐싱 설정 (통합) ---

app = FastAPI(
    title="MenuMate AI - 통합 추천 모듈",
    description="영양성분 기반 후보 생성(A) 및 LLM 최종 추천(B) 통합 API"
)

# 전역 변수 (AI A와 B의 전역 변수 통합)
preprocessor: Optional[MenuDataPreprocessor] = None
cache: Optional[redis.Redis] = None


@app.on_event("startup")
async def startup():
    """서버 시작 시 데이터 로드, 전처리, 캐시 연결"""
    global preprocessor, cache

    # 1. AI A의 시작 로직
    preprocessor = MenuDataPreprocessor()
    preprocessor.load_nutrition_data()  # CSV 경로 지정 가능
    preprocessor.extract_features()
    print("✅ (AI A) 데이터 전처리 모듈 준비 완료!")

    # 2. AI B의 시작 로직 (Redis 캐시 연결)
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
        "version": "1.0.0",
        "llm_ready": (chain is not None),
        "preprocessor_ready": (preprocessor is not None and preprocessor.menu_df is not None)
    }


# --- 5. 내부 로직 함수 (기존 엔드포인트를 내부 함수로 변경) ---

async def _generate_candidates(preferences: UserPreferences) -> RecommendationRequest:
    """
    (구 AI A 로직) 사용자 선호도에 맞는 후보군 리스트 생성
    """
    if preprocessor is None or preprocessor.menu_df is None:
        raise HTTPException(status_code=503, detail="데이터 전처리기가 준비되지 않았습니다.")

    try:
        # 1. 점수 계산
        scores = preprocessor.calculate_scores(preferences)

        # 2. 상위 후보 선택 (top 20)
        df = preprocessor.menu_df.copy()
        df['base_score'] = scores

        # 예산 필터링
        if preferences.budget and '가격' in df.columns:
            df = df[df['가격'] <= preferences.budget * 1.2]  # 예산 20% 초과까지 허용

        # 점수순 정렬
        df = df.sort_values('base_score', ascending=False).head(20)

        # 3. AI B 형식(MenuCandidate)으로 변환
        candidates = []
        for _, row in df.iterrows():
            candidates.append(MenuCandidate(
                restaurant_name=row['식당명'],
                menu_name=row['제품명'],
                price=int(row['가격']),
                base_score=float(row['base_score']),
                tags=row['태그'] if isinstance(row['태그'], list) else []
            ))

        # 4. AI B로 전달할 요청 객체(RecommendationRequest) 생성
        ai_b_request = RecommendationRequest(
            candidates=candidates,
            user_prompt=preferences.user_prompt or "맛있고 건강한 메뉴",
            price=PriceRange(
                max=preferences.budget
            ) if preferences.budget else None,
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
    """
    if chain is None:
        raise HTTPException(status_code=503, detail="LLM 체인이 준비되지 않았습니다. (API 키 확인 필요)")

    # --- 1. 캐싱 (Caching) ---
    cache_key = None
    if cache:
        try:
            cache_key = f"recommend:{hash(request.model_dump_json())}"
            cached_result = cache.get(cache_key)
            if cached_result:
                print("캐시된 결과를 반환합니다.")
                return FinalRecommendation.model_validate_json(cached_result)
        except Exception as e:
            print(f"캐시 조회 중 오류 발생: {e}")  # 캐시 오류 시 그냥 LLM 호출 진행

    # --- 2. LLM 입력값 가공 ---
    candidates_str = "\n".join(
        [f"- {c.restaurant_name} '{c.menu_name}' (가격: {c.price}원, 점수: {c.base_score}, 태그: {c.tags})"
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

    # --- 3. LLM 체인 호출 (AI B의 핵심 작업) ---
    try:
        print(f"LLM 호출 시작: (User: {request.user_prompt}, Price: {price_str}, Meals: {target_meals_str})")
        result = await chain.ainvoke({
            "candidates_str": candidates_str,
            "user_prompt": request.user_prompt,
            "price_str": price_str,
            "target_meals_str": target_meals_str,
            "history": "\n".join(request.conversation_history or [])
        })

        # --- 4. 캐시에 결과 저장 ---
        if cache and cache_key:
            try:
                cache.set(cache_key, result.model_dump_json(), ex=3600)  # 1시간 캐시
                print("새 결과를 생성하고 캐시에 저장했습니다.")
            except Exception as e:
                print(f"캐시 저장 중 오류 발생: {e}")

        return result

    except Exception as e:
        print(f"LLM 파싱 또는 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"메뉴 추천(AI B) 실패: {str(e)}")


# --- 6. 통합 엔드포인트 ---

@app.post("/recommend/full", response_model=FinalRecommendation)
async def get_full_recommendation(preferences: UserPreferences):
    """
    사용자 선호도(UserPreferences)를 받아
    AI A(후보군 생성)와 AI B(LLM 정제)를 순차적으로 실행
    """
    try:
        # 1단계: AI A 로직 호출 (후보군 생성)
        print(f"1단계: 후보군 생성 시작 (User: {preferences.user_prompt})")
        candidate_request = await _generate_candidates(preferences)

        if not candidate_request.candidates:
            print("1단계 결과: 추천할 후보군이 없습니다.")
            # 후보가 없으면 LLM을 호출할 필요 없이 빈 결과를 반환
            return FinalRecommendation(morning=None, lunch=None, dinner=None)

        print(f"1단계 완료: {len(candidate_request.candidates)}개 후보 생성")

        # 2단계: AI B 로직 호출 (LLM 정제)
        print("2단계: LLM 정제 시작")
        final_recommendation = await _refine_recommendations(candidate_request)
        print("2단계 완료: 최종 추천 생성")

        return final_recommendation

    except HTTPException as he:
        # 이미 처리된 HTTP 예외는 그대로 전달
        raise he
    except Exception as e:
        # 기타 예외 처리
        print(f"전체 추천 파이프라인 오류: {e}")
        raise HTTPException(status_code=500, detail=f"전체 추천 프로세스 실패: {str(e)}")


# --- 7. (선택) 기존 헬퍼 엔드포인트 ---

@app.get("/menu/all")
async def get_all_menus():
    """(From AI A) 전체 메뉴 리스트 조회 (디버깅용)"""
    if preprocessor is None or preprocessor.menu_df is None:
        raise HTTPException(status_code=503, detail="데이터가 로드되지 않음")

    df = preprocessor.menu_df.copy()
    # Pydantic 또는 JSON 직렬화를 위해 list를 string으로 변환
    df['태그'] = df['태그'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')

    return df.to_dict(orient='records')


# --- 8. 서버 실행 ---

if __name__ == "__main__":
    # 통합 앱은 8000번 포트에서 실행
    uvicorn.run(app, host="127.0.0.1", port=8000)