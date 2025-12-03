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


# --- 1. Pydantic 모델 정의 (수정됨) ---

# 다이어트 정보 모델
class DietInfo(BaseModel):
    height: Optional[int] = Field(default=None, description="키 (cm)")
    weight: Optional[int] = Field(default=None, description="몸무게 (kg)")


# 가격 범위 모델
class PriceRangeInput(BaseModel):
    minPrice: Optional[int] = Field(default=None, description="최소 가격")
    maxPrice: Optional[int] = Field(default=None, description="최대 가격")


# [수정] 내부 데이터 Payload (Node.js 스키마와 일치시킴)
class UserPayload(BaseModel):
    category: Optional[str] = Field(default=None, description="카테고리: 'DIET', 'VEGETARIAN', 'HALAL' 등")
    dietInfo: Optional[DietInfo] = Field(default=None, description="다이어트 정보")
    meals: List[str] = Field(description="추천받을 식사: ['BREAKFAST', 'LUNCH', 'DINNER']")
    price: Optional[PriceRangeInput] = Field(default=None, description="가격 범위")
    prompt: Optional[str] = Field(default="", description="사용자 요청사항")
    campus: List[str] = Field(default=[], description="위치: ['science_campus', 'humanities_campus']")


# [수정] 외부 껍데기 ({"user": ...} 구조 대응)
class UserRequestWrapper(BaseModel):
    user: UserPayload


# UserPreferences: 내부 로직 처리용 모델
class UserPreferences(BaseModel):
    budget: Optional[int] = Field(default=10000)
    min_budget: Optional[int] = Field(default=None)
    meal_type: str = Field(default="점심")
    target_meals: List[str] = Field(default=["lunch"])
    user_prompt: str = Field(default="")

    # 영양/카테고리 관련
    prefer_low_calorie: bool = Field(default=False)
    category: Optional[str] = Field(default=None)
    height: Optional[int] = Field(default=None)
    weight: Optional[int] = Field(default=None)
    location: List[str] = Field(default=[])

    # 사용하지 않는 필드 (호환성 유지)
    preferred_categories: List[str] = []
    prefer_high_protein: bool = False
    prefer_low_sodium: bool = False


# AI A와 B가 공통으로 사용하는 모델들
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
    price: Optional[PriceRange] = Field(default=None)
    target_meals: List[str]
    conversation_history: Optional[List[str]] = []


class RecommendedMenu(BaseModel):
    restaurant_name: str = Field(description="식당 이름")
    menu_name: str = Field(description="메뉴 이름")
    price: int = Field(description="메뉴의 실제 가격")
    justification: str = Field(description="이 메뉴를 추천하는 이유 (영업시간 고려 포함)")
    new_score: float = Field(description="LLM이 재평가한 최종 점수 (0.0 ~ 1.0)")
    reason_hashtags: List[str] = Field(description="추천 이유를 요약하는 3-5개의 해시태그")


class FinalRecommendation(BaseModel):
    morning: Optional[RecommendedMenu] = Field(default=None)
    lunch: Optional[RecommendedMenu] = Field(default=None)
    dinner: Optional[RecommendedMenu] = Field(default=None)


# --- 2. 입력 변환 함수 (수정됨) ---

def convert_user_input(payload: UserPayload) -> UserPreferences:
    """
    UserPayload -> UserPreferences 변환
    - 식사 시간 우선순위 조정 (점심 우선)
    - 카테고리 매핑 (VEGETARIAN -> VEGAN 등)
    """
    # 1. 식사 타겟 문자열 변환
    meal_mapping = {
        "BREAKFAST": "morning",
        "LUNCH": "lunch",
        "DINNER": "dinner"
    }
    target_meals = [meal_mapping.get(m.upper(), m.lower()) for m in payload.meals]

    # 2. 기준 시간대 설정 (데이터가 많은 '점심' 우선)
    if "lunch" in target_meals:
        meal_type = "점심"
    elif "dinner" in target_meals:
        meal_type = "저녁"
    else:
        meal_type = "아침"

    # 3. 카테고리 매핑
    category_map = {
        "VEGETARIAN": "VEGAN",
        "HALAL": "MUSLIM",
        "DIET": "DIET",
        "LOW_SUGAR": "LOW_SUGAR"
    }
    raw_category = payload.category.upper() if payload.category else ""
    mapped_category = category_map.get(raw_category, raw_category)

    # 4. DIET 플래그
    prefer_low_calorie = (mapped_category == "DIET")

    # 5. 가격 처리
    max_price = 10000
    min_price = None
    if payload.price:
        if payload.price.maxPrice:
            max_price = payload.price.maxPrice
        if payload.price.minPrice:
            min_price = payload.price.minPrice

    # 6. 키/몸무게 처리
    u_height = payload.dietInfo.height if payload.dietInfo else None
    u_weight = payload.dietInfo.weight if payload.dietInfo else None

    return UserPreferences(
        budget=max_price,
        min_budget=min_price,
        meal_type=meal_type,
        target_meals=target_meals,
        user_prompt=payload.prompt or "",
        prefer_low_calorie=prefer_low_calorie,
        category=mapped_category,
        height=u_height,
        weight=u_weight,
        location=payload.campus  # campus -> location
    )


# --- 3. 데이터 전처리 클래스 ---

class MenuDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.menu_features = None
        self.menu_df = None
        self.db_engine = self._create_db_engine()

    def _create_db_engine(self):
        load_dotenv()
        db_user = os.environ.get("DB_USER")
        db_pass = os.environ.get("DB_PASSWORD")
        db_host = os.environ.get("DB_HOST")
        db_port = os.environ.get("DB_PORT")
        db_name = os.environ.get("DB_NAME")

        if not all([db_user, db_pass, db_host, db_port, db_name]):
            print("경고: DB 접속 정보(.env) 불완전. 샘플 데이터 사용 예정.")
            return None

        DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        try:
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ DB 연결 성공")
            return engine
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            return None

    def load_nutrition_data(self):
        print("🔄 [DEBUG] 데이터 로드 시작...")

        if self.db_engine is None:
            print("⚠️ DB 엔진 없음 -> 샘플 데이터 사용")
            df = self._get_sample_data()
        else:
            # SQL 쿼리 (기존과 동일)
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
                print(f"✅ DB 쿼리 성공: 총 {len(df)}개 행 가져옴")

                # [핵심 수정] DB에는 연결됐는데 데이터가 0개인 경우 -> 샘플 사용
                if len(df) == 0:
                    print("⚠️ 가져온 데이터가 0개입니다! (테이블이 비었거나 조인 실패)")
                    print("👉 강제로 샘플 데이터를 로드합니다.")
                    df = self._get_sample_data()

            except Exception as e:
                print(f"❌ 쿼리 실패 에러: {e}")
                df = self._get_sample_data()

        # --- 전처리 로직 (데이터가 있어야 수행) ---
        if df is None or len(df) == 0:
            print("❌ [CRITICAL] 사용할 데이터가 전혀 없습니다.")
            self.menu_df = None
            return None

        # 1. 시간 변환
        def safe_time_convert(x):
            if pd.isna(x): return None
            if isinstance(x, time): return x
            try:
                return pd.to_datetime(str(x), format='%H:%M:%S').time()
            except:
                return None

        df['오픈시간'] = df['오픈시간'].apply(safe_time_convert)
        df['마감시간'] = df['마감시간'].apply(safe_time_convert)

        # 2. 평점/캠퍼스 처리
        df['평점'] = pd.to_numeric(df['평점'], errors='coerce').fillna(3.0)

        def clean_campus(x):
            if isinstance(x, list): return str(x[0]) if len(x) > 0 else '정보없음'
            if pd.isna(x): return '정보없음'
            return str(x)

        df['캠퍼스'] = df['캠퍼스'].apply(clean_campus)

        # 3. 열량 처리
        df['열량'] = pd.to_numeric(df['열량'], errors='coerce')
        df['열량'] = df['열량'].fillna(df['열량'].median() if not df['열량'].isna().all() else 500)

        # 4. 태그/카테고리 처리
        if '태그' in df.columns:
            df['태그'] = df['태그'].apply(
                lambda x: x if isinstance(x, list) else (str(x).split(',') if pd.notna(x) and x else [])
            )
        else:
            df['태그'] = [[] for _ in range(len(df))]

        if '카테고리' in df.columns:
            df['카테고리'] = df['태그']

        self.menu_df = df
        print(f"✅ [DEBUG] 최종 전처리 완료 데이터 개수: {len(self.menu_df)}")
        return df

    def _get_sample_data(self):
        # 샘플 데이터 (DB 연결 실패 시)
        return pd.DataFrame([
            {'제품명': '샘플 비건 비빔밥', '식당명': '한식당 A', '가격': 8000, '열량': 550, '카테고리': ['채식'], '태그': ['채식', '비건', '밥'],
             '평점': 4.2, '오픈시간': '10:00:00', '마감시간': '22:00:00', '캠퍼스': '인문계캠퍼스'},
        ])

    def extract_features(self):
        # (간소화) 피처 추출 로직은 필요 시 원본 유지
        if self.menu_df is not None:
            # 여기서는 단순히 df만 있으면 됨
            pass

    def calculate_scores(self, user_preferences: UserPreferences):
        print("🔄 [DEBUG] 점수 계산 시작...")
        if self.menu_df is None:
            raise ValueError("메뉴 데이터 없음")

        df = self.menu_df.copy()

        # [중요] 모든 메뉴는 기본 1점부터 시작 (절대 0점이 되지 않음)
        scores = np.ones(len(df)) * 1.0

        # --- Soft Filters (가산점만 적용, 감점 없음) ---

        # (A) 카테고리 가산점
        if user_preferences.category:
            cat_upper = user_preferences.category.upper()
            if cat_upper == "VEGAN":
                # 태그에 비건/채식 있으면 +5점
                is_vegan = df['태그'].apply(lambda tags: '채식' in tags or '비건' in tags)
                scores[is_vegan] += 5.0
            elif cat_upper == "MUSLIM":
                is_muslim = df['태그'].apply(lambda tags: '할랄' in tags or '무슬림' in tags)
                scores[is_muslim] += 5.0

        # (B) 캠퍼스 가산점
        if user_preferences.location:
            location_map = {'science_campus': '자연계캠퍼스', 'humanities_campus': '인문계캠퍼스'}
            target_campuses = [location_map.get(loc) for loc in user_preferences.location if location_map.get(loc)]
            if target_campuses:
                is_match = df['캠퍼스'].apply(lambda c: (c in target_campuses) or (c == '둘다'))
                scores[is_match] += 2.0

                # (C) 예산 가산점
        if user_preferences.budget:
            within = df['가격'] <= user_preferences.budget
            scores[within] += 3.0

        print(f"✅ [DEBUG] 점수 계산 완료. (평균 점수: {scores.mean():.2f})")
        return scores


# --- 4. LangChain 및 LLM 설정 (수정됨) ---

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
parser = PydanticOutputParser(pydantic_object=FinalRecommendation)

if OPENAI_API_KEY:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=OPENAI_API_KEY)
else:
    llm = None
    print("경고: LLM 키 없음")

# [수정] 영업시간 유연성 반영 프롬프트
prompt_template = """
당신은 고려대학교 근처 맛집 메뉴 추천 AI 'MenuMate'입니다.
사용자의 요청에 맞춰 AI A가 선별한 메뉴 후보들을 검토하여 최종 추천을 해주세요.

[후보 리스트] (메뉴명, 가격, 점수, 특징(영업시간 포함))
{candidates_str}

[사용자 요청]
- 세부 요청: "{user_prompt}"
- 가격 제한: {price_str}
- 원하는 끼니: {target_meals_str}

[지시사항]
1. **영업시간 확인 및 유연한 적용:**
   - [후보 리스트]의 태그에 있는 '영업시간'을 확인하세요.
   - 사용자가 원하는 끼니 시간대(아침/점심/저녁)에 방문 가능한지 판단하세요.
   - **중요:** 정확한 시간대가 아니더라도, 오픈 시간이 조금 늦거나 빨라도 메뉴가 훌륭하다면 추천하세요. (예: 10:30 오픈이어도 늦은 아침 메뉴로 추천 가능)

2. **추천 이유(justification) 작성:**
   - 추천하는 이유를 대략 40자 정도로 적으세요.
   - 만약 영업시간이 조금 애매하다면, 그 내용을 포함해 주세요. (예: "10시 오픈이라 조금 늦지만, 최고의 비건 샌드위치입니다.")

3. **슬롯 채우기:**
   - [사용자가 원하는 끼니] 목록({target_meals_str})에 해당하는 슬롯은 최대한 채워주세요. (Null 반환 지양)

4. 사용자 세부 요청("{user_prompt}")을 반영하여 메뉴를 선정하세요.

반드시 다음 JSON 형식으로만 응답해야 합니다:
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

# --- 5. FastAPI 앱 설정 ---

app = FastAPI()
preprocessor: Optional[MenuDataPreprocessor] = None
cache: Optional[redis.Redis] = None


@app.on_event("startup")
async def startup():
    global preprocessor, cache
    preprocessor = MenuDataPreprocessor()
    preprocessor.load_nutrition_data()
    print("✅ 데이터 전처리 준비 완료")

    try:
        cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        cache.ping()
        print("✅ Redis 연결 성공")
    except:
        print("⚠️ Redis 연결 실패 (캐싱 미사용)")
        cache = None


# --- 6. 내부 로직 함수 (수정됨) ---

async def _generate_candidates(preferences: UserPreferences) -> RecommendationRequest:
    if preprocessor is None or preprocessor.menu_df is None:
        raise HTTPException(status_code=503, detail="데이터 준비 안됨")

    # 점수 계산 (시간 필터 꺼짐)
    scores = preprocessor.calculate_scores(preferences)
    df = preprocessor.menu_df.copy()
    df['base_score'] = scores

    # 상위 20개 추출
    df_filtered = df[df['base_score'] > 0]
    df_sorted = df_filtered.sort_values('base_score', ascending=False).head(8)

    candidates = []
    for _, row in df_sorted.iterrows():
        tags_list = row['태그'] if isinstance(row['태그'], list) else []

        # [수정] 영업시간 정보를 태그에 텍스트로 추가 (LLM 전달용)
        open_t = str(row['오픈시간'])[:5] if row['오픈시간'] else "??"
        close_t = str(row['마감시간'])[:5] if row['마감시간'] else "??"
        tags_list.append(f"영업시간:{open_t}~{close_t}")

        candidates.append(MenuCandidate(
            restaurant_name=row['식당명'],
            menu_name=row['제품명'],
            price=int(row['가격']),
            base_score=float(row['base_score']),
            tags=tags_list
        ))

    return RecommendationRequest(
        candidates=candidates,
        user_prompt=preferences.user_prompt,
        price=PriceRange(min=preferences.min_budget, max=preferences.budget),
        target_meals=preferences.target_meals
    )


async def _refine_recommendations(request: RecommendationRequest) -> FinalRecommendation:
    if chain is None:
        raise HTTPException(status_code=503, detail="LLM 설정 안됨")

    # 캐시 로직 (생략 가능하나 유지)
    cache_key = f"rec:{hash(request.model_dump_json())}" if cache else None
    if cache and cache.get(cache_key):
        return FinalRecommendation.model_validate_json(cache.get(cache_key))

    # LLM 입력 문자열 생성
    candidates_str = "\n".join([
        f"- {c.restaurant_name} '{c.menu_name}' (가격: {c.price}, 점수: {c.base_score:.2f}, 특징: {c.tags})"
        for c in request.candidates
    ])

    price_str = f"{request.price.min or 0} ~ {request.price.max or '무제한'}원" if request.price else "제한 없음"
    target_meals_str = ", ".join(request.target_meals)

    try:
        result = await chain.ainvoke({
            "candidates_str": candidates_str,
            "user_prompt": request.user_prompt,
            "price_str": price_str,
            "target_meals_str": target_meals_str
        })

        if cache and cache_key:
            cache.set(cache_key, result.model_dump_json(), ex=1800)

        return result
    except Exception as e:
        print(f"LLM 오류: {e}")
        # 오류 시 빈 객체 반환
        return FinalRecommendation()


# --- 7. 엔드포인트 (수정됨) ---

@app.post("/recommend", response_model=FinalRecommendation)
async def get_recommendation(wrapper: UserRequestWrapper):  # Wrapper 사용
    try:
        user_payload = wrapper.user
        print(f"📥 요청 수신: {user_payload.category}, {user_payload.meals}")

        # 1. 변환
        preferences = convert_user_input(user_payload)

        # 2. 후보 생성 (AI A)
        req = await _generate_candidates(preferences)
        if not req.candidates:
            print("⚠️ 후보군 없음 (조건 완화 필요)")
            return FinalRecommendation()

        # 3. 정제 (AI B)
        final_res = await _refine_recommendations(req)
        print("✅ 추천 완료")
        return final_res

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)