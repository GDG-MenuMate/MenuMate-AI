import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import redis
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# --- 1. Pydantic 모델 (API 입/출력 명세) ---

#   BE B/FE가 요청한대로 'price' 필드 추가
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

#   UI/BE B 요청대로 List가 아닌 '끼니 슬롯' 형태로 변경
class FinalRecommendation(BaseModel):
    """최종 API 응답 형식 (FE의 Result 화면과 일치)"""
    morning: Optional[RecommendedMenu] = Field(
        default=None, description="추천 아침 메뉴 (해당 없으면 null)"
    )
    lunch: Optional[RecommendedMenu] = Field(
        default=None, description="추천 점심 메뉴 (해당 없으면 null)"
    )
    dinner: Optional[RecommendedMenu] = Field(
        default=None, description="추천 저녁 메뉴 (해당 없으면 null)"
    )


#   AI A / BE A가 전달할 '가격' 정보 추가
class MenuCandidate(BaseModel):
    """메인 BE(or AI A)에서 전달받을 1차 필터링 후보"""
    restaurant_name: str
    menu_name: str
    price: int  # LLM이 가격을 고려할 수 있도록 메뉴의 실제 가격
    base_score: float  # AI A가 매긴 기본 점수
    tags: List[str]


#   BE A가 요청한 가격 범위 객체
class PriceRange(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None


#   BE A가 요청한 모든 정보를 받는 메인 객체
class RecommendationRequest(BaseModel):
    """AI B 모듈이 받을 요청 Body 전체"""
    candidates: List[MenuCandidate]
    user_prompt: str

    price: Optional[PriceRange] = Field(
        default=None, description="사용자 가격 제한 (예: {'max': 10000})"
    )
    target_meals: List[str] = Field(
        description="추천받고자 하는 끼니 목록 (예: ['lunch', 'dinner'])"
    )

    conversation_history: Optional[List[str]] = []  # (확장용) 대화 이력


# --- 2. LangChain 및 LLM 설정 ---

# OpenAI API 키 설정 (환경 변수에서 불러오는 것을 권장)
# os.environ["OPENAI_API_KEY"] = "sk-..."

# 1. Output Parser (출력 파서): LLM의 응답을 'FinalRecommendation' JSON으로 강제
parser = PydanticOutputParser(pydantic_object=FinalRecommendation)

# 2. LLM 모델: GPT-4 mini 사용, temperature=0.6 (랜덤한 답변)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

# 💡 (수정 6: 프롬프트) 가격, 끼니 정보를 모두 포함하도록 LLM 지시서 수정
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
4. 목록에 없는 다른 끼니 슬롯은 반드시 null로 설정해야 합니다.
5. 만약 사용자가 '점심'만 원했다면 'morning'과 'dinner'는 반드시 null이어야 합니다.

반드시 다음 JSON 형식으로만 응답해야 합니다 (다른 말은 절대 하지 마세요):
{format_instructions}
"""

# 3. 프롬프트 템플릿 완성
prompt = ChatPromptTemplate.from_template(
    template=prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. LangChain '체인' 구성 (입력 -> 프롬프트 -> LLM -> JSON 파서)
chain = prompt | llm | parser

# --- 3. FastAPI 앱 및 캐싱 설정 ---

app = FastAPI(
    title="MenuMate - AI B Module",
    description="LLM을 이용한 메뉴 최종 추천 및 후처리 API"
)

# Redis 클라이언트 (실제 운영 시 host, port, password 등 설정 필요)
try:
    cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    cache.ping()
    print("Redis 캐시 서버에 연결되었습니다.")
except redis.ConnectionError as e:
    print(f"경고: Redis 서버에 연결할 수 없습니다. 캐싱이 비활성화됩니다. (오류: {e})")
    cache = None  # 캐시 연결 실패 시 None으로 설정


@app.post("/recommend/refine", response_model=FinalRecommendation)
async def get_refined_recommendations(request: RecommendationRequest):
    """
    AI A의 후보 리스트와 사용자 프롬프트를 받아 LLM으로 최종 메뉴를 추천합니다.
    """

    # --- 1. 캐싱 (Caching) ---
    cache_key = None
    if cache:
        try:
            # 요청 객체를 기반으로 고유한 캐시 키 생성 (안정적인 해시 필요)
            cache_key = f"recommend:{hash(request.model_dump_json())}"
            cached_result = cache.get(cache_key)
            if cached_result:
                print("캐시된 결과를 반환합니다.")
                return FinalRecommendation.model_validate_json(cached_result)
        except Exception as e:
            print(f"캐시 조회 중 오류 발생: {e}")
            # 캐시 오류 시, 그냥 LLM 호출 진행

    # --- 2. LLM 입력값 가공 ---

    #   후보 리스트 문자열에 '가격' 포함
    candidates_str = "\n".join(
        [f"- {c.restaurant_name} '{c.menu_name}' (가격: {c.price}원, 점수: {c.base_score}, 태그: {c.tags})"
         for c in request.candidates]
    )

    #   가격 제한 객체를 LLM이 알아들을 문자열로 변환
    price_str = "제한 없음"
    if request.price:
        parts = []
        if request.price.min is not None:
            parts.append(f"{request.price.min}원 이상")
        if request.price.max is not None:
            parts.append(f"{request.price.max}원 이하")
        if parts:
            price_str = " ".join(parts)

    #   목표 끼니 리스트를 문자열로 변환
    target_meals_str = ", ".join(request.target_meals)  # 예: "lunch, dinner"

    # --- 3. LLM 체인 호출 (AI B의 핵심 작업) ---
    try:
        print(f"LLM 호출 시작: (User: {request.user_prompt}, Price: {price_str}, Meals: {target_meals_str})")
        result = await chain.ainvoke({
            "candidates_str": candidates_str,
            "user_prompt": request.user_prompt,
            "price_str": price_str,
            "target_meals_str": target_meals_str,
            "history": "\n".join(request.conversation_history)
        })

        # --- 4. 캐시에 결과 저장 ---
        if cache and cache_key:
            try:
                # LLM API 비용 절약을 위해 1시간(3600초) 동안 캐시
                cache.set(cache_key, result.model_dump_json(), ex=3600)
                print("새 결과를 생성하고 캐시에 저장했습니다.")
            except Exception as e:
                print(f"캐시 저장 중 오류 발생: {e}")

        # --- 5. FE로 최종 JSON 반환 ---
        return result

    except Exception as e:
        # --- 6. 오류 처리 ---
        print(f"LLM 파싱 또는 API 오류: {e}")
        # (실제 운영 시) 여기에 오류 로깅(Logging) 로직 추가
        raise HTTPException(status_code=500, detail=f"메뉴 추천에 실패했습니다. (AI B 모듈 오류: {e})")


# --- 4. (선택) 서버 실행 코드 ---
if __name__ == "__main__":
    import uvicorn

    # PyCharm에서 '실행' 버튼을 누르는 대신, 터미널에서
    # uvicorn main_ai_b_module:app --reload --port 8000
    # 명령으로 실행하는 것을 권장합니다.
    uvicorn.run(app, host="127.0.0.1", port=8000)