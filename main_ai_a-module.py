import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from datetime import datetime

# ==================== Pydantic 모델 (AI A → AI B 연동) ====================

class MenuCandidate(BaseModel):
    """AI B로 전달할 메뉴 후보"""
    restaurant_name: str
    menu_name: str
    price: int
    base_score: float
    tags: List[str]

class PriceRange(BaseModel):
    """가격 범위"""
    min: Optional[int] = None
    max: Optional[int] = None

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

# ==================== 데이터 전처리 클래스 ====================

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
        
        CSV 형식 예시:
        제품명,식당명,중량(g),열량(kcal),단백질(g),포화지방(g),당류(g),나트륨(mg),가격,카테고리,태그
        """
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            # 샘플 데이터 (예시)
            df = pd.DataFrame([
                # 영양성분표 1 (한식)
                {'제품명': '치킨 텐더바이트', '식당명': '브랜드 A', '중량': 16, '열량': 222, '탄백질': 35, '포화지방': 2, '당류': 4, '나트륨': 222, '가격': 6500, '카테고리': '양식', '태그': '고단백,가벼운'},
                {'제품명': '치킨 마리네이드 샐러드', '식당명': '브랜드 A', '중량': 74, '열량': 492, '탄백질': 44, '포화지방': 6, '당류': 1, '나트륨': 492, '가격': 8500, '카테고리': '양식', '태그': '고단백,샐러드'},
                {'제품명': '더블 치킨 타르타르', '식당명': '브랜드 A', '중량': 71, '열량': 615, '탄백질': 71, '포화지방': 5, '당류': 10, '나트륨': 615, '가격': 9000, '카테고리': '양식', '태그': '든든한,고단백'},
                {'제품명': '치킨 칠리 샐러드', '식당명': '브랜드 A', '중량': 88, '열량': 591, '탄백질': 44, '포화지방': 7, '당류': 10, '나트륨': 591, '가격': 8500, '카테고리': '양식', '태그': '샐러드,고단백'},
                {'제품명': '비프 칠리 샐러드', '식당명': '브랜드 A', '중량': 89, '열량': 731, '탄백질': 51, '포화지방': 19, '당류': 9, '나트륨': 731, '가격': 9500, '카테고리': '양식', '태그': '든든한,고단백'},
                
                # 영양성분표 2 (서브웨이 스타일)
                {'제품명': '스파이시 바비큐', '식당명': '샌드위치 전문점', '중량': 256, '열량': 374, '탄백질': 25.2, '포화지방': 7.4, '당류': 15.0, '나트륨': 903, '가격': 6500, '카테고리': '양식', '태그': '샌드위치,간편'},
                {'제품명': '스파이시 쉬림프', '식당명': '샌드위치 전문점', '중량': 213, '열량': 245, '탄백질': 16.5, '포화지방': 0.9, '당류': 9.1, '나트륨': 570, '가격': 7000, '카테고리': '양식', '태그': '샌드위치,가벼운,저칼로리'},
                {'제품명': '스파이시 이탈리안', '식당명': '샌드위치 전문점', '중량': 224, '열량': 464, '탄백질': 20.7, '포화지방': 9.1, '당류': 8.7, '나트륨': 1250, '가격': 6500, '카테고리': '양식', '태그': '샌드위치,든든한'},
                {'제품명': 'K-바비큐', '식당명': '샌드위치 전문점', '중량': 256, '열량': 372, '탄백질': 25.6, '포화지방': 2.1, '당류': 14.7, '나트륨': 899, '가격': 7500, '카테고리': '양식', '태그': '샌드위치,한국풍'},
                
                # 추가 샘플 (다양한 카테고리)
                {'제품명': '김치찌개', '식당명': '한식당 A', '중량': 350, '열량': 450, '탄백질': 25, '포화지방': 8, '당류': 5, '나트륨': 1200, '가격': 7000, '카테고리': '한식', '태그': '국물,따뜻한,든든한'},
                {'제품명': '비빔밥', '식당명': '한식당 A', '중량': 400, '열량': 550, '탄백질': 20, '포화지방': 6, '당류': 12, '나트륨': 800, '가격': 8000, '카테고리': '한식', '태그': '건강한,영양균형'},
                {'제품명': '제육볶음', '식당명': '한식당 B', '중량': 300, '열량': 620, '탄백질': 35, '포화지방': 15, '당류': 18, '나트륨': 1500, '가격': 8500, '카테고리': '한식', '태그': '든든한,고단백'},
                {'제품명': '냉면', '식당명': '한식당 C', '중량': 450, '열량': 420, '탄백질': 15, '포화지방': 3, '당류': 10, '나트륨': 900, '가격': 9000, '카테고리': '한식', '태그': '시원한,여름'},
                {'제품명': '돈까스', '식당명': '일식당 A', '중량': 350, '열량': 750, '탄백질': 30, '포화지방': 20, '당류': 8, '나트륨': 950, '가격': 9000, '카테고리': '일식', '태그': '튀김,든든한'},
                {'제품명': '라멘', '식당명': '일식당 B', '중량': 500, '열량': 650, '탄백질': 28, '포화지방': 18, '당류': 6, '나트륨': 2000, '가격': 9500, '카테고리': '일식', '태그': '국물,면요리'},
            ])
        
        # 컬럼명 정리
        df.columns = df.columns.str.strip()
        
        # 태그를 리스트로 변환
        if '태그' in df.columns:
            df['태그'] = df['태그'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        
        self.menu_df = df
        print(f"✅ 메뉴 데이터 로드 완료: {len(df)}개")
        return df
    
    def extract_features(self):
        """
        영양성분 기반 피처 추출 및 정규화
        """
        df = self.menu_df.copy()
        
        # 1. 수치형 피처 (영양성분)
        numeric_features = ['중량', '열량', '단백질', '포화지방', '당류', '나트륨', '가격']
        
        # 결측치 처리
        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # 2. 파생 피처 생성
        # 단백질 비율 (g당 단백질)
        df['단백질_비율'] = df['단백질'] / df['중량']
        
        # 칼로리 밀도 (g당 칼로리)
        df['칼로리_밀도'] = df['열량'] / df['중량']
        
        # 나트륨 수준 (고/중/저)
        df['나트륨_레벨'] = pd.cut(df['나트륨'], bins=[0, 600, 1200, np.inf], labels=[0, 1, 2])
        
        # 칼로리 수준
        df['칼로리_레벨'] = pd.cut(df['열량'], bins=[0, 400, 600, np.inf], labels=[0, 1, 2])
        
        # 가격 수준
        df['가격_레벨'] = pd.cut(df['가격'], bins=[0, 7000, 9000, np.inf], labels=[0, 1, 2])
        
        # 3. 카테고리 원-핫 인코딩
        if '카테고리' in df.columns:
            category_dummies = pd.get_dummies(df['카테고리'], prefix='카테고리')
            df = pd.concat([df, category_dummies], axis=1)
        
        # 4. 태그 원-핫 인코딩
        if '태그' in df.columns:
            all_tags = set()
            for tags in df['태그']:
                all_tags.update(tags)
            
            for tag in all_tags:
                df[f'태그_{tag}'] = df['태그'].apply(lambda x: 1 if tag in x else 0)
        
        # 5. 정규화할 피처 선택
        feature_cols = numeric_features + ['단백질_비율', '칼로리_밀도']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # StandardScaler로 정규화
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        # 6. 최종 피처 벡터 생성
        tag_cols = [col for col in df.columns if col.startswith('태그_')]
        category_cols = [col for col in df.columns if col.startswith('카테고리_')]
        level_cols = ['나트륨_레벨', '칼로리_레벨', '가격_레벨']
        
        all_feature_cols = feature_cols + tag_cols + category_cols + level_cols
        self.menu_features = df[all_feature_cols].values
        
        print(f"✅ 피처 추출 완료: {self.menu_features.shape[1]}개 피처")
        
        return df
    
    def calculate_scores(self, user_preferences: UserPreferences):
        """
        사용자 선호도 기반 메뉴별 점수 계산
        """
        df = self.menu_df.copy()
        scores = np.zeros(len(df))
        
        # 1. 예산 적합도 (가장 중요)
        if user_preferences.budget:
            price_diff = abs(df['가격'] - user_preferences.budget)
            price_score = 1 - (price_diff / user_preferences.budget).clip(0, 1)
            scores += price_score * 3.0  # 가중치 3.0
        
        # 2. 카테고리 매칭
        if user_preferences.preferred_categories:
            for category in user_preferences.preferred_categories:
                category_match = df['카테고리'].str.contains(category, case=False, na=False)
                scores += category_match.astype(float) * 2.0  # 가중치 2.0
        
        # 3. 영양 선호도
        if user_preferences.prefer_high_protein:
            # 고단백 (단백질 비율 높은 순)
            protein_ratio = df['탄백질'] / df['중량']
            protein_score = (protein_ratio - protein_ratio.min()) / (protein_ratio.max() - protein_ratio.min())
            scores += protein_score * 1.5
        
        if user_preferences.prefer_low_calorie:
            # 저칼로리
            calorie_score = 1 - ((df['열량'] - df['열량'].min()) / (df['열량'].max() - df['열량'].min()))
            scores += calorie_score * 1.5
        
        if user_preferences.prefer_low_sodium:
            # 저나트륨
            sodium_score = 1 - ((df['나트륨'] - df['나트륨'].min()) / (df['나트륨'].max() - df['나트륨'].min()))
            scores += sodium_score * 1.5
        
        # 4. 알레르기 필터링 (점수 0으로)
        if user_preferences.allergens:
            # 필요시 추가
            pass
        
        # 5. 식사 시간대 적합도
        meal_tags = {
            '아침': ['가벼운', '샌드위치', '샐러드'],
            '점심': ['든든한', '고단백', '영양균형'],
            '저녁': ['든든한', '국물', '따뜻한']
        }
        
        if user_preferences.meal_type in meal_tags:
            for tag in meal_tags[user_preferences.meal_type]:
                tag_match = df['태그'].apply(lambda tags: tag in tags)
                scores += tag_match.astype(float) * 1.0
        
        # 정규화 (0-1)
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores

# ==================== FastAPI 앱 ====================

app = FastAPI(
    title="MenuMate AI A - 추천 모델",
    description="영양성분 기반 메뉴 추천 및 후보 생성"
)

# 전역 변수
preprocessor = None

@app.on_event("startup")
async def startup():
    """서버 시작 시 데이터 로드 및 전처리"""
    global preprocessor
    
    preprocessor = MenuDataPreprocessor()
    preprocessor.load_nutrition_data()  # CSV 경로 지정 가능
    preprocessor.extract_features()
    
    print("✅ AI A 모듈 준비 완료!")

@app.get("/")
async def root():
    return {
        "service": "MenuMate AI A",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/recommend/candidates", response_model=dict)
async def get_menu_candidates(preferences: UserPreferences):
    """
    AI B로 전달할 메뉴 후보 리스트 생성
    
    반환 형식: AI B의 RecommendationRequest에 맞춤
    """
    try:
        # 1. 점수 계산
        scores = preprocessor.calculate_scores(preferences)
        
        # 2. 상위 후보 선택 (top 20)
        df = preprocessor.menu_df.copy()
        df['base_score'] = scores
        
        # 예산 필터링
        if preferences.budget:
            df = df[df['가격'] <= preferences.budget * 1.2]  # 예산 20% 초과까지 허용
        
        # 점수순 정렬
        df = df.sort_values('base_score', ascending=False).head(20)
        
        # 3. AI B 형식으로 변환
        candidates = []
        for _, row in df.iterrows():
            candidates.append(MenuCandidate(
                restaurant_name=row['식당명'],
                menu_name=row['제품명'],
                price=int(row['가격']),
                base_score=float(row['base_score']),
                tags=row['태그']
            ))
        
        # 4. AI B로 전달할 요청 객체 생성
        ai_b_request = {
            "candidates": [c.model_dump() for c in candidates],
            "user_prompt": preferences.user_prompt or "맛있고 건강한 메뉴",
            "price": {
                "max": preferences.budget
            } if preferences.budget else None,
            "target_meals": preferences.target_meals,
            "conversation_history": []
        }
        
        return ai_b_request
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 생성 실패: {str(e)}")

@app.get("/menu/all")
async def get_all_menus():
    """전체 메뉴 리스트 조회"""
    if preprocessor is None or preprocessor.menu_df is None:
        raise HTTPException(status_code=503, detail="데이터가 로드되지 않음")
    
    df = preprocessor.menu_df.copy()
    
    # 태그를 문자열로 변환
    df['태그'] = df['태그'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    
    return df.to_dict(orient='records')

# ==================== 서버 실행 ====================

if __name__ == "__main__":
    import uvicorn
    # AI A는 8001 포트에서 실행
    uvicorn.run(app, host="127.0.0.1", port=8001)