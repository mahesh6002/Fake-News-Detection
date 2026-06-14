from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import joblib
import time
import sys
import os
import re
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from api.schemas import PredictRequest, PredictResponse, HealthResponse, ModelInfoResponse, VerifyResponse, VerifyArticleResponse
from src.preprocessing import clean_text

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Fake News Detector API",
    description="Classify news articles as Fake or Real using ML",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/pipeline.pkl")
pipeline = None

@app.on_event("startup")
async def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Could not load model: {e}", file=sys.stderr)

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("20/minute")
async def predict(request: Request, body: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    clean = clean_text(body.text)

    if len(clean.strip()) < 5:
        raise HTTPException(status_code=422, detail="Article text too short after preprocessing")

    label_id = pipeline.predict([clean])[0]
    proba = pipeline.predict_proba([clean])[0]
    confidence = float(max(proba)) * 100
    label = "REAL" if label_id == 1 else "FAKE"
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return PredictResponse(
        label=label,
        confidence=round(confidence, 2),
        fake_probability=round(float(proba[0]) * 100, 2),
        real_probability=round(float(proba[1]) * 100, 2),
        processing_time_ms=elapsed_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if pipeline is not None else "degraded",
        model_loaded=pipeline is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    clf = pipeline.named_steps["clf"]
    tfidf = pipeline.named_steps["tfidf"]
    return ModelInfoResponse(
        model_type=type(clf).__name__,
        vocabulary_size=len(tfidf.vocabulary_),
        n_features=tfidf.max_features,
    )

@app.get("/verify", response_model=VerifyResponse)
async def verify(query: str, from_date: Optional[str] = None):
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return VerifyResponse(articles=[])

    if not query.strip():
        return VerifyResponse(articles=[])

    headers = {"User-Agent": "VeritasNewsDetector/1.0"}
    
    async def fetch_news(url: str, params: dict):
        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 200:
                    return resp.json().get("articles", [])
        except Exception as e:
            print(f"Error fetching news from {url}: {e}", file=sys.stderr)
        return []

    # Query parameters
    top_params = {"q": query, "apiKey": api_key}
    everything_params = {"q": query, "apiKey": api_key}
    if from_date:
        everything_params["from"] = from_date

    import asyncio
    results = await asyncio.gather(
        fetch_news("https://newsapi.org/v2/top-headlines", top_params),
        fetch_news("https://newsapi.org/v2/everything", everything_params)
    )
    
    raw_articles = results[0] + results[1]
    
    seen_urls = set()
    unique_articles = []
    
    # Pre-process query keywords for relevance score
    q_words = set(re.findall(r"\w+", query.lower()))
    q_words = {w for w in q_words if len(w) > 2}
    
    for art in raw_articles:
        url = art.get("url")
        title = art.get("title")
        if not url or not title or url in seen_urls:
            continue
        # Ignore removed articles
        if "[removed]" in title.lower() or "[removed]" in url.lower():
            continue
        seen_urls.add(url)
        
        t_words = set(re.findall(r"\w+", title.lower()))
        
        if not q_words:
            relevance = 0.0
        else:
            relevance = (len(q_words.intersection(t_words)) / len(q_words)) * 100.0
            
        unique_articles.append({
            "title": title,
            "source": art.get("source", {}).get("name", "Unknown Source"),
            "published_at": art.get("publishedAt", ""),
            "url": url,
            "relevance_score": round(relevance, 2)
        })
        
    # Sort by relevance descending, then by date descending
    unique_articles.sort(key=lambda x: (x["relevance_score"], x["published_at"]), reverse=True)
    
    # Return top 5 articles
    top_5 = unique_articles[:5]
    return VerifyResponse(articles=[VerifyArticleResponse(**art) for art in top_5])

# Mount static frontend (checks if directory exists)
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
