import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai

app = FastAPI(title="Perfscanimal Health Backend", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Models
class LocationData(BaseModel):
    region: str
    department: str
    commune: str

class HealthAnalysisRequest(BaseModel):
    species: str
    symptoms: List[str]
    clinical_data: Optional[dict] = None

class TreatmentRequest(BaseModel):
    species: str
    diseases: List[str]
    location: LocationData

class ReportRequest(BaseModel):
    species: str
    diseases: List[str]
    location: LocationData

# Routes
@app.get("/")
async def root():
    return {"message": "Perfscanimal Health Backend API", "version": "1.0.0"}

@app.post("/api/analyze-health")
async def analyze_health(request: HealthAnalysisRequest):
    """Analyze animal health using Gemini AI"""
    try:
        prompt = f"Analyze the health condition of a {request.species} with symptoms: {', '.join(request.symptoms)}. Provide diagnosis and recommendations."
        if GEMINI_API_KEY:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return {"analysis": response.text, "success": True}
        else:
            return {"analysis": f"Analysis for {request.species}: Symptoms detected - {', '.join(request.symptoms)}", "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/api/generate-treatment")
async def generate_treatment(request: TreatmentRequest):
    """Generate treatment plan for confirmed diseases"""
    try:
        prompt = f"Generate a detailed treatment plan for a {request.species} with {', '.join(request.diseases)}. Location: {request.location.commune}, {request.location.department}, {request.location.region}."
        if GEMINI_API_KEY:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return {"treatment_plan": response.text, "success": True}
        else:
            return {"treatment_plan": f"Treatment plan generated for {', '.join(request.diseases)} in {request.location.commune}", "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/api/report-case")
async def report_case(request: ReportRequest):
    """Report epidemiological data"""
    try:
        return {
            "message": "Case reported successfully",
            "data": {
                "species": request.species,
                "diseases": request.diseases,
                "location": request.location.dict()
            },
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
