from fastapi import FastAPI
from pydantic import BaseModel
from crews.competitor_crew import run_competitor_crew
from crews.opportunity_crew import run_opportunity_crew
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

class CrewInput(BaseModel):
    zipcode: str
    dealership: str
    customer: str
    person: str

@app.post("/run-competitor")
async def competitor(data: CrewInput):
    result = run_competitor_crew(**data.dict())
    return {"result": result}

@app.post("/run-opportunity")
async def opportunity(data: CrewInput):
    result = run_opportunity_crew(**data.dict())
    return {"result": result}
