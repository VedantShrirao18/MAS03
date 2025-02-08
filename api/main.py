import uvicorn
from fastapi import FastAPI
from ai_pipeline import query_agent


app = FastAPI()

@app.get("/query")
def query(query: str):
    return query_agent(query)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
