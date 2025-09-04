from fastapi import FastAPI

# Create app instance
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, API is working!"}

# Example endpoint with input
@app.get("/greet/{name}")
def greet_user(name: str):
    return {"message": f"Hello {name}, welcome to the API!"}

# Example with query parameter
@app.get("/add")
def add_numbers(a: int, b: int):
    return {"result": a + b}
