from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from Utils.mongo_utils import MongoUtils
from Classes.user_data_handler import UserDataHandler

app = FastAPI()
handler = Mangum(app)

# Enable CORS (Cross-Origin Resource Sharing) for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_instance = MongoUtils()
mongo = mongo_instance.connect_to_database()



@app.post("/login", response_model=dict)
async def login_user(event: dict):
    user_handler = UserDataHandler(event)
    return await user_handler.authenticate_user(mongo)


@app.post("/logout", response_model=dict)
async def logout_user(event: dict):
    user_handler = UserDataHandler(event)
    return await user_handler.logout_user(mongo)


@app.post("/putUserData", response_model=dict)
async def put_user_data(event: dict):
    user_handler = UserDataHandler(event)
    return await user_handler.put_user_data(mongo)


@app.get("/getUserData", response_model=dict)
async def get_user_data(event: dict):
    try:
        user_handler = UserDataHandler(event)
        return await user_handler.get_user_data(mongo)
    except HTTPException as e:
        return {"statusCode": e.status_code, "error": str(e.detail)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8090)
