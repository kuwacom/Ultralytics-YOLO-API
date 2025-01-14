from fastapi import APIRouter
from .detect import detectRouter
from .models import modelsRouter

# v1ルーターを作成
v1Router = APIRouter()
# プレフィックスを設定
prefix = '/v1'

v1Router.include_router(detectRouter, prefix=prefix)
v1Router.include_router(modelsRouter, prefix=prefix)