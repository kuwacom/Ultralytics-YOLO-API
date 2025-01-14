from fastapi import APIRouter
from .v1 import v1Router

# APIのルーターを作成
router = APIRouter()

# v1のエンドポイントを追加
router.include_router(v1Router, tags=["YOLO11 v1"])
