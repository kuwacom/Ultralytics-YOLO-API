import argparse
from fastapi import FastAPI
import uvicorn
import logging
import utils
from utils.configManager import config

from api.api import router
import utils.configManager

app = FastAPI()

# APIルートをFastAPIに登録
app.include_router(router)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultralytics YOLO API Server")
    parser.add_argument("--host", type=str, help="Host IP address")
    parser.add_argument("--port", type=int, help="Port number")
    parser.add_argument("--reload", type=bool, help="dev reload", default=False)
    args = parser.parse_args()

    # ログの設定
    logging.basicConfig(
        level=logging.DEBUG,  # ログレベル（DEBUGより重要なログを全て記録）
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # フォーマット
        handlers=[logging.StreamHandler()] # 出力先（コンソール）
    )
    logger = logging.getLogger(__name__)

    # 設定ファイルの読み込み
    utils.configManager.loadConfig()

    # 引数が指定されていなければ設定ファイルから読み込む
    host = args.host if args.host else config.get('server', 'host', fallback='127.0.0.1')
    port = args.port if args.port else config.getint('server', 'port', fallback=8000)

    # モデルを初期化
    logger.info('Initialize models...')
    try:
        utils.modelManager.initModels(config.items('models'))
    except Exception as e:
        logger.error('Failed to initialize models!')
        logger.error(str(e))
        exit()

    # Uvicornで起動
    uvicorn.run(app, host=host, port=port, reload=args.reload)