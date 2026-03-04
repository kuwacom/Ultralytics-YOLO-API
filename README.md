# Ultralytics-YOLO-API
Ultralytics の YOLO v8 / v11 に対応した REST API

## 開発環境セットアップ (uv)

```bash
uv sync
```

開発依存込みで入れる場合:

```bash
uv sync --dev
```

## 起動

```bash
uv run python src/main.py --host 0.0.0.0 --port 8080
```
