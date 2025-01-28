import numpy as np
from PIL import Image
from io import BytesIO
import base64

def ndarray2base64(imageArray: np.ndarray, format: str = 'PNG', toRGB = True) -> str:
    """
    numpy.ndarrayの画像をbase64エンコードされた文字列に変換
    
    :param image_array: numpy.ndarrayの画像データ
    :param format: 画像のフォーマット（'PNG'、'JPEG'など）
    :return: Base64エンコードされた文字列
    """
    
    # BGRをRGBに変換 (OpenCVの画像などでBGR順の場合)
    if imageArray.shape[-1] == 3 and toRGB:  # チャンネルが3の場合のみ
        imageArray = imageArray[..., ::-1]  # 最後の次元を反転（BGR -> RGB）

    # Convert numpy.ndarray to PIL.Image
    image = Image.fromarray(imageArray)
    
    # Save the image to a BytesIO stream
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    
    # Encode the BytesIO stream to Base64
    base64String = base64.b64encode(buffer.read()).decode('utf-8')
    return base64String