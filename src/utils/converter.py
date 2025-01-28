import numpy as np
from PIL import Image
from io import BytesIO
import base64

def ndarray2base64(imageArray: np.ndarray, format: str = 'PNG') -> str:
    """
    numpy.ndarrayの画像をbase64エンコードされた文字列に変換
    
    :param image_array: numpy.ndarrayの画像データ
    :param format: 画像のフォーマット（'PNG'、'JPEG'など）
    :return: Base64エンコードされた文字列
    """
    # Convert numpy.ndarray to PIL.Image
    image = Image.fromarray(imageArray)
    
    # Save the image to a BytesIO stream
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    
    # Encode the BytesIO stream to Base64
    base64String = base64.b64encode(buffer.read()).decode('utf-8')
    return base64String