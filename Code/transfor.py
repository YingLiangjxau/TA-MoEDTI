from PIL import Image
import os

def png_to_eps(input_path: str, output_path: str = None):
    """
    将 PNG 图片转换为 EPS 格式

    :param input_path:  输入 PNG 图片路径
    :param output_path: 输出 EPS 路径，若为 None 则在同目录生成同名 .eps
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"找不到文件: {input_path}")

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".eps"

    img = Image.open(input_path)

    # EPS 不支持透明度，需要先转成 RGB（白色背景）
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])  # 用 alpha 通道做遮罩
        img = background
    else:
        img = img.convert("RGB")

    img.save(output_path, format="EPS")
    print(f"转换成功: {input_path} -> {output_path}")
    return output_path


if __name__ == "__main__":
    # 在这里填写你的图片路径
    input_png = r"E:\WSPACE\NEW_DTI2\image\flow.png"

    png_to_eps(input_png)