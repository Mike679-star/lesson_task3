import PIL.Image as pil_image


image_path = "img/baby.png"
scale = 4

hr = pil_image.open(image_path).convert('RGB')
# 取放大倍数的倍数, width, height为可被scale整除的训练数据尺寸
hr_width = (hr.width // scale) * scale
hr_height = (hr.height // scale) * scale
# 图像大小调整,得到高分辨率图像Hr
hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
# 低分辨率图像缩小
lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
lr.save("img/baby_bicubic.png")
