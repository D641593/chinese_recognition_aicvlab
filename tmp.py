from PIL import Image

img = Image.open("tmp.jpg")
img = img.rotate(90,expand=True)
img.show()