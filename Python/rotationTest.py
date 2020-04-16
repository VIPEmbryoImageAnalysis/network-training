#Simply rotates a single image

from PIL import Image

img = Image.open("0001_mouse5201.png")
img = img.rotate(180)
img.save("rotated180-2.png")
img.show(img)

