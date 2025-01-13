# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:50:30 2024

@author: Anusk
"""

import PIL
from PIL import Image
from IPython.display import display
from PIL import ImageFilter,ImageDraw, ImageFont
from PIL import ImageEnhance

file1="1.jpg"
image1=Image.open(file1)
file2="2.jpeg"
image2=Image.open(file2)
#1. Sa se incarce si sa se afiseze primele 2 imagini.

display(image1)
display(image2)

#2.Sa se salveze una dintre ele cu alta extensie fata de cea initiala.
image1.save("imagine1.png")

#3.Sa se roteasca o imagine in sensul acelor de ceasornic la (90-data nasterii fiecarui student) grade. 

image3=image2.rotate(13)
display(image3)
#4.Sa se aplice cate 2 filtre diferite celor 2 imagini si sa se afiseze imaginile filtrate.

filtru1=image1.filter(PIL.ImageFilter.BLUR)
filtru2=image1.filter(PIL.ImageFilter.SMOOTH_MORE)
filtru3=image2.filter(PIL.ImageFilter.EMBOSS)
filtru4=image2.filter(PIL.ImageFilter.SHARPEN)
display(filtru1)
display(filtru2)
display(filtru3)
display(filtru4)


#5 Sa se creeze o imagine noua ce contine alaturarea celor 2 imagini.
#redimensionarea imaginilor
image1_size = image1.size
image2_size = image2.size

if image2_size<image1_size:
    image1 = image1.resize((image2_size[0], image2_size[1]))
else:
    image2 = image2.resize((image1_size[0], image1_size[1]))
#crearea imaginii finale
new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
#adaugarea celor 2 imagini initiale
new_image.paste(image1,(0,0))
new_image.paste(image2,(image1_size[0],0))
#salvarea imaginii rezultat
new_image.save("merged_image.jpg","JPEG")
display(new_image)

#6 decupare element din imagine 
display(image2.crop((400,700,950,1300)))

#7 deseneaza dreptunghi
drawing_object=ImageDraw.Draw(image2)
drawing_object.rectangle((400,700,950,1300), fill = None, outline ='blue',width=10)
display(image2)

#8
draw1 = ImageDraw.Draw(image1)
draw2 = ImageDraw.Draw(image2)
text = "Branzea Ana-Maria"
font = ImageFont.truetype('arial.ttf', 13)

# adaugarea textului 
draw1.text((10,10), text, font=font,fill=(55,13,80))
draw2.text((10,10), text, font=font,fill=(55,13,80))
#Salvarea si afisarea imaginii 
image1.save('watermark1.jpg')
image2.save('watermark2.jpg')
display(image1)
display(image2)

#9 foaie de contact
# se construieste o lista cu 8 imagini
enhancer=ImageEnhance.Contrast(image1)
images=[]
for i in range(1, 9):
    images.append(enhancer.enhance(i/10))
    
first_image=images[0]
contact_sheet=PIL.Image.new(first_image.mode, (first_image.width*2,first_image.height*4))
x=0
y=0

# se parcurge lista cu cele 8 imagini 
for img in images:
    # se adauga imaginea curenta in foaia de contact
    contact_sheet.paste(img, (x, y) )
    # Se actualizeaza valoarea parametrilor x si y ce indica pozitia. 
    #Daca s-a atins latimea imaginii, se seteaza x la 0 si y la urmatoarea linie de inserat
    if x+first_image.width == contact_sheet.width:
        x=0
        y=y+first_image.height
    else:
        x=x+first_image.width

# se redimensioneaza foaia de contact si se afiseaza
contact_sheet = contact_sheet.resize((int(contact_sheet.width/2),int(contact_sheet.height/2) ))
display(contact_sheet)