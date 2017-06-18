import matplotlib
from Tkinter import *
import tkFileDialog as filedialog
import threading
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
#import ImageChops
import math
from PIL import Image
from PIL import ImageChops
import operator
import random
import copy
from PIL import *
from PIL import ImageStat
from scipy.fftpack import dct

# thread para correr el algoritmo sin que se pegue la imagen


class MyThread (threading.Thread):

    def run(self):
        try:
            cruce_probabilidad = int(probabilidad_cruce.get())
            poblacion_tamano = int(tamano_poblacion.get())
            aptos_menos = int(menos_aptos.get())
            mutacion_porcentaje = int(porcentaje_mutacion.get())
            mutacion_probabilidad = int(probabilidad_mutacion.get())
            algoritmo = variable.get()
            path = filename.get()

            if(algoritmo == "Distancia Euclidiana"):
                a = genetico_euclidiano_RGB(
                    path, cruce_probabilidad, poblacion_tamano, aptos_menos, mutacion_porcentaje, mutacion_probabilidad)
                print(a)
                a.show()
            elif (algoritmo == "Propio"):
                a = genetico_euclidiano_RGB(
                    path, cruce_probabilidad, poblacion_tamano, aptos_menos, mutacion_porcentaje, mutacion_probabilidad)
                a.show()
            else:
                a = genetico_euclidiano_RGB(
                    path, cruce_probabilidad, poblacion_tamano, aptos_menos, mutacion_porcentaje, mutacion_probabilidad)
                a.show()
        except:
            toplevel = Toplevel()
            toplevel.geometry("100x100+0+0")
            toplevel.config(bg="#f03d3f")
            label1 = Label(toplevel, text="Datos erroneos",
                           height=0, width=100)
            label1.pack(fill=Y)


def loadImage(path):
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixels = image.load()
    x, y = image.size
    # img=np.ones((x,y,3),dtype=np.int16)
    """
    for i in range(0,x/100):
        for j in range(0,y/100):
            a=pixels[i,j]
            print a
            img[i,j,0]=a[0]
            img[i,j,1]=a[1]
            img[i,j,2]=a[2]
    print img
    """
    img = np.array(image)
    return img, x, y

# dice si un numero esta dentro de un rango con respecto a un delta del
# numero que se quiere que este


def isSimilar(num1, num2):
    delta = 40
    # print "numeros: ",num1,num2
    if(num2 > num1 - delta and num2 < num1 + delta):
        return True
    else:
        return False
# devuelve un numero del 0 al 3 de acuerdo en que tan similares son los
# colores del pixel de la invencion con el objetivo


def compareRGB(rgbGoal, rgbImage):
    res = 0
    R, G, B = 0, 1, 2
    # print rgbGoal,rgbImage
    r = isSimilar(rgbGoal[R], rgbImage[R])
    g = isSimilar(rgbGoal[G], rgbImage[G])
    b = isSimilar(rgbGoal[B], rgbImage[B])
    if(r):
        res += 1
    if(g):
        res += 1
    if(b):
        res += 1
    return res
# compara todos los pixeles, recibe el objetivo y la otra imagen


def compareImages(pathGoal, pathNot):
    temp = loadImage(pathGoal)
    imgGoal = temp[0]
    x = temp[1]
    y = temp[2]
    imgNo = loadImage(pathNot)[0]
    comparison = 0
    for i in range(0, x):
        for j in range(0, y):
            a = compareRGB(imgGoal[i, j], imgNo[i, j])
            comparison += a

    print(comparison, "/", x * 3 * y)


def getHistogram(img):
    image = Image.open(img)
    print(image.histogram())


def getDifference(img1, img2):
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    show = ImageChops.difference(image1, image2)
    show.show()

# funciones de ordenamiento


def dividir_lista(lista):
    if(lista == []):
        return False
    else:
        return dividir_lista_aux(lista[0], lista[1:], [], [])


def dividir_lista_aux(pivote, lista, menores, mayores):
    if(lista == []):
        return [menores, mayores]
    elif lista[0] <= pivote:
        return dividir_lista_aux(pivote, lista[1:], menores + [lista[0]], mayores)
    else:
        return dividir_lista_aux(pivote, lista[1:], menores, mayores + [lista[0]])


def quick_sort(lista):
    if(lista == []):
        return []
    else:
        pivote = lista[0]
        menores_mayores = dividir_lista(lista)
        return quick_sort(menores_mayores[0] + [pivote] + quick_sort(menores_mayores[1]))
# aList tiene los numeros,bList tiene las imagenes


def insertionsort(aList, bList):
    for i in range(1, len(aList)):
        tmp1 = aList[i]
        tmp2 = bList[i]
        k = i
        while k > 0 and tmp1 < aList[k - 1]:
            aList[k] = aList[k - 1]
            bList[k] = bList[k - 1]
            k -= 1
        aList[k] = tmp1
        bList[k] = tmp2
    return aList, bList
# fin de funciones de ordenamiento

# funciones de diferencia de hash


def AverageHash(imagen):
        # Convertir la imagen a grayscale
    imagen = imagen.convert("L")  # 8-bit grayscale

    # Cambiar tamano a 8x8
    imagen = imagen.resize((8, 8), Image.ANTIALIAS)

    # Calcular el promedio
    averageValue = ImageStat.Stat(imagen).mean[0]

    # 1 si el tono es >= al promedio, 0 si no
    averageHash = 0
    for row in range(8):
        for col in range(8):
            averageHash <<= 1
            averageHash |= 1 * (imagen.getpixel((col, row)) >= averageValue)

    return averageHash


def DifferenceHash(imagen):
    # Convertir la imagen a grayscale
    imagen = imagen.convert("L")  # 8-bit grayscale

    # Cambiar tamano a 8x8
    imagen = imagen.resize((8, 8), Image.ANTIALIAS)

    # 1 si el pixel es igual o mas brillante, 0 si no
    # Empieza en el pixel 64
    previousPixel = imagen.getpixel((0, 7))

    differenceHash = 0
    for row in range(0, 8, 2):

        # izq a der en filas impares.
        for col in range(8):
            differenceHash <<= 1
            pixel = imagen.getpixel((col, row))
            differenceHash |= 1 * (pixel >= previousPixel)
            previousPixel = pixel

        row += 1

        # der a izq en filas pares
        for col in range(7, -1, -1):
            differenceHash <<= 1
            pixel = imagen.getpixel((col, row))
            differenceHash |= 1 * (pixel >= previousPixel)
            previousPixel = pixel

    return differenceHash


def HashDiff(tipo, hashmeta, imagen2):
    if tipo == 0:
        hash2 = DifferenceHash(imagen2)
        #print ("\n" + '%(hash)016x' %{"hash": hash1})
        #print ('%(hash)016x' %{"hash": hash2} + "\n")
        # XOR hash1 y hash2 and para contar las diferencias
        #return 100 - (((64 - bin(goalhash ^ hash2).count("1")) * 100.0) / 64.0)
    elif tipo == 1:
        hash2 = AverageHash(imagen2)
        #print ("\n" + '%(hash)016x' %{"hash": hash1})
        #print ('%(hash)016x' %{"hash": hash2} + "\n")
        # XOR hash1 y hash2 and para contar las diferencias
        #return 100 - (((64 - bin(goalhash ^ hash2).count("1")) * 100.0) / 64.0)
    else:
        hash2 = PerceptualHash(imagen2)
        #print ("\n" + '%(hash)016x' %{"hash": hash1})
        #print ('%(hash)016x' %{"hash": hash2} + "\n")
        # XOR hash1 y hash2 and para contar las diferencias
        return 100 - (((64 - bin(hashmeta ^ hash2).count("1")) * 100.0) / 64.0)

# retorna un array con num imagenes generadas aleatoriamente en color


def generateImages(num, x, y):
    result = []

    for n in range(num):
        a = np.random.rand(x, y, 3) * 255
        im = Image.fromarray(a.astype('uint8')).convert('RGB')
        result.append(im)
    return result


# genera imagenes en escala de grises
# no es usada en ninguna parte
"""
def generateImages_gris(num,x,y):
    result=[]

    for n in range(num):
        a = np.random.rand(x,y)
        im = Image.fromarray(a, 'L')
        result.append(im)
    return result
"""

# funcion para mutar imagenes en escala de grises
# no es usada en ninguna parte
"""
def mutar_gris(image):
    img1=np.array(image)
    x,y=image.size
    for i in range(x):
        for j in range(y):
            img1[i,j]=abs(img1[i,j]-256)
    image=Image.fromarray(img1.astype('uint8')).convert('L')
    return image
"""

# algoritmos de cruce que no se usan
"""

#algoritmos de cruce
def cruzar_intercambio(im1,im2,divs):
    x,y=im1.size
    x1=0
    y1=0
    x2=0
    y2=0
    cuando=True     #se intercambian los cuadrantes cuando es true
    for i in range(divs):
        #cuando=not cuando
        x1=0
        x2=x/divs
        y1=y2
        y2=y1+y/divs
        for j in range(divs):
            if(cuando):
                temp1=im1.crop((x1,y1,x2,y2))
                temp2=im2.crop((x1,y1,x2,y2))
                temp2=np.array(temp2)
                offset=(x1,y1)
                im2.paste(temp1,offset)
                temp2=Image.fromarray(temp2.astype('uint8')).convert('RGB')
                im1.paste(temp2,offset)


            x1=x2
            x2=x1+x/divs
            cuando=not cuando
    return im1,im2


def cruzar_random(im1,im2,probabilidad):
    x,y=im1.size
    x1=0
    x2=0
    for i in range(0,y-1,2):
        cruzar=random.randint(0,99)
        if(cruzar<probabilidad):
            cuando=True
        else:
            cuando=False
        if(cuando):
            x1=random.randint(0,y-3)
            x2=random.randint(x1+1,y-1)
            temp1=im1.crop((x1,i,x2,i+1))
            temp2=im2.crop((x1,i,x2,i+1))
            temp2=np.array(temp2)
            offset=(x1,i)
            im2.paste(temp1,offset)
            temp2=Image.fromarray(temp2.astype('uint8')).convert('RGB')
            im1.paste(temp2,offset)
    return im1,im2

#se cruzan mas bits que en la anterior
def cruzar_random2(im1,im2,probabilidad):
    ran=[]
    x,y=im1.size
    im3 = Image.new("L", (x, y))
    im4 = Image.new("L", (x, y))
    x1=0
    x2=0
    for i in range(0,x-1,2):
        x1=random.randint(0,x//10)
        x2=random.randint(2*x//10,x-1)
        ran.append((x1,x2))
        temp1=im1.crop((x1,i,x2,i+1))
        temp2=im2.crop((x1,i,x2,i+1))
        temp22=np.array(temp2)
        offset=(x1,i)
        im2.paste(temp1,offset)
        temp2=Image.fromarray(temp22.astype('uint8')).convert('L')
        im1.paste(temp2,offset)
        im3.paste(im1,(0,0))
        im4.paste(im2,(0,0))
    # print ran
    return im3,im4

def cruzar_random3(im1,im2,probabilidad):
    ran=[]
    x,y=im1.size
    im3 = Image.new("L", (x, y))
    im4 = Image.new("L", (x, y))
    x1=0
    x2=0
    for i in range(0,y-1,2):
        for j in range(0,x-5):
            x1=random.randint(j,x-5)
            x2=random.randint(x1+1,x)
            j=x2
            ran.append((x1,x2))
            temp1=im1.crop((x1,i,x2,i+1))
            temp2=im2.crop((x1,i,x2,i+1))
            temp22=np.array(temp2)
            offset=(x1,i)
            im2.paste(temp1,offset)
            temp2=Image.fromarray(temp22.astype('uint8')).convert('L')
            im1.paste(temp2,offset)
            im3.paste(im1,(0,0))
            im4.paste(im2,(0,0))
    # print ran
    return im3,im4
"""

# funcion que genera una imagen esqueleto de y*10x para representar el
# resultado del algoritmo genetico


def generate_result_image(x, y):
    a = np.empty([y, (x * 10) + 9, 3], dtype=int)
    a.fill(0)
    im = Image.fromarray(a.astype('uint8')).convert('RGB')
    return im


# diferencia euclidiana sobre el arreglo que es la imagen en RGB
def euclidiana(im1, im2):
    dif = 0
    img1 = np.array(im1)
    img2 = np.array(im2)
    x, y, z = img1.shape
    for i in range(x):
        for j in range(y):
            for k in range(3):
                dif += ((int(img1[i, j, k]) - int(img2[i, j, k]))**2)

    dif = math.sqrt(dif)
    return dif

# funcion de diferencia que usa el mse, es similar a la euclidiana
# no se usa, pero puede reemplazar a la funcion externa si no se encuentra
# una buena


def mse(im1, im2):
    dif = 0
    img1 = np.array(im1)
    img2 = np.array(im2)
    x, y, z = img1.shape
    for i in range(x):
        for j in range(y):
            for k in range(3):
                dif += ((int(img1[i, j, k]) - int(img2[i, j, k]))**2)

    dif = dif / (x * y)
    return dif

# funcion de diferencia propia que se usa en genetico_propio_RGB


def propia2(im1, im2):
    dif = 0
    img1 = np.array(im1)
    img2 = np.array(im2)
    x, y, z = img1.shape
    dif_rtemp = 0
    dif_gtemp = 0
    dif_btemp = 0
    dif_r = 0
    dif_g = 0
    dif_b = 0
    cont = 0
    for i in range(0, x - x // 10, x // 10):
        for j in range(0, y - y // 10, y // 10):
            cont += 1
            dif_rtemp = 0
            dif_gtemp = 0
            dif_btemp = 0
            cont_temp = 0
            for a in range(x // 10):
                for b in range(y // 10):
                    cont_temp += 1
                    dif_rtemp += (int(img1[i + a, j + b, 0]) -
                                  int(img2[i + a, j + b, 0]))**2
                    dif_gtemp += (int(img1[i + a, j + b, 1]) -
                                  int(img2[i + a, j + b, 1]))**2
                    dif_btemp += (int(img1[i + a, j + b, 2]) -
                                  int(img2[i + a, j + b, 2]))**2
            dif_r += (dif_rtemp / cont_temp)
            dif_g += (dif_gtemp / cont_temp)
            dif_b += (dif_btemp / cont_temp)
    ret = (dif_r**2 + dif_g**2 + dif_b**2) / cont
    return ret

    dif = dif / (x * y)
    return dif


# diferencia euclidiana para imagenes en escala de grises
# no se usa
"""
def euclidiana_gris(im1,im2):
    dif=0
    img1=np.array(im1)
    img2=np.array(im2)
    x,y=im1.size
    for i in range(x):
        for j in range(y):
            dif+=((int(img1[i,j])-int(img2[i,j]))**2)
    dif=math.sqrt(dif)
    return dif
"""
# funciones de prueba de integridad estructural que no funcionan muy bien
"""
def ssim(im1,im2):
    res=0
    cont=0
    x,y=im1.size
    for i in range(0,x,10):
        for j in range(0,y,10):
            cont+=1
            a=im1.crop((i,j,i+9,j+9))
            b=im2.crop((i,j,i+9,j+9))
            res+=ssim2(a,b)
    return res/cont
def ssim2(im1,im2):
    x,y=im1.size
    img1=np.array(im1)
    img2=np.array(im2)
    k1=0.01
    k2=0.03
    #promedio
    u1=np.sum(img1)//(3*x*y)
    u2=np.sum(img2)//(3*x*y)
    img11=np.empty([x,y])
    img22=np.empty([x,y])
    for i in range(x):
        for j in range(y):
            img11[i,j]=(int(img1[i,j,0])+int(img1[i,j,1])+int(img1[i,j,2]))/3
            img22[i,j]=(int(img2[i,j,0])+int(img2[i,j,1])+int(img2[i,j,2]))/3
    #varianza
    o1=np.sum(np.square(img11-u1))//(x*y)-1
    o2=np.sum(np.square(img22-u2))//(x*y)-1
    l=255
    #covarianza
    o3=(np.sum(img11-o1)*np.sum(img22-o2))/(x*y)
    c1=(k1*l)**2
    c2=(k2*l)**2
    a=((2*u1*u2+c1)*(2*o3+c2))
    b=((u1**2+u2**2+c1)*(o1**2+o2**2+c2))
    si=a/b
    return si
"""

# cruza dos imagenes a color


def cruzar_color(im1, im2):
    x, y = im1.size
    x1 = 0
    x2 = 0
    im3 = Image.new("RGB", (x, y))
    im4 = Image.new("RGB", (x, y))
    # para no modificar directamente las imagenes, sino crear una copia
    im3.paste(im1, (0, 0))
    im4.paste(im2, (0, 0))
    for i in range(0, y - 1, 2):
        x1 = random.randint(0, y - 3)
        x2 = random.randint(x1 + 1, y - 1)
        temp1 = im3.crop((x1, i, x2, i + 1))
        temp2 = im4.crop((x1, i, x2, i + 1))
        temp2 = np.array(temp2)
        offset = (x1, i)
        im4.paste(temp1, offset)
        temp2 = Image.fromarray(temp2.astype('uint8')).convert('RGB')
        im3.paste(temp2, offset)
    return im3, im4
# muta con el porcentaje de genes a mutar


def mutar_porcentaje(img, porcentaje):
    im = np.array(img)
    x, y = img.size
    num_pixels = x * y
    # pixeles que se van a mutar
    pixels_mutar = (porcentaje * num_pixels) // 100
    for i in range(pixels_mutar):
        px = random.randint(0, x - 1)
        py = random.randint(0, y - 1)
        im[px, py, 0] = abs(im[px, py, 0] - 256)
        im[px, py, 1] = abs(im[px, py, 1] - 256)
        im[px, py, 2] = abs(im[px, py, 2] - 256)
    image = Image.fromarray(im.astype('uint8')).convert('RGB')
    return image


# cruce RGB con el algoritmo euclidiano
def genetico_euclidiano_RGB(input, probabilidad_cruce, tamano_poblacion, menos_aptos, porcentaje_mutacion, probabilidad_mutacion):
    puntos_grafo = []  # donde se guardan los valores para el grafo
    last_10 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    imagenes_resultado = []
    meta = Image.open(input)
    meta=meta.resize((32, 32), Image.ANTIALIAS)
    tamano_x, tamano_y = meta.size
    resultado = generate_result_image(tamano_x, tamano_y)
    menos_aptos = (menos_aptos // 10) * 10
    finished = False
    # generateImages_gris(tamano_poblacion)
    poblacion2 = generateImages(tamano_poblacion, tamano_x, tamano_y)
    poblacion2values = []
    # el numero de malos que se deben mantener
    cuenta2 = menos_aptos * tamano_poblacion // 100
    # el numero de buenos que tendra la nueva poblacion
    cuenta1 = tamano_poblacion - cuenta2
    while not finished:
        poblacion1 = list(poblacion2)
        poblacion2values = []
        poblacion2 = []
        j = 0

        cont = 0
        # poblacion2.append(poblacion1[0])
        for i in range(0, tamano_poblacion // 2):
            new2 = Image.new("RGB", (tamano_x, tamano_y))
            new1 = Image.new("RGB", (tamano_x, tamano_y))
            cont += 2
            ran = random.randint(0, 100)  # si se va a cruzar
            if(i == 0):
                j = 1
            elif (i == tamano_poblacion // 2):
                j = tamano_poblacion + 1
            else:
                j = i * 2
            i = random.randint(0, tamano_poblacion - 1)
            j = random.randint(0, tamano_poblacion // 3)
            if(ran < probabilidad_cruce):
                # j=random.randint(0,tamano_poblacion//10)
                new1, new2 = cruzar_color(poblacion1[i], poblacion1[j])
            else:
                # j=random.randint(0,tamano_poblacion//10)
                new2.paste(poblacion1[i], (0, 0))
                new1.paste(poblacion1[j], (0, 0))

            ran = random.randint(0, 100)  # si se va a mutar
            if(ran < probabilidad_mutacion):
                new2 = mutar_porcentaje(new2, porcentaje_mutacion)
            ran = random.randint(0, 100)  # si se va a mutar
            if(ran < probabilidad_mutacion):
                new1 = mutar_porcentaje(new1, porcentaje_mutacion)
            poblacion2.append(new2)
            poblacion2.append(new1)
            # poblacion2values.append(euclidiana(meta,new1))
            if(cont >= cuenta1):  # el numerode buenos individuos que se cruzaran
                break

        for i in range(cuenta2):  # coger los mas malos
            new2 = Image.new("RGB", (tamano_x, tamano_y))
            new2.paste(poblacion1[tamano_poblacion - 1 - i], (0, 0))
            poblacion2.append(new2)

        for i in range(tamano_poblacion):
            # poblacion2values.append(euclidiana(meta,poblacion2[i]))
            poblacion2values.append(euclidiana(meta, poblacion2[i]))
        poblacion2values, poblacion2 = insertionsort(
            poblacion2values, poblacion2)
        last_10.append(poblacion2values[0])
        last_10 = last_10[1:]
        last = last_10[0]
        finished = True
        for i in last_10:
            if(abs(i - last) > 0.4):
                finished = finished and False
                break
            else:
                last = i
                finished = finished and True

        puntos_grafo.append(poblacion2values[0])  # se pone el mas parecido
        imagenes_resultado.append(poblacion2[0])

        print(int(poblacion2values[0]))
    plot(puntos_grafo, "euclidiano")
    j = 0
    for i in range(0, len(imagenes_resultado), len(imagenes_resultado) // 10):
        offset = ((tamano_x + 1) * j, 0)
        resultado.paste(imagenes_resultado[i], offset)
        j += 1
    # resultado.show()
    resultado.save("imageneuclidiana.png")
    return resultado


# funcion que usa el algoritmo genetico propio
def genetico_propio_RGB(input, probabilidad_cruce, tamano_poblacion, menos_aptos, porcentaje_mutacion, probabilidad_mutacion):
    puntos_grafo = []  # donde se guardan los valores para el grafo
    last_10 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    imagenes_resultado = []
    meta = Image.open(input)
    meta=meta.resize((32, 32), Image.ANTIALIAS)
    tamano_x, tamano_y = meta.size
    resultado = generate_result_image(tamano_x, tamano_y)
    menos_aptos = (menos_aptos // 10) * 10
    finished = False
    # generateImages_gris(tamano_poblacion)
    poblacion2 = generateImages(tamano_poblacion, tamano_x, tamano_y)
    poblacion2values = []
    # el numero de malos que se deben mantener
    cuenta2 = menos_aptos * tamano_poblacion // 100
    # el numero de buenos que tendra la nueva poblacion
    cuenta1 = tamano_poblacion - cuenta2
    while not finished:
        poblacion1 = list(poblacion2)
        poblacion2values = []
        poblacion2 = []
        j = 0

        cont = 0
        # poblacion2.append(poblacion1[0])
        for i in range(0, tamano_poblacion // 2):
            new2 = Image.new("RGB", (tamano_x, tamano_y))
            new1 = Image.new("RGB", (tamano_x, tamano_y))
            cont += 2
            ran = random.randint(0, 100)  # si se va a cruzar
            if(i == 0):
                j = 1
            elif (i == tamano_poblacion // 2):
                j = tamano_poblacion + 1
            else:
                j = i * 2
            i = random.randint(0, tamano_poblacion - 1)
            j = random.randint(0, tamano_poblacion // 3)
            if(ran < probabilidad_cruce):
                # j=random.randint(0,tamano_poblacion//10)
                new1, new2 = cruzar_color(poblacion1[i], poblacion1[j])
            else:
                # j=random.randint(0,tamano_poblacion//10)
                new2.paste(poblacion1[i], (0, 0))
                new1.paste(poblacion1[j], (0, 0))

            ran = random.randint(0, 100)  # si se va a mutar
            if(ran < probabilidad_mutacion):
                new2 = mutar_porcentaje(new2, porcentaje_mutacion)
            ran = random.randint(0, 100)  # si se va a mutar
            if(ran < probabilidad_mutacion):
                new1 = mutar_porcentaje(new1, porcentaje_mutacion)
            poblacion2.append(new2)
            poblacion2.append(new1)
            # poblacion2values.append(euclidiana(meta,new1))
            if(cont >= cuenta1):  # el numerode buenos individuos que se cruzaran
                break

        for i in range(cuenta2):  # coger los mas malos
            new2 = Image.new("RGB", (tamano_x, tamano_y))
            new2.paste(poblacion1[tamano_poblacion - 1 - i], (0, 0))
            poblacion2.append(new2)

        for i in range(tamano_poblacion):
            # poblacion2values.append(euclidiana(meta,poblacion2[i]))
            poblacion2values.append(propia2(meta, poblacion2[i]))
        poblacion2values, poblacion2 = insertionsort(
            poblacion2values, poblacion2)
        last_10.append(poblacion2values[0])
        last_10 = last_10[1:]
        last = last_10[0]
        finished = True
        for i in last_10:
            if(abs(i - last) > 0.4):
                finished = finished and False
                break
            else:
                last = i
                finished = finished and True

        puntos_grafo.append(poblacion2values[0])  # se pone el mas parecido
        imagenes_resultado.append(poblacion2[0])

        print(int(poblacion2values[0]))
    plot(puntos_grafo, "propio")
    j = 0
    for i in range(0, len(imagenes_resultado), len(imagenes_resultado) // 10):
        offset = ((tamano_x + 1) * j, 0)
        resultado.paste(imagenes_resultado[i], offset)
        j += 1
    resultado.show()
    resultado.save("imagenpropia.png")
    return resultado


# funcion que usa el algoritmo de diferencia externo
# para probar un algoritmo de diferencia solo hay que poner el algoritmo en la linea 789
# donde dice HashDiff
def genetico_externo_RGB(input, probabilidad_cruce, tamano_poblacion, menos_aptos, porcentaje_mutacion, probabilidad_mutacion):
    puntos_grafo = []  # donde se guardan los valores para el grafo
    last_10 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    imagenes_resultado = []
    meta = Image.open(input)
    meta=meta.resize((32, 32), Image.ANTIALIAS)
    tamano_x, tamano_y = meta.size
    resultado = generate_result_image(tamano_x, tamano_y)
    menos_aptos = (menos_aptos // 10) * 10
    finished = False
    # generateImages_gris(tamano_poblacion)
    poblacion2 = generateImages(tamano_poblacion, tamano_x, tamano_y)
    poblacion2values = []
    # el numero de malos que se deben mantener
    cuenta2 = menos_aptos * tamano_poblacion // 100
    # el numero de buenos que tendra la nueva poblacion
    hashmeta=PerceptualHash(meta)
    print (hashmeta)
    cuenta1 = tamano_poblacion - cuenta2
    while not finished:
        poblacion1 = list(poblacion2)
        poblacion2values = []
        poblacion2 = []
        j = 0

        cont = 0
        # poblacion2.append(poblacion1[0])
        for i in range(0, tamano_poblacion // 2):
            new2 = Image.new("RGB", (tamano_x, tamano_y))
            new1 = Image.new("RGB", (tamano_x, tamano_y))
            cont += 2
            ran = random.randint(0, 100)  # si se va a cruzar
            if(i == 0):
                j = 1
            elif (i == tamano_poblacion // 2):
                j = tamano_poblacion + 1
            else:
                j = i * 2
            i = random.randint(0, tamano_poblacion - 1)
            j = random.randint(0, tamano_poblacion // 3)
            if(ran < probabilidad_cruce):
                # j=random.randint(0,tamano_poblacion//10)
                new1, new2 = cruzar_color(poblacion1[i], poblacion1[j])
            else:
                # j=random.randint(0,tamano_poblacion//10)
                new2.paste(poblacion1[i], (0, 0))
                new1.paste(poblacion1[j], (0, 0))

            ran = random.randint(0, 100)  # si se va a mutar
            if(ran < probabilidad_mutacion):
                new2 = mutar_porcentaje(new2, porcentaje_mutacion)
            ran = random.randint(0, 100)  # si se va a mutar
            if(ran < probabilidad_mutacion):
                new1 = mutar_porcentaje(new1, porcentaje_mutacion)
            poblacion2.append(new2)
            poblacion2.append(new1)
            # poblacion2values.append(euclidiana(meta,new1))
            if(cont >= cuenta1):  # el numerode buenos individuos que se cruzaran
                break

        for i in range(cuenta2):  # coger los mas malos
            new2 = Image.new("RGB", (tamano_x, tamano_y))
            new2.paste(poblacion1[tamano_poblacion - 1 - i], (0, 0))
            poblacion2.append(new2)

        for i in range(tamano_poblacion):
            # poblacion2values.append(euclidiana(meta,poblacion2[i]))
            poblacion2values.append(HashDiff(2, hashmeta, poblacion2[i]))
        poblacion2values, poblacion2 = insertionsort(
            poblacion2values, poblacion2)
        last_10.append(poblacion2values[0])
        last_10 = last_10[1:]
        last = last_10[0]
        finished = True
        for i in last_10:
            if(abs(i - last) > 0.4):
                finished = finished and False
                break
            else:
                last = i
                finished = finished and True

        puntos_grafo.append(poblacion2values[0])  # se pone el mas parecido
        imagenes_resultado.append(poblacion2[0])
        had = poblacion2[0]
        had.save("mejor.png")

        print(int(poblacion2values[0]))
    plot(puntos_grafo, "externo")
    j = 0
    for i in range(0, len(imagenes_resultado), len(imagenes_resultado) // 10):
        offset = ((tamano_x + 1) * j, 0)
        resultado.paste(imagenes_resultado[i], offset)
        j += 1
    # resultado.show()
    resultado.save("imagenmse.png")
    return resultado

# para hacer un grafo de los valores


def plot(values, tipo):
    x = list(range(len(values)))
    plt.plot(x, values, ':rs')
    plt.axis([0, len(values), 0, values[0]])
    plt.xlabel("Generaciones")
    plt.ylabel("Similitud")
    plt.savefig("grafo" + tipo + ".png")
    # plt.show()


def PerceptualHash(imagen):

    im=imagen.convert("L") # 8-bit grayscale
    im = im.resize((32,32), Image.ANTIALIAS)
    vector=np.array(im)
    dctResul=dct(dct(dct(vector, axis=0), axis=1))
    resul=np.empty((8,8))
    for i in range(8):
        for j in range(8):
           	resul[i][j] = dctResul[i][j]
    resul = resul.flatten()
    averageValue = np.median(resul)
    perceptualHash=0
    for i in range(64):
        perceptualHash <<= 1
        perceptualHash |= 1 * (resul[i] <= averageValue)
    return perceptualHash


def dct2(vector):
    n = len(vector)
    c = np.zeros(n)
    rows = int(np.sqrt(n))

    for i in range(0, n):
        for j in range(0, n):
            c[i] = c[i] + vector[j] * np.cos((np.pi / n) * (j + 0.5) * i)

        c[i] = c[i] * np.sqrt(2.0 / n)

    c = np.reshape(c, (rows, rows))
    return c
# funcion que hace las cosas con un thread para que la interfaz no se
# quede pegada


def go():
    try:
        cruce_probabilidad = int(probabilidad_cruce.get())
        poblacion_tamano = int(tamano_poblacion.get())
        aptos_menos = int(menos_aptos.get())
        mutacion_porcentaje = int(porcentaje_mutacion.get())
        mutacion_probabilidad = int(probabilidad_mutacion.get())
        algoritmo = variable.get()
        path = filename.get()
        if(algoritmo == "Distancia Euclideana"):
            a = (
                path, cruce_probabilidad, poblacion_tamano, aptos_menos, mutacion_porcentaje, mutacion_probabilidad)
            thread=Thread(target = genetico_euclidiano_RGB, args = a)
            thread.start()
        elif (algoritmo == "Propio"):
            a = (
                path, cruce_probabilidad, poblacion_tamano, aptos_menos, mutacion_porcentaje, mutacion_probabilidad)
            thread=Thread(target = genetico_propio_RGB, args = a)
            thread.start()
        else:
            a = (
                path, cruce_probabilidad, poblacion_tamano, aptos_menos, mutacion_porcentaje, mutacion_probabilidad)
            thread=Thread(target = genetico_externo_RGB, args = a)
            thread.start()
    except:
        toplevel = Toplevel()
        toplevel.geometry("100x100+0+0")
        toplevel.config(bg="#f03d3f")
        label1 = Label(toplevel, text="Datos erroneos",
                       height=0, width=100)
        label1.pack(fill=Y)



# si se quiere usar la interfaz hay que descomentartodo lo siguiente
# si se quiere llamar a la funcion directamenta hay que bajar hasta
# despues de este comentario y poner los parametros



#interfaz
principal=Tk()
principal.config(bg="#f03d3f")
principal.title("Generador de imagenes")
principal.resizable(0,0)#no permite modificar el tamano de la ventana
principal.geometry("250x185+0+0")
#probabilidad de cruce
frame1 = Frame(principal)
frame1.pack(fill=X)
wid=30
lbl1 = Label(frame1, text="Probabilidad de cruce", width=wid)
lbl1.pack(side=LEFT, padx=0, pady=0,fill=X)

probabilidad_cruce = Entry(frame1)
probabilidad_cruce.pack( side=RIGHT,padx=0, expand=False)

#tamano poblacion
frame2 = Frame(principal)
frame2.pack(fill=X)

lbl2 = Label(frame2, text="Tamano de la poblacion", width=wid)
lbl2.pack(side=LEFT, padx=0, pady=0,fill=X)

tamano_poblacion = Entry(frame2)
tamano_poblacion.pack( side=RIGHT,padx=0, expand=False)

#Menos aptos
frame3 = Frame(principal)
frame3.pack(fill=X)

lbl3 = Label(frame3, text="Porcentaje de menos aptos", width=wid)
lbl3.pack(side=LEFT, padx=0, pady=0,fill=X)

menos_aptos = Entry(frame3)
menos_aptos.pack( side=RIGHT,padx=0, expand=False)

#Porcentaje mutaacion
frame4 = Frame(principal)
frame4.pack(fill=X)

lbl4 = Label(frame4, text="Porcentaje de genes a mutar", width=wid)
lbl4.pack(side=LEFT, padx=0, pady=0,fill=X)

porcentaje_mutacion = Entry(frame4)
porcentaje_mutacion.pack( side=RIGHT,padx=0, expand=False)

#probabilidad mutacion
frame5 = Frame(principal)
frame5.pack(fill=X)

lbl5 = Label(frame5, text="Tamano de la poblacion", width=wid)
lbl5.pack(side=LEFT, padx=0, pady=0,fill=X)

probabilidad_mutacion = Entry(frame5)
probabilidad_mutacion.pack( side=RIGHT,padx=0, expand=False)

#algoritmo
frame6=Frame(principal)
frame6.pack(fill=X)

lbl6 = Label(frame6, text="Algoritmo", width=10)
lbl6.pack(side=LEFT, padx=0, pady=0,fill=X)

variable = StringVar(frame6)
variable.set("Distancia Euclideana") # default value
w = OptionMenu(frame6, variable, "Distancia Euclidiana", "Propio", "Externo")
w.pack(side=RIGHT,padx=0, expand=False)

probabilidad_mutacion.insert(0, "2")
tamano_poblacion.insert(0,"100")
menos_aptos.insert(0,"5")
porcentaje_mutacion.insert(0,"25")
probabilidad_cruce.insert(0,"95")

#imagen
#obtiene el nombre del archivo a convertir
def getfile():
    file_path = filedialog.askopenfilename()
    filename.set(file_path)
frame7=Frame(principal)
frame7.pack(fill=X)

l5 = Label(frame7,text = "Image")
l5.pack(side=LEFT, padx=0, pady=0,fill=X)
filename = StringVar(frame7)
filename.set("Select File")
button3 = Button(frame7,textvariable = filename,command=getfile)
button3.pack()



frame8=Frame(principal)
frame8.pack(fill=X)
button4 = Button(frame8,text = "Start",command=go)
button4.pack()


principal.mainloop()


#fin de interfaz
"""


# si se quiere usar la interfaz hay que comentar lo siguiente y
# descomentar la interfaz

probabilidad_cruce = 95
tamano_poblacion = 200
menos_aptos = 5
porcentaje_mutacion = 25
probabilidad_mutacion = 2

#a = genetico_externo_RGB("325.png", probabilidad_cruce, tamano_poblacion,
#                         menos_aptos, porcentaje_mutacion, probabilidad_mutacion)

a = genetico_euclidiano_RGB("325.png", probabilidad_cruce, tamano_poblacion,
                         menos_aptos, porcentaje_mutacion, probabilidad_mutacion)
a.show()
"""
