#!/usr/bin/env python3
import numpy as numpy
import matplotlib.pyplot as plot
import cv2
import EncoderLZW
import DecoderLZW
import sys, getopt

# Variables globales
subsampling_j = 4
subsampling_a = 2
subsampling_b = 0
recursion_level = 3
step = 4
dead_zone_size = 10

# Permet de convertir RGB en YUV
def rgb_to_yuv(r, g, b):
    y = numpy.zeros((len(r), len(r[0])))
    u = numpy.zeros((len(r), len(r[0])))
    v = numpy.zeros((len(r), len(r[0])))

    for i in range(len(r)) :
        for j in range(len(r[0])) :
            # Formules données dans l'énoncé
            y[i][j] = (((r[i][j]) + 2*(g[i][j]) + (b[i][j])) / 4)
            u[i][j] = (b[i][j]) - (g[i][j])
            v[i][j] = (r[i][j]) - (g[i][j])

    return y, u, v

# Permet de convertir YUV en RGB
def yuv_to_rgb(y, u, v):
    r = numpy.zeros((len(y), len(y[0])))
    g = numpy.zeros((len(y), len(y[0])))
    b = numpy.zeros((len(y), len(y[0])))

    for i in range(len(y)) :
        for j in range(len(y[0])) :
            # Formules données dans l'énoncé
            g[i][j] = y[i][j] - ((u[i][j] + v[i][j])/4)
            r[i][j] = v[i][j] + g[i][j]
            b[i][j] = u[i][j] + g[i][j]

    return r, g, b

# Permet de faire un sous-échantillonnage de ratio 4:2:0 
def subsampling(u, v):
    for line in range(0, len(u), 2) :
        onlyFirstLine = False
        if line + 1 >= len(u) :
            onlyFirstLine = True

        u_1 = u[line]
        v_1 = v[line]

        if not onlyFirstLine :
            u_2 = u[line + 1]
            v_2 = v[line + 1]

        for column in range(len(u[0])) :
            line1_sample = int(subsampling_j / subsampling_a)
            pos = int(column / line1_sample) * line1_sample
            u_1[column] = u_1[pos]
            v_1[column] = v_1[pos]

            if not onlyFirstLine :
                # Se fait seulement dans le cas où subsampling_b = 0 (ce qui est notre cas pour ce TP)
                u_2[column] = u_1[column]
                v_2[column] = v_1[column]

    return numpy.array(u), numpy.array(v)
   
# Permet d'appliquer DWT de Haar, cette fonction appelle le DWT sur chaque composante de yuv
def DWT(yuv, recursion):
    transformedYUV = []
    for component in yuv : 
        transformedComp = DWT_component(component, recursion)
        transformedYUV.append(numpy.array(transformedComp))
    return transformedYUV
    
# Permet d'appliquer DWT sur chaque composante (appel récursif de la fonction où n = recursion)
def DWT_component(component, recursion) : 
    # Si on est rendu au dernier niveau de récursion, on retourne la composante sans changement
    if recursion == 0 :
        return component 

    # Applique un filtre passe-bas
    component_l = (component[:,::2] + component[:,1::2])/2
    component_ll = (component_l[::2,:] + component_l[1::2,:])/2
    component_lh = (component_l[::2,:] - component_l[1::2,:])/2
    
    # Applique un filtre passe-haut
    component_h = (component[:,::2] - component[:,1::2])/2
    component_hl = (component_h[::2,:] + component_h[1::2,:])/2
    component_hh = (component_h[::2,:] - component_h[1::2,:])/2
    
    # Ceci se fait en récursion, donc la fonction est rappelée avec (recursion - 1)
    component_ll = DWT_component(component_ll, recursion - 1)
    
    # On calcule la somme qui sera retournée
    transformedComp = numpy.concatenate((numpy.concatenate((component_ll, component_hl)), numpy.concatenate((component_lh, component_hh))), axis = 1)
    
    return transformedComp
    
# Permet d'inverser la DWT de Haar, cette fonction appelle DWT sur chaque composante de yuv
def inverted_DWT(yuv, recursion) :
    transformedYUV = []
    for component in yuv : 
        transformedComp = inverted_DWT_component(component, recursion)
        transformedYUV.append(transformedComp)

    return transformedYUV

# Permet d'inverser la DWT de chaque composante (fait une appel récursif de la fonction où n = recursion)  
def inverted_DWT_component(component, recursion) :
    if (recursion == 0) :
        return component
        
    TL_corner = component[:len(component) >> 1, :len(component[0]) >> 1]

    # Inverse le filtre passe-bas, on appelle en récursion la fonction jusqu'à temps que recursion - 1 = 0
    component_ll = inverted_DWT_component(TL_corner, recursion - 1)
    component_lh = component[:len(component) >> 1, len(component[0]) >> 1:]
    component_l = numpy.zeros((len(component_ll) * 2, len(component_ll[0])))
    
    for i in range(len(component_ll)) :
        for j in range(len(component_ll[0])) :
            component_l[2 * i, j] = component_ll[i, j] + component_lh[i, j]
            component_l[2 * i + 1, j] = component_ll[i, j] - component_lh[i, j]

    # Inverse le filtre passe-haut
    component_hl = component[len(component) >> 1:, :len(component[0]) >> 1]
    component_hh = component[len(component) >> 1:, len(component[0]) >> 1:]
    component_h = numpy.zeros((len(component_hl) * 2, len(component_hl[0])))

    for i in range(len(component_hl)) :
        for j in range(len(component_hl[0])) :
                component_h[2 * i, j] = component_hl[i, j] + component_hh[i, j]
                component_h[2 * i + 1, j] = component_hl[i, j] - component_hh[i, j]

    # Calcul l'estimation de la valeur initiale de la composante initiale
    transformedComp = numpy.zeros((len(component), len(component[0])))

    for i in range(len(component_l)) :
        for j in range(len(component_l[0])) :
            transformedComp[i, j * 2] = component_l[i, j] + component_h[i, j]
            transformedComp[i, j * 2 + 1] = component_l[i, j] - component_h[i, j]

    return transformedComp

# Permet de faire la quantification de yuv
def quantify(yuv, step, deadZoneSize) :
    quantifiedYUV = []
    for component in yuv : 
        quantifiedComp = [[quantify_component(component[i, j], step, deadZoneSize) for j in range(len(component[0]))] for i in range(len(component))]
        quantifiedYUV.append(numpy.array(quantifiedComp))
        
    return quantifiedYUV   

# Calcule la valeur des composantes quantifiées. (0 si dans la zoneMorte)
def quantify_component(component, step, deadZoneSize):
    if (abs(int(component * 255)) < deadZoneSize):
        return 0
    else:
        return int(int(component * 255) / step) * step 
     
def main(inputfile):
    # On lit d'abord l'image comme float et on la sépare en composantes RGB
    image = (cv2.imread(inputfile)).astype(float)
    b, g, r = cv2.split(image)

    # S'assure que les valeurs flottantes de RGB se trouvent entre [0, 1]
    b, g, r = [component/255 for component in [b, g, r]]

    # Commencement du PIPELINE
    # On applique d'abord la conversion pour passer de RGB à YUV -> cette étape est sans perte
    y, u, v = rgb_to_yuv(r, g, b)

    # On fait par la suite un sous-échantillonnage de ratio 4:2:0  -> cette étape est avec perte d'information
    u, v = subsampling(u, v)

    # On applique la DWT de Haar avec un niveau de récursion = 3  -> cette étape est sans perte
    yuv = [y, u, v]
    transformedYUV = DWT(yuv, recursion_level)

    # On fait la quantification des yuv tranformés  -> cette étape est avec perte
    quantifiedYUV = quantify(transformedYUV, step, dead_zone_size)

    # On transforme ensuite la matrice 2D en 1D
    y = quantifiedYUV[0]
    columnLength = len(y[0])
    
    for component in quantifiedYUV: 
        component = component.ravel()

    # On prépare le message à encoder avec LZW en transformant le yuv en représentation binaire
    for component in quantifiedYUV: 
        component += 255

    y = quantifiedYUV[0]
    binaryYUVLength = len(''.join([str(bin(x))[2:].zfill(9) for x in y.ravel()]))
    
    binaryYUV = ''
    for component in quantifiedYUV: 
        componentBinString = ''.join([str(bin(x))[2:].zfill(9) for x in component.ravel()])  
        binaryYUV += componentBinString # Représentation finale de YUV en binaire

    # On utilise l'algorithme de compression LZW
    binaryYUVEncSize = EncoderLZW.compressLZW(binaryYUV, 15)

    # Permet de calculer et afficher le taux de compression en pourcentage 
    binaryYUVRepSize = len(binaryYUV)
    compressionRate = (1 - (float(binaryYUVEncSize) / binaryYUVRepSize)) * 100
    print("Le taux de compression : " + str(compressionRate) + "%")
    print("La taille initiale du fichier : " + str(binaryYUVRepSize))
    print("La taille après la compression : " + str(binaryYUVEncSize))

    # Par la suite on décompresse l'image en utilisant la méthode de décodage de LZW
    binaryYUV = DecoderLZW.decompressLZW(15)

    # On revient vers 1D contenant des entiers dans [-255, 255]. Ceci permet de trouver les composante y, u, v
    yBinString = binaryYUV[0 : binaryYUVLength]
    y = numpy.array([int(yBinString[i : i + 9], 2) for i in range(0, len(yBinString), 9)])
    y -= 255

    uBinString = binaryYUV[binaryYUVLength : 2 * binaryYUVLength]
    u = numpy.array([int(uBinString[i : i + 9], 2) for i in range(0, len(uBinString), 9)])
    u -= 255

    vBinString = binaryYUV[2 * binaryYUVLength:]
    v = numpy.array([int(vBinString[i : i + 9], 2) for i in range(0, len(vBinString), 9)])
    v -= 255

    # On transforme ensuite de 1D en 2D
    y = numpy.array([y[i : i + columnLength] for i in range(0, len(y), columnLength)])
    u = numpy.array([u[i : i + columnLength] for i in range(0, len(u), columnLength)])
    v = numpy.array([v[i : i + columnLength] for i in range(0, len(v), columnLength)])

    # Par la suite en remet les valeurs de yuv en float
    y, u, v = [x/255.0 for x in [y, u, v]]

    # On applique la transformée inverse de DWT de Haar avec le même niveau de récursion (soit 3 étages)
    YUV = [y, u, v]
    inversedYUV = inverted_DWT(YUV, recursion_level)

    # Finalement, on fait la conversion de YUV vers RGB
    r, g, b = yuv_to_rgb(inversedYUV[0], inversedYUV[1], inversedYUV[2])

    # En utilisant matplotlib, on affiche l'image résultante (celle qui a traversé le pipeline JPEG2000)
    rgb_image = cv2.merge([r, g, b])  # switch to rgb
    plot.imshow((rgb_image * 255).astype(numpy.uint8))
    plot.show() 
    
if __name__ == '__main__':
    # Permet de lire le fichier spécifié par l'utilisateur en input
    argv = sys.argv[1:]
    inputfile = ''
    try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
    except getopt.GetoptError:
      print ('test.py -i <inputfile>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
    print ('Input file is ', inputfile)

    # Appel du main, soit les étapes du pipeline
    main(inputfile)
