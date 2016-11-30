from scipy import misc
import random
import numpy
import pickle

def average(pixel):
    return (pixel[0] + pixel[1] + pixel[2]) / 3

path_dir = "/home/antonio/Documentos/SensoresInteligentes/Base/"
image = []

# abre a base de dados
for k in xrange(15):
    image_file = str(k) + ".jpg"
    image = misc.imread(path_dir + image_file, flatten=1)
    new_image_file = path_dir + 'test/' + image_file
    misc.imsave(new_image_file, image)
