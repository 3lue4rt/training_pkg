#!/usr/bin/env python

import rospy #importar ros para  python
from std_msgs.msg import String, Int32 # importar mensajes de ROS tipo String y tipo Int32
from geometry_msgs.msg import Twist, Point # importar mensajes de ROS tipo geometry / Twist
from sensor_msgs.msg import Image # importar mensajes de ROS tipo Image
import cv2 # importar libreria opencv
from cv_bridge import CvBridge # importar convertidor de formato de imagenes
import numpy as np # importar libreria numpy



class Template(object):
	def __init__(self, args):
		super(Template, self).__init__()
		self.args = args
		#Suscribrirse a la camara
		self.Sub_Cam = rospy.Subscriber("/duckiebot/camera_node/image/raw", Image, self.procesar_img)
        #Publicar imagen(es)
		self.pub_img = rospy.Publisher("/duckiebot/camera_node/image/cocaretornable", Image, queue_size = 1)
		#self.pub_img = rospy.Publisher("mas", Image, queue_size = 1)


	def procesar_img(self, msg):
		#Transformar Mensaje a Imagen
		bridge = CvBridge()
		image = bridge.imgmsg_to_cv2(msg, "bgr8")

		#Espacio de color

			#cv2.COLOR_RGB2HSV
			#cv2.COLOR_RGB2GRAY
			#cv2.COLOR_RGB2BGR
			#cv2.COLOR_BGR2HSV
			#cv2.COLOR_BGR2GRAY
			#cv2.COLOR_BGR2RGB

		image_out = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

		#Definir rangos para la mascara

		amarillo_feo_RGB=np.array([219,219,18])
		amarillo_feo_BGR=np.array([18,219,219])
		amarillo_feo_HSV=np.array([60,92,86])
		variacion=np.array([20,20,100])
		HSV_lower=np.array([10,100,100])
		HSV_upper=np.array([40,250,250])


		lower_limit =HSV_lower
		upper_limit =HSV_upper

		#Mascara
		mask = cv2.inRange(image_out, lower_limit, upper_limit)
		image_out = cv2.bitwise_and(image, image, mask=mask)

		# Operaciones morfologicas, normalmente se utiliza para "limpiar" la mascara
		kernel = np.ones((5 , 5), np.uint8)
		img_erode = cv2.erode(mask, kernel, iterations=1) #Erosion
		img_dilate = cv2.dilate(img_erode, kernel, iterations=1) #Dilatar 

		# Definir blobs
		_,contours, hierarchy = cv2.findContours(img_dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			AREA = cv2.contourArea(cnt)
			if AREA>10: #Filtrar por tramo de blobs
				x,y,w,h = cv2.boundingRect(cnt)
				cv2.rectangle(image, (x,y), (w,h), (104,58,5), 20)
			else:
				None

		alturapato=2,5
		

		# Publicar imagen final
		msg = bridge.cv2_to_imgmsg(image_out, "bgr8")
		self.pub_img.publish(msg)

def main():
	rospy.init_node('test') #creacion y registro del nodo!

	obj = Template('args') # Crea un objeto del tipo Template, cuya definicion se encuentra arriba

	#objeto.publicar() #llama al metodo publicar del objeto obj de tipo Template

	rospy.spin() #funcion de ROS que evita que el programa termine -  se debe usar en  Subscribers


if __name__ =='__main__':
	main()
