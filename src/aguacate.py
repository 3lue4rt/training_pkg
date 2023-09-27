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
		self.Sub_Cam = rospy.Subscriber("/duckiebot/camera_node/image/raw", Image, self.procesar_img) #benjadoc: este el topico del usb_cam del compu
        #Publicar imagen(es)
		self.pub_img = rospy.Publisher("hola", Image, queue_size = 1) #benjadoc: aqui me invente cualquier topico
		self.pub_posc = rospy.Publisher("/duckiebot/posicion_pato", Point, queue_size = 1)


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

		image_out = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #benjadoc: poner el colorspace de BGR a HSV

		#Definir rangos para la mascara

		lower_limit = np.array([20, 100, 0])
		upper_limit =np.array([40, 255 ,255]) #benjadoc: estos limites son para el amarillo, de ahi se pueden hacer mas para otros colores

		#Mascara
		mask = cv2.inRange(image_out, lower_limit, upper_limit)
		image_out = cv2.bitwise_and(image, image, mask=mask)

		# Operaciones morfologicas, normalmente se utiliza para "limpiar" la mascara
		kernel = np.ones((5 , 5), np.uint8)
		img_erode = cv2.erode(mask, kernel, iterations=1) #Erosion
		img_dilate = cv2.dilate(img_erode, kernel, iterations=1) #Dilatar 

		# Definir blobs
		_,contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #benjadoc: aqui habia que eliminar un _, que habia antes del contours
		for cnt in contours:
			AREA = cv2.contourArea(cnt)
			if AREA>150 and AREA<1000: #Filtrar por tramo de blobs #benjadoc: aumente el area, pq o si no se hacian demasiadas weas chicas
				x,y,w,h = cv2.boundingRect(cnt)
				dist=Point()
				dist.z=3*175/h
				self.pub_posc.publish(dist)
				cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,100),2) #benjadoc: el tercer parametro habia que poner los x+ e y+
				image = cv2.putText(image, str(round(3*175/h))+"cm", (int(round(x+w/2)),int(round(y+h/2))),cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 2, cv2.LINE_AA) #benjadoc: este es un metodo que encontre para poner texto en las imagenes
			else:
				None

		# Publicar imagen final
		msg = bridge.cv2_to_imgmsg(image, "bgr8")
		self.pub_img.publish(msg)

		#benjadoc: los pasos que ocupe para que funcionara en mi compu fue:
		#1.- hacer un workspace en el note con los tutoriales de ros (incluye sourcear los setup.bash de ros y del catkin)
		#2.- roscore en una ventana de terminal
		#3.- roslaunch  usb_cam usb_cam-test.launch en otra ventana aparte terminal (es para prender la camara,
		# mantenerla abierta por el resto del procedimiento, ademas desbloquea varios topicos)
		#4.- abrir otro terminal para trabajar con el training package y los nodos.

def main():
	rospy.init_node('test') #creacion y registro del nodo!

	obj = Template('args') # Crea un objeto del tipo Template, cuya definicion se encuentra arriba

	#objeto.publicar() #llama al metodo publicar del objeto obj de tipo Template

	rospy.spin() #funcion de ROS que evita que el programa termine -  se debe usar en  Subscribers


if __name__ =='__main__':
	main()
