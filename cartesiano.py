import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from mw import Ui_MainWindow
import numpy as np
import random
import prueba
import time

class AdalineThread(QThread):
    update_signal = pyqtSignal(list, list, list,bool,list,str,float,float)

    def __init__(self, parent=None):
        super(AdalineThread, self).__init__(parent)
        self.coordenadas = []
        self.salidas_deseadas = []
        self.limite_de_epocas = 0
        self.factor_de_aprendizaje = 0
        self.w1 = 0
        self.w2 = 0
        self.bias = 0
        self.bandera = True
        self.neuronas = 0
        
    def convertir_a_decimal(self,lista):
        numeros_decimales = []
        for fila in lista:
            numero_decimal = 0
            for i, bit in enumerate(fila):
                numero_decimal += bit * (2 ** (len(fila) - i - 1))
            numeros_decimales.append(numero_decimal)
        return numeros_decimales[::-1]

    # Convertir cada columna a decimal
    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))
    def run(self):
        #error cuadratico medio
        ecm = 1
        error_objetivo = 0.01
        #condicion de paro 
        er2 = 0
        while error_objetivo <= ecm and self.limite_de_epocas > 0:
            x1 = 0
            x2 = 0
            y_true = []  # Lista para almacenar las salidas verdaderas
            y_pred = []  # Lista para almacenar las salidas 
            errors = []
            
            #algorimo de entrenamiento
            for i in range(len(self.coordenadas)):
                x1 = self.coordenadas[i][0]
                x2 = self.coordenadas[i][1]
                w = np.array([self.w1, self.w2])
                x = np.array(self.coordenadas[i])

                y = np.array(np.dot(x,w)) + self.bias
                print('producto punto',y)
                y = self.sigmoid(y)

                for j in range(self.neuronas):
                    e = self.salidas_deseadas[i][j] - y[j]
                    errors.append(e)
                    self.w1[j]   = self.w1[j]   + self.factor_de_aprendizaje * e  * x1
                    self.w2[j]   = self.w2[j]   + self.factor_de_aprendizaje * e  * x2
                    self.bias[j] = self.bias[j] + self.factor_de_aprendizaje * e 

            ecm = np.mean(np.array(errors) ** 2)
            W = np.array([self.w1,self.w2])
            X = np.array(self.coordenadas)

            
            decimales = self.convertir_a_decimal(self.salidas_deseadas)    
            nombre = prueba.plot_contour(X,W,self.bias,decimales)
            # mandamos la seña al hilo principal
            self.update_signal.emit(self.w1, self.w2, self.bias,self.bandera,y_pred,nombre,ecm,self.limite_de_epocas)
            self.limite_de_epocas -= 1
            
            time.sleep(2)

        self.bandera = False
        self.update_signal.emit(self.w1, self.w2, self.bias,self.bandera,y_pred,nombre,ecm,self.limite_de_epocas)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.filename = None
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 750, 750)
        self.coordenadas = []
        self.salidas_deseadas = []
        self.y_pred = None
        self.limite_de_epocas = 0
        self.factor_de_aprendizaje = 0
        self.w1 = []
        self.w2 = []
        self.bias = []
        self.bandera = True
        self.pixmap_item = None
        self.Cartesiano("plano_cartesiano.png")
        self.ui.pushButton_graficar.clicked.connect(self.grafica)
        self.ui.pushButton_reset.clicked.connect(self.reset)
        self.ui.pushButton_exportar.clicked.connect(self.AbrirArchivo)
        self.ui.pushButton.clicked.connect(self.Archivo_Salidas)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_next_image)

    #seleccion de las entradas
    def AbrirArchivo(self):
        archivo, _ = QFileDialog.getOpenFileName(None, "Seleccionar archivo", "", "Archivos de texto (*.txt)")
        try:
            with open(archivo, 'r') as f:
                for linea in f:
                    x, y = map(float, linea.strip().split(','))
                    # Ajuste de coordenadas
                    self.coordenadas.append((x, y))
            nombre = prueba.plano_cartesiano(self.coordenadas)        
            self.Cartesiano(nombre)

        except Exception as e:
            QMessageBox.warning(self, 'Error', 'Archivo no válido')

    #seleccion de las salidas
    def Archivo_Salidas(self):
        archivo, _ = QFileDialog.getOpenFileName(None, "Seleccionar archivo", "", "Archivos de texto (*.txt)")
        try:
            with open(archivo, 'r') as f:
                for linea in f:
                    linea = linea.strip().split(',')
                    linea = [float(item) for item in linea]
                    self.salidas_deseadas.append(linea)
            self.iniciar_pesos(len(self.salidas_deseadas[0]))

        except Exception as e:
            QMessageBox.warning(self, 'Error', 'Archivo no válido')

    def iniciar_pesos(self,n):
        w1 = [round(random.random(), 7) for _ in range(n)]
        w2 = [round(random.random(), 7) for _ in range(n)]
        bias = [round(random.random(), 7) for _ in range(n)]
        self.w1 = w1
        self.w2 = w2
        self.bias = bias
        self.ui.lineEdit_bias.setText(str(self.bias))
        self.ui.lineEdit_w1.setText(str(self.w1))
        self.ui.lineEdit_w2.setText(str(self.w2))
    #al trabajar con una interfaz grafica de qt requiere un hilo secundario para actualizar los datos de la interfaz
    #un hilo se encarga de las actualziaciones mientras otro de las operaciones
    def adaline(self):
        
        self.thread = AdalineThread()
        self.thread.coordenadas = self.coordenadas
        self.thread.salidas_deseadas = self.salidas_deseadas
        self.thread.limite_de_epocas = self.limite_de_epocas
        self.thread.factor_de_aprendizaje = self.factor_de_aprendizaje
        self.thread.bias = self.bias
        self.thread.w1 = self.w1
        self.thread.w2 = self.w2
        self.thread.neuronas = len(self.salidas_deseadas[0])
        self.thread.update_signal.connect(self.actualizar_interfaz)
        self.thread.start()
        print('hola')
        self.timer.start(1500) 


    def actualizar_interfaz(self, w1, w2, bias,bandera,y_pred,filename,error,epocas):
        self.w1 = w1
        self.w2 = w2
        self.bias =bias
        self.ui.lineEdit_w1.setText(str(w1))
        self.ui.lineEdit_w2.setText(str(w2))
        self.ui.lineEdit_bias.setText(str(bias))
        self.bandera = bandera
        self.y_pred = y_pred
        self.filename = filename
        self.ui.lineEdit_error.setText(str(error))
        self.ui.lineEdit_restantes.setText(str(epocas))

    def reset(self):
        self.scene.clear()
        self.coordenadas.clear()
        self.Cartesiano("plano_cartesiano.png")
        self.salidas_deseadas.clear()
        self.w1.clear()
        self.w2.clear()
        self.bias.clear()
        self.ui.lineEdit_bias.setText(str(self.bias))
        self.ui.lineEdit_w1.setText(str(self.w1))
        self.ui.lineEdit_w2.setText(str(self.w2))
        self.ui.lineEdit_limite.clear()
        self.ui.lineEdit_restantes.clear()
        self.ui.lineEdit_error.clear()


    def show_next_image(self):
        if self.bandera:
            self.filename
            self.Cartesiano(self.filename)
        else:
            self.timer.stop()
    def grafica(self):
        if self.validacion():
            self.adaline()
            
    def validacion(self):
        try:
            self.factor_de_aprendizaje = float(self.ui.lineEdit_factor.text())
            self.limite_de_epocas = float(self.ui.lineEdit_limite.text())
            if self.limite_de_epocas < 0:
                QMessageBox.warning(self, 'Captura no válida', 'Ingrese solo números enteros o reales positivos.')
                return False
            if len(self.coordenadas) == 0:
                QMessageBox.warning(self, 'Ingrese entradas', 'Seleccione entradas en el plano')
                return False
            return True
        except ValueError:
            QMessageBox.warning(self, 'Captura no válida', 'Ingrese solo números enteros o reales positivos.')
            return False

    def Cartesiano(self, filename):
        pixmap = QPixmap(filename)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.ui.graphicsView.setScene(self.scene)

        self.ui.lineEdit_bias.setText(str(self.bias))
        self.ui.lineEdit_w1.setText(str(self.w1))
        self.ui.lineEdit_w2.setText(str(self.w2))

app = QApplication(sys.argv)
ventana = Window()
ventana.show()
sys.exit(app.exec_())
