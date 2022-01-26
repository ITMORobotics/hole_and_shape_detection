from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np

import sys
from PyQt5.QtWidgets import *

from threading import Thread


class GraphicRealsense(QDialog):

    def __init__(self): # Create default constructor
        super().__init__()
        self.initializeUI()

    def initializeUI(self):
        """Initialize the window and display its contents."""
        # self.setMinimumSize(1280, 800)
        # self.setWindowTitle('3D Bar Graph')
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()  ## show widget alone in its own window
        self.win.setWindowTitle('pyqtgraph example: ImageItem')
        self.view = self.win.addViewBox()
        self.view.setAspectLocked(True) ## lock the aspect ratio so pixels are always square
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)
        # self.show()

    def updateIMG(self, color_source):
        if (color_source.shape[0] < color_source.shape[1]):
            color_source = np.rot90(color_source, k=-1)
        if (len(color_source.shape) == 2):
            color_source = np.stack((color_source,) * 3, axis=-1)
        self.img.setImage(color_source)
