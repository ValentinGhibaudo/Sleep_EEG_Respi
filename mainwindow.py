# -*- coding: utf-8 -*-
from myqt import QT

import pyqtgraph as pg

from params import subjects




from mainviewer import get_viewer_from_run_key



class MainWindow(QT.QMainWindow) :
    def __init__(self, parent = None,):
        QT.QMainWindow.__init__(self, parent)

        self.resize(400,600)

        self.mainWidget = QT.QWidget()
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QT.QHBoxLayout()
        self.mainWidget.setLayout(self.mainLayout)

        self.tree = pg.widgets.TreeWidget.TreeWidget()
        self.tree.setAcceptDrops(False)
        self.tree.setDragEnabled(False)

        self.mainLayout.addWidget(self.tree)
        self.refresh_tree()

        self.tree.setContextMenuPolicy(QT.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.open_menu)

        self.all_viewers = []

    def refresh_tree(self):
        for subject in subjects:
            item  = QT.QTreeWidgetItem([u'{}'.format(subject)])
            self.tree.addTopLevelItem(item)
            item.key = subject
            
            #~ for key, row in main_index.loc[index].iterrows():
                #~ text = u' '.join('{}={}'.format(k, row[k]) for k in unique_keys)
                #~ child = QT.QTreeWidgetItem([text])
                #~ child.key = key
                #~ item.addChild(child)

    def open_menu(self, position):

        indexes = self.tree.selectedIndexes()
        if len(indexes) ==0: return

        items = self.tree.selectedItems()

        index = indexes[0]
        level = 0
        index = indexes[0]
        while index.parent().isValid():
            index = index.parent()
            level += 1
        menu = QT.QMenu()

        if level == 0:
       
            print(items[0].key)
            
            act = menu.addAction('Open viewer')
            act.key = items[0].key
            act.triggered.connect(self.open_viewer)

        #~ elif level == 1:

            #~ act = menu.addAction('Open viewer with video')
            #~ act.key = items[0].key
            #~ act.triggered.connect(self.open_viewer_with_video)

            #~ act = menu.addAction('Open states encoder')
            #~ act.key = items[0].key
            #~ act.triggered.connect(self.open_states_encoder)


        menu.exec_(self.tree.viewport().mapToGlobal(position))

    

    def _open_viewer(self, with_video=False):
        run_key = self.sender().key
        print(run_key)
        w = get_viewer_from_run_key(run_key=run_key, parent=self, with_video=with_video)
        w.show()
        w.setWindowTitle(run_key)
        self.all_viewers.append(w)

        for w in [w  for w in self.all_viewers if w.isVisible()]:
            self.all_viewers.remove(w)
    
    
    def open_viewer(self):
        self._open_viewer(with_video=False)
    
    #~ def open_viewer_with_video(self):
        #~ self._open_viewer(with_video=True)

    #~ def open_states_encoder(self):
        #~ run_key = self.sender().key
        #~ print(run_key)
        #~ w = get_encoder(run_key=run_key, parent=self)
        #~ w.show()
        #~ w.setWindowTitle(run_key)
        
        #~ self.all_viewers.append(w)

        #~ for w in [w  for w in self.all_viewers if w.isVisible()]:
            #~ self.all_viewers.remove(w)
        

if __name__ == '__main__':
    app = pg.mkQApp()
    w = MainWindow()
    w.show()
    app.exec_()
