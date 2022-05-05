# importing the required libraries
import os
import shutil
import sys
import cv2
import time
from glob import glob
import nibabel as nib
import numpy as np
from PIL import Image
import json
import nrrd


from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import * 
from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSignal, QEvent
from PyQt5 import uic
# Load modules 
from scripts import Autils, Autils_fault_prediction, Autils_fault_detection_DSP, Autils_fault_detection_DSP_3D, Autils_Object_detection
from scripts import Jutils_3D_registration, Jutil_2D_registration, Jutils_2D_registration_BS, Jutils_3D_characterization, Jutils_3D_pre_proc
from scripts.util_sql import *

# VTK
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support

# turn off warnings
import warnings
warnings.filterwarnings("ignore")
# turn off vtk warnings window
vtk_out = vtk.vtkOutputWindow()
vtk_out.SetInstance(vtk_out)
# turn off vtk warnings in command prompt
from vtkmodules.vtkCommonCore import vtkLogger
vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_OFF)

# the connection to all database functions
sqlcon = SQLConnect('data/')


# Scanner
class UserClass():
    def __init__(self, user_name = 'User0',user_type = 'EndUser' ):
        # enter image dimensions (width x height x depth)
        self.user_name = user_name
        self.user_type = user_type

## MAIN WINDOWS
class MainApp(QMainWindow):
    def __init__(self):
        #Parent constructor
        super(MainApp,self).__init__()
        self.hide()
        self.user = UserClass(user_name = 'User0',user_type = 'EndUser')
        #self.user = UserClass(user_name = 'User0',user_type = 'ExpertUser')
        sep_OS = os.path.join('1','1')[1] 
        self.sep_OS = sep_OS
        self.root_folder = 'data/'
        self.ext_list_3d = ['.rek' , '.nrrd', '.nii']
        self.ext_list_2d = ['.tif' , '.tiff', '.jpg', '.png']

        ## SQL variable
        self.tool_name = ''
        self.data_root = ''
        self.tool_list_names = [""]
        self.tool_list_names.extend(sqlcon.get_all_tool())

        # selected files/folders
        self.file_path = None
        self.param=[]
        self.msg = ''
        self.empthy_img = 'files/empty_img.jpg'
        self.folder_before = 'data/tool1/CS0/'
        self.folder_after = 'data/tool1/CS1/'
        # algorithm
        self.select_algo = None
        self.select_model = None
        self.algorithm = None
        self.type_algo = '2D'
        self.model_path_list = None
        self.output_dic = None

        # 2D plots 
        self.plot_1 = None
        self.plot_2 = None
        self.plot_3 = None

        # 3D plots 
        self.vtk_plot_1 = None
        self.vtk_plot_2 = None
        self.vtk_plot_3 = None

        self.vtk_widget_1 = None
        self.vtk_widget_2 = None
        self.vtk_widget_3 = None
        # 3D plots 
        self.config_algo = 0

        # Backgrouond subtraction
        self.root_backgrouond_subtraction = os.getcwd()  + '/models/backgrouond-subtraction/'
        self.list_backgrouond_subtraction_algorithms =   self.get_subfolders(self.root_backgrouond_subtraction) 
        self.models_list_backgrouond_subtraction_algorithms= [''] 

        # Registration
        self.root_registration = os.getcwd()  + '/models/registration/'
        self.list_registration_algorithms =   self.get_subfolders(self.root_registration) 
        self.models_list_registration_algorithms= [''] 

        # classification
        self.root_classification = os.getcwd()  + '/models/classification/'
        self.list_classification_algorithms =   self.get_subfolders(self.root_classification) 
        self.models_list_classification_algorithms = [''] 
        # Localization
        self.root_localization = os.getcwd()  + '/models/localization/'
        self.list_localization_algorithms =  self.get_subfolders(self.root_localization) 
        self.models_list_localization_algorithms = ['']#self.get_trained_models(self.root_localization +  "2D*.pth") 
        # charachterization
        self.root_charachterization = os.getcwd()  + '/models/charachterization/'

        self.list_charachterization_algorithms = self.get_subfolders(self.root_charachterization) 
        self.models_list_charachterization_algorithms =['']# self.get_trained_models(self.root_charachterization +  "2D*.pth") 

        self.createMenu()
        self.windows_format()

        # to complete - apply progress for the following:
        # 1. loading 'very large' data
        # 2. loading multiple scans takes time
        # 3. applying algorithms that take time
        # 4. timing training algorithms
        
        # progress bar 
        self.progress_windows = Progress_bar_windows()
        self.progress_windows.hide()
        self.progress_percentage = 0
        self.progress_trigger =QLineEdit()
        self.progress_trigger.hide()
        self.progress_trigger.textChanged.connect(self.show_progress_bar)
        # self.progress_name = 'progress'        
        # self.simulate_progress()
        
    def show_progress_bar(self):
        print('self.progress_trigger.Text()=', self.progress_trigger.text() )
        if self.progress_trigger.text()=='100':
            self.stop_progress()

        else: 
            percentage =  float( self.progress_trigger.text() ) 
            self.progress_windows.bar1.setValue(percentage)
            print('update the progress' + str(percentage) + '%')

    def start_progress(self, title = 'progress'):
        self.progress_windows.show()
        self.progress_windows.bar1.setValue(0)
        self.progress_windows.setWindowTitle(title)

    def stop_progress(self):
        self.progress_windows.hide()
        self.progress_windows.bar1.setValue(0)
        self.progress_percentage = 0

    def set_percentage(self, percentage):
        self.progress_trigger.setText(str(percentage)) 

    def simulate_progress(self):
        self.start_progress()
        for k in range(6):
            self.increment_progress(3)  # increment progreee with 3 

    def update_table(self, table, list_scan_data):

        #self.start_progress(title = 'Loading data from SQL')
        # input - dataframe
        print("Table inside GUI:", list_scan_data)
        
        if list_scan_data.shape[0] == 0:
            table.clear()
            return

        list_scan_data_col = pd.DataFrame([list_scan_data.columns.values], columns=list_scan_data.columns)
        list_scan_data = pd.concat([list_scan_data_col, list_scan_data], axis=0)
        table.setRowCount(list_scan_data.shape[0])
        table.setColumnCount(list_scan_data.shape[1])
        
        #Matthew:  setup progress bar 
        cnt = 0
        total_cnt = list_scan_data.shape[0]*list_scan_data.shape[1] 

        for i in range(list_scan_data.shape[0]):
            for j in range(list_scan_data.shape[1]):
                print(list_scan_data.iloc[i, j])
                table.setItem(i, j, QTableWidgetItem(str(list_scan_data.iloc[i, j])))

                #Matthew:   increment bar
                #cnt = cnt +1
                #step_pregress = 100*cnt/(total_cnt-1)
                #self.set_percentage(step_pregress)
           
        
        #Matthew:   stop the progress bar 
        #self.stop_progress()
                
        #         combo = QComboBox()
        #         # if i  folder: 
        #         for k in ["images","volume", "fault", "mask"]: 
        #             combo.addItem(k)

        #         self.table.setCellWidget(i, 2, combo)

        # for i in range(self.table.rowCount()):
        #     ch = QCheckBox(parent=self.table)
        #     ch.clicked.connect(self.onStateChanged)
        #     self.table.setCellWidget(i, 1, ch)
        

        #return table
    
    def update_table_training(self, table, list_scan_data):
        # input - dataframe
        print("Table inside GUI:", list_scan_data)
        
        if list_scan_data.shape[0] == 0:
            table.clear()
            return

        list_scan_data_col = pd.DataFrame([list_scan_data.columns.values], columns=list_scan_data.columns)
        list_scan_data = pd.concat([list_scan_data_col, list_scan_data], axis=0)
        table.setRowCount(list_scan_data.shape[0])
        table.setColumnCount(list_scan_data.shape[1])
        
        for i in range(list_scan_data.shape[0]):
            for j in range(list_scan_data.shape[1]):
                print(list_scan_data.iloc[i, j])
                table.setItem(i, j, QTableWidgetItem(str(list_scan_data.iloc[i, j])))


                '''
                combo = QComboBox()
                # if i  folder: 
                for k in ["images","volume", "fault", "mask"]: 
                    combo.addItem(k)

                self.table.setCellWidget(i, 2, combo)

        for i in range(self.table.rowCount()):
            ch = QCheckBox(parent=self.table)
            ch.clicked.connect(self.onStateChanged)
            self.table.setCellWidget(i, 1, ch)
        '''

        return table
    
    def create_new_folder(self, DIR):
        if not os.path.exists(DIR):
            os.makedirs(DIR)

    def showdialog_newTool(self):
        root_folder = self.root_folder 
        self.dialog_widget = Dialog_ToolForm(root_folder)
        self.dialog_widget.show()

    def showdialog_newScan(self, tool_name):
        root_folder = os.path.join(self.root_folder , tool_name)
        self.dialog_widget = Dialog_ScanForm(root_folder, user=self.user.user_name)

        self.dialog_widget.show()

    def showdialog_updateTool(self, tool_name):
        # using the update_flag (update) to differentiate between new tool and update tool
        self.dialog_widget = Dialog_ToolForm(tool_name, update_flag=1)
        self.dialog_widget.show()

    def showdialog_updateScan(self, tool_name, scan_name):
        # using the update_flag (update) to differentiate between new tool and update tool
        self.dialog_widget = Dialog_ScanForm(tool_name, scan_name=scan_name, user=self.user.user_name, update_flag=1)
        self.dialog_widget.show()
        
    def showdialog_question(self, title, message):
        reply = QMessageBox.question(self, title, message,
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply
    
    def showdialog_deleteTool(self, tool_name):
        delete_tool_title = "Tool To Be Deleted"
        delete_tool_msg = "Are you sure you want to delete tool - " + str(tool_name) + " - ?\nNote: All scans under the tool and related data will also be deleted!" 
        print(delete_tool_msg)
        retval = self.showdialog_question(delete_tool_title, delete_tool_msg)
                
        if retval == QMessageBox.Yes:
            tool_location = sqlcon.get_tool_location(tool_name)
            if os.path.isdir(tool_location):
                shutil.rmtree(tool_location)
            sqlcon.delete_tool(tool_name)
            Dataset_Win.list_tools.clear()
            tool_name_list = [""]
            tool_name_list.extend(sqlcon.get_all_tool())
            Dataset_Win.list_tools.addItems(tool_name_list)
            
            delete_tool_msg = "The tool - " + str(tool_name) + " - has been deleted!" 
            print(delete_tool_msg)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(delete_tool_msg)
            msg.setWindowTitle("Tool Deletion Complete")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()
            #self.showdialog_information(delete_tool_msg)

    def showdialog_deleteScan(self, tool_name, scan_name):
        delete_scan_title = "Scan To Be Deleted"
        delete_scan_msg = "Are you sure you want to delete scan - " + str(scan_name) + " - under the tool - " + str(tool_name) + " - ?\nNote: All scan specific data will be deleted!" 
        print(delete_scan_msg)
        retval = self.showdialog_question(delete_scan_title, delete_scan_msg)
        
        if retval == QMessageBox.Yes:
            scan_location = sqlcon.get_scan_location(scan_name)
            if os.path.isdir(scan_location):
                shutil.rmtree(scan_location)
            sqlcon.delete_scan(scan_name)
            list_tool_scans = sqlcon.get_all_scan_table(tool_name)
            Dataset_Win.update_table(Dataset_Win.table, list_tool_scans)
            
            delete_scan_msg = "The scan - " + str(scan_name) + " - under the tool - " + str(tool_name) + " - has been deleted!" 
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(delete_scan_msg)
            msg.setWindowTitle("Scan Deletion Complete")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()
            #self.showdialog_information(delete_scan_msg)
      
    def showdialog_registration(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Do you accept the registration?")
        msg.setInformativeText("Info: Registration is the operation of geometrically  aligning two tools")
        msg.setWindowTitle("Expert validation")
        msg.setDetailedText("If you select yes, that data will be saved to database.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()

        if retval == 65536:
                print("Saving without registration\n --> outputs = ", self.output_dic)

                #save data
                # Matthew: call function to save data without preprocessing
        else: 
                print("Saving with registration\n --> outputs = ", self.output_dic)
                main_Win.showdialog_operation_success("The preprocessed data is saved to the database.")
                # Matthew: call function to save data after preprocessing
        
        print("value of pressed message box button:", retval)

    def showdialog_FullInspection_results_saving(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Are you satisfied by the full inspection?")
        msg.setInformativeText( "Info: The full inspection conducts the following operations: \n- Detects/Recognizes the type, location and name of the defect \
                              (missing parts, broken parts) \n - Characterizes the fault area percentage ")
        msg.setWindowTitle("Expert validation")
        msg.setDetailedText("If you select yes, the result will be saved for future learning.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()

        if retval == 65536:
                print("Saving results as unverified knowledge base.\n --> outputs = ", self.output_dic)

                #save data
                # Matthew: call function to save data without preprocessing
                
        else: 
                print("Saving results as verified knowledge base\n --> outputs = ", self.output_dic)

                main_Win.showdialog_operation_success("The data is saved to the database  for future learning")
                # Matthew: call function to save data after preprocessing

    def showdialog_Localization_results_saving(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Are you satisfied by this defect localization?")
        msg.setInformativeText("Info: Defect localization predicts the exact location of all potential faults (missing parts, broken parts)  in an the fauts in the 3D XCT scans ")
        msg.setWindowTitle("Expert validation")
        msg.setDetailedText("If you select yes, the result will be saved for future learning.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()

        if retval == 65536:
                print("Saving results as unverified knowledge base\n --> outputs = ", self.output_dic)

                #save data
                # Matthew: call function to save data without preprocessing 

        else: 
                print("Saving results as verified knowledge base\n --> outputs = ", self.output_dic)

                main_Win.showdialog_operation_success("The data is saved to the database for future learning.")
                # Matthew: call function to save data after preprocessing
    
    def showdialog_train_verify(self, data):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Do you accept the performance of the model training?")
        # msg.setInformativeText("Info: Registration is the operation of geometrically  aligning two tools")
        msg.setWindowTitle("Expert validation")
        msg.setDetailedText("If you select yes, the model will be saved. Otherwise provide more data for training")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()
        # print("value of pressed message box button:", retval)
        return retval

    def showdialog_fault_prediction_verify(self, data):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Do you accept the result of fault prediction?")
        # msg.setInformativeText("Info: Registration is the operation of geometrically  aligning two tools")
        msg.setWindowTitle("Expert validation")
        msg.setDetailedText("If you select yes, the result will be saved for future learning")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()

        if retval == 65536:
                print("save the data but do not use for future learning")
                # Matthew: save result data but not for future learning 
        else: 
                print("Save data for future learning")

                # Matthew: save result data for future learning
                main_Win.showdialog_operation_success("Result is saved for future learning.")
        
        print("value of pressed message box button:", retval)

    def showdialog_operation_success(self, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText(text)
        # msg.setInformativeText("Info: Registration is the operation of geometrically  aligning two tools")
        msg.setWindowTitle("Operation Successful")
        # msg.setDetailedText("If you select yes, the result will be saved for future learning")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()

        if retval == 65536:
                print("save the data but do not use for future learning")
                # call function 
                # Matthew: save result data but not for future learning
        else: 
                print("Save data for future learning")
                # Matthew: save result data for future learning
        
        print("value of pressed message box button:", retval)

    def showdialog_request_more_data(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Do you have more data to train the model?")
        # msg.setInformativeText("Info: Registration is the operation of geometrically  aligning two tools")
        msg.setWindowTitle("Expert validation")
        msg.setDetailedText("If you select yes, we will re-train the model with more data. If no, model will not be trained")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()

        if retval == 65536:
                print("no data available")
        else: 
                print("retrain the model")
                inpath = QFileDialog.getOpenFileName(self, "Add new scan")[0]
                
                # call model retrain with more data
        
        print("value of pressed message box button:", retval)

    def msgbtn(self, i):
        print("Button pressed is:",i.text())

    def get_subfolders(self, root, patern = ''):
        return [ name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)) if patern in name  ]
    
    def get_trained_models(self, root):
        from glob import glob
        paths =  glob(root) 
        model_names = []
        for path in paths:
            model_names.append( os.path.basename(path) )
        return model_names

    def windows_format(self):
        # set the title
        self.setWindowTitle("CT-Based Integrity Monitoring System (CTIMS) 2021")
        # setting  the geometry of window
        # self.setGeometry(0, 0, 400, 300)
        self.setWindowIcon(QIcon('files/icon.png'))
        # self.resize(1800, 1000)
        self.showMaximized()
        self.hide()

    def createMenu(self):
        self.hide()
        self.text = QPlainTextEdit()
        #/// menu: dataset
        self.menu_stdrd = self.menuBar().addMenu("&File")
        self.action_add_new_tool = QAction("&Add new tool")
        self.action_add_new_tool.triggered.connect(self.add_new_tool)
        self.action_add_new_tool.setShortcut(QKeySequence.Open)
        self.menu_stdrd.addAction(self.action_add_new_tool)

        self.open_new_scan  = QAction("&Add new scan")
        self.open_new_scan.triggered.connect(self.add_new_scan)
        self.menu_stdrd.addAction(self.open_new_scan)

        self.update_tool_data = QAction("&Update tool data")
        self.update_tool_data.triggered.connect(self.update_tool_values)
        self.menu_stdrd.addAction(self.update_tool_data)

        self.update_scan_data = QAction("&Update scan data")
        self.update_scan_data.triggered.connect(self.update_scan_values)
        self.menu_stdrd.addAction(self.update_scan_data)

        self.delete_tool_data = QAction("&Delete tool")
        self.delete_tool_data.triggered.connect(self.delete_tool_values)
        self.menu_stdrd.addAction(self.delete_tool_data)

        self.delete_scan_data = QAction("&Delete scan")
        self.delete_scan_data.triggered.connect(self.delete_scan_values)
        self.menu_stdrd.addAction(self.delete_scan_data)

        self.del_DB = QAction("&Clean/Reset the database")
        self.del_DB.triggered.connect(self.clean_reeset_DB)
        self.menu_stdrd.addAction(self.del_DB)

        self.menu_stdrd.addSeparator()

        self.close_windows_action = QAction("&Exit")
        self.close_windows_action.triggered.connect(self.close)
        self.menu_stdrd.addAction(self.close_windows_action)

        #/// menu: Dataset
        self.dataset_menu = self.menuBar().addMenu('&Dataset preparation')
        # sub-menu: Vizualisation
        self.viz_data_action = self.dataset_menu.addAction("&Visualization")
        self.viz_data_action.triggered.connect(self.visualize_dataset)

        # # # sub-menu: upload
        # self.annotate_dataset = self.dataset_menu.addMenu('Annotation')
        # self.annotate_obj_dtection = self.annotate_dataset.addAction('Object detection ')
        # self.annotate_segmentation = self.annotate_dataset.addAction('Segmentation ')

        #/// menu: Preprocessing
        self.peprocessing = self.menuBar().addMenu('&Pre-processing')
        # sub-menu: Background subtraction
        # self.pre_backgnd_supp = self.peprocessing.addAction('Background subtraction')
        # self.pre_bkgnd_supp_2D = self.pre_backgnd_supp.addAction('2D image ')
        # self.pre_bkgnd_supp_3D = self.pre_backgnd_supp.addAction('3D volume ')

        # self.peprocessing.addSeparator()
        # sub-menu: registration
        self.pre_registration = self.peprocessing.addAction('Registration')
        self.pre_registration.triggered.connect(self.Registration)

        #/// menu: Model deployment
        self.inspection = self.menuBar().addMenu('&CT Inspection')

        if self.user.user_type == 'EndUser':
            # sub-menu: Fault characterization
            self.run_fault_characterization_action =self.inspection.addAction("Fault characterization (Automated full inspection)")
            self.run_fault_characterization_action.triggered.connect(self.Automated_framework)

        else:
            
            # sub-menu: Fault prediction
            self.run_fault_detection_action =self.inspection.addAction("Fault prediction")
            self.run_fault_detection_action.triggered.connect(self.Fault_prediction)
            
            # sub-menu: Fault localization
            self.run_fault_localization__action =self.inspection.addAction("Fault localization")
            self.run_fault_localization__action.triggered.connect(self.Fault_localization)
            
            # sub-menu: Fault characterization
            self.run_fault_characterization_action =self.inspection.addAction("Fault characterization (Automated full inspection)")
            self.run_fault_characterization_action.triggered.connect(self.Fault_characterization)

        #/// menu: help
        self.help_menu = self.menuBar().addMenu("&Help")
        self.about_action = QAction("About")
        self.help_menu.addAction(self.about_action)
        self.help_Tutorials = QAction("Tutorials")
        self.help_menu.addAction(self.help_Tutorials)
        self.help_contact = QAction("Contact")
        self.help_menu.addAction(self.help_contact)
        self.about_action.triggered.connect(self.show_about_dialog)

    def showdialog_information(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Information message")
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval

    def showdialog_warnning(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Error message")
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval

    def path_end(self, location):
        if os.path.isfile(location) == True:
            sample_pth = "/".join([x for x in location.split('/')[-3:]])
            return sample_pth
        else:
            return location

    def Plot_array(self, img, figure):

        dim=(500,500)
        # resize image
        image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        image = QImage(image.data, image.shape[0], image.shape[1], QImage.Format_RGB888).rgbSwapped()
        figure.setPixmap(QPixmap.fromImage(image))

    def Plot_image(self, path, figure):
        
        if self.file_image_or_volume(path) == '2D':
            msg = ''
            img = cv2.imread(path)
            dim=(500,500)
            # resize image
            image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            image = QImage(image.data, image.shape[0], image.shape[1], QImage.Format_RGB888).rgbSwapped()
            figure.setPixmap(QPixmap.fromImage(image))

            # pixmap = QPixmap(path)
            # figure.setPixmap(pixmap)
            # self.resize(100,100)
            figure.show()

        else: 
            msg = '\n Warning(2) : The selected image file is not supported!! \n - file path = '+ self.path_end(path)

        return msg

    def get_image(self, path):
        import numpy as np
        print(' The selected image is :', path)
        filename, file_extension = os.path.splitext(path)
        img = cv2.imread(path,0)

        return img

    def get_supported_volume(self, root, ext_list = ['.rek' , '.nrrd', '.nii']):
        from glob import glob 
        vol_list = []
        for ext in ext_list:
            vol_list = vol_list + glob( os.path.join(root, '*' + ext) )

        return vol_list


    def get_volume(self, path):
        import numpy as np
        print(' Loading the file :', path)
 
        filename, file_extension = os.path.splitext(path)
        
        if os.path.exists(path):
            # read 3D volume from rek file
            if file_extension=='.rek':
                # enter slice dimensions (width x height x depth)
                image_width = 500
                image_height = 500
                image_depth = 500
                # enter voxel datatype ("float32" or "uint16")
                voxel_datatype = "uint16"
                # read 3D volume from rek file
                volume_array = Autils_fault_detection_DSP_3D.simpleRek2Py(path, image_width, image_height, image_depth, voxel_datatype)  
            
            elif file_extension=='.bd': 

                volume_array = Autils_fault_detection_DSP_3D.VGI2Py(path)
  

            elif file_extension=='.nii':
                volume_array = nib.load(path).get_data()
                # volume_array =np.asanyarray( nib.load(path).dataobj)

            elif file_extension=='.nrrd':
                import nrrd
                volume_array, _ = nrrd.read(path)

            else: 
                print(' Warning: The file format is not supported. a ones volume will be generated!!!!')
                volume_array = np.ones((10, 10, 10))

        else:
            print(' Warning: The file is not found. a ones volume will be generated!!!!')
            volume_array = np.ones((10, 10, 10))
        
        # print('volume_array values= ', np.unique(volume_array.ravel()) )

        volume_vtk = Autils.get_vtk_volume_from_3d_array(volume_array)            
        # print(' volume shape = ' , volume_array.shape)
        # print(' volume unique = ' , np.unique(volume_array))
        # print(' volume segments = ' , len(np.unique(volume_array)) )
        # print(' volume type = ' , type(volume_array) )
        
        return volume_array, volume_vtk

    def file_image_or_volume(self, path):
        filename, file_extension = os.path.splitext(path)
        if file_extension =='.rek' or file_extension =='.bd' or file_extension =='.nii' or file_extension =='.nrrd':
            return '3D' 
        elif file_extension =='.tif' or file_extension =='.tiff' or file_extension =='.jpg' or file_extension =='.png':
            return '2D'

        elif file_extension =='.txt':
            return 'txt'

        else: 
            print('Error : the input format is not supported as image of volume:', path)

    def Plot_volume(self, vtk_widget, volume, volume_segments=[0]):
        # update the plot
        vtk_widget.update_volume(volume, volume_segments=volume_segments)
        vtk_widget.start()

    def initiate_widget(self, vtk_widget, vtk_plot):
        vtk_widget = QGlyphViewer()      
        vtk_layout_1 = QHBoxLayout()
        vtk_layout_1.addWidget(vtk_widget)
        vtk_layout_1.setContentsMargins(0,0,0,0)
        vtk_plot.setLayout(vtk_layout_1)
        vtk_widget.start()
        return vtk_widget, vtk_plot

    def create_vtk_widget(self, vtk_widget, vtk_plot, path0):
        vtk_widget = QGlyphViewer()      
        volume_arr, volume = self.get_volume( path0)
        volume_segments = np.array(np.unique(volume_arr.ravel()) )
        vtk_widget.update_volume(volume, volume_segments=volume_segments)
        vtk_layout_1 = QHBoxLayout()
        vtk_layout_1.addWidget(vtk_widget)
        vtk_layout_1.setContentsMargins(0,0,0,0)
        vtk_plot.setLayout(vtk_layout_1)
        vtk_widget.start()
        return vtk_widget, vtk_plot

    def save_as(self):
        # Matthew : save to database 
        path = QFileDialog.getSaveFileName(self, "Save As")[0]
        print("Save as function path:", path)
        if path:
            self.file_path = path
            self.save()

    def save(self):
        if self.file_path is None:
            print("Save path missing:", self.file_path)
            self.save_as()
        else:
            print("Save function path:", self.file_path)
            with open(self.file_path, "w") as f:
                f.write(self.text.toPlainText())
            self.text.document().setModified(False)

    def show_about_dialog(self):
        
        text = "<center>" \
            "<h1> CT-Based Integrity Monitoring System (CTIMS) </h1>" \
            "&#8291;" \
            "<img src=files/logo.png>" \
            "</center>" \
            "<p>  This software processes CT scans of an object before and after its use for Vault maintenance. It uses new algorithms to detect defects and localize their positions/locations within this object<br/>" \
            "Email:amrb@nvscanada.ca<br/></p>"
        QMessageBox.about(self, "About CTIMS 2021", text)
 

    def tool_msgbtn1(self):
        return

    def add_new_tool(self):
        print('Add new tool',)
        Dataset_Win.show()
        Algorithms_Win.hide()
        main_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()
        
        if Dataset_Win.list_tools.currentText() != "":
            select_tool = "Please empty the name from the \"Select the Tool\" drop down box"
            print(select_tool)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(select_tool)
            msg.setWindowTitle("Not Data Directory")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()
        else:
            Dataset_Win.showdialog_newTool()

    def information_mssage():
        a=1
        

    def add_new_scan(self):
        print('Add new scan',)
        Dataset_Win.show()
        Algorithms_Win.hide()
        main_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()

        if Dataset_Win.list_tools.currentText() == "":
            select_tool = "Please select a tool from the \"Select the Tool\" drop down box"
            print(select_tool)
            self.showdialog_warnning(select_tool)
            # msg = QMessageBox()
            # msg.setIcon(QMessageBox.Information)sss
            # msg.setText(select_tool)
            # msg.setWindowTitle("Tool Not Selected")
            # msg.setStandardButtons(QMessageBox.Ok)
            # retval = msg.exec_()
        else:
    
            tool_name = Dataset_Win.list_tools.currentText()
            Dataset_Win.showdialog_newScan(tool_name)
        
    def update_tool_values(self):
        print('Updating selected tool')
        Dataset_Win.show()
        Algorithms_Win.hide()
        main_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()
        
        if Dataset_Win.list_tools.currentText() == "":
            select_tool = "Please select which tool you want to update from the \"Select the Tool\" drop down box"
            print(select_tool)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(select_tool)
            msg.setWindowTitle("Tool Not Selected")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()
        else:
            tool_name = Dataset_Win.list_tools.currentText()
            Dataset_Win.showdialog_updateTool(tool_name)
    
    def get_scan_from_tool(self, tool_name):
        scans = sqlcon.get_all_tool_scan(tool_name)
        scan, ok_button = QInputDialog.getItem(self, "Get Scan Name", "Available Scan Names:", scans, 0, False)
        if ok_button and scan:
            print("Get scan to be updated from tool:", scan)
            return scan
    
    def update_scan_values(self):
        print('Updating selected scan',)
        Dataset_Win.show()
        Algorithms_Win.hide()
        main_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()

        if Dataset_Win.list_tools.currentText() == "":
            select_tool = "Please select a tool from the \"Select the Tool\" drop down box"
            print(select_tool)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(select_tool)
            msg.setWindowTitle("Tool Not Selected")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()
        else:
            tool_name = Dataset_Win.list_tools.currentText()
            scan_count = len(sqlcon.get_all_tool_scan(tool_name))
            if scan_count == 0:
                msg = "No available scans for this tool!"
                self.showdialog_warnning(msg)
            else:
                scan_name = self.get_scan_from_tool(tool_name)
                if scan_name == "" or scan_name == None:
                    return
                Dataset_Win.showdialog_updateScan(tool_name, scan_name)

    def delete_tool_values(self):
        print('Deleting selected tool')
        Dataset_Win.show()
        Algorithms_Win.hide()
        main_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()
        
        if Dataset_Win.list_tools.currentText() == "":
            select_tool = "Please select which tool you want to delete from the \"Select the Tool\" drop down box"
            print(select_tool)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(select_tool)
            msg.setWindowTitle("Tool Not Selected")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()
        else:
            tool_name = Dataset_Win.list_tools.currentText()
            Dataset_Win.showdialog_deleteTool(tool_name)
    
    def get_scan_from_tool(self,tool_name):
            scans = sqlcon.get_all_tool_scan(tool_name)
            scan, ok_button = QInputDialog.getItem(self, "Get Scan Name", "Available Scan Names:", scans, 0, False)
            if ok_button and scan:
                print("Get scan to be deleted from tool:", scan)
                return scan

    def delete_scan_values(self):
        print('Deleting selected scan',)
        Dataset_Win.show()
        Algorithms_Win.hide()
        main_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()
   
        if Dataset_Win.list_tools.currentText() == "":
            select_tool = "Please select a tool from the \"Select the Tool\" drop down box"
            print(select_tool)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(select_tool)
            msg.setWindowTitle("Tool Not Selected")
            msg.setStandardButtons(QMessageBox.Ok)
            retval = msg.exec_()
        else:
            tool_name = Dataset_Win.list_tools.currentText()
            scan_count = len(sqlcon.get_all_tool_scan(tool_name))
            if scan_count == 0:
                msg_txt = "No available scans for this tool!"
                print(msg_txt)
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText(msg_txt)
                msg.setWindowTitle("No Available Scans")
                msg.setStandardButtons(QMessageBox.Ok)
                retval = msg.exec_()
            else:
                scan_name = self.get_scan_from_tool(tool_name)
                if scan_name == "" or scan_name == None:
                    return
                Dataset_Win.showdialog_deleteScan(tool_name, scan_name)

    def showdialog_clean_DB(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)

        list_tools = sqlcon.get_all_tool()

        msg.setText("Are you sure that you want the " + str(len(list_tools)) + " CT scans from database permanetly! ?")
        msg.setWindowTitle("Clean the database")
        msg.setDetailedText("If you select yes, all CT-scan files and database will be deleted and  cannot be restored later.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()
        return retval

    def clean_reeset_DB(self):
        print('Cleaning and resetting the database',)
        Dataset_Win.show()
        Algorithms_Win.hide()
        main_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()

        # warning 
        retval = self.showdialog_clean_DB()

        print('retval=', retval)

        if retval == 1024 : # Clean
            self.start_progress(title = 'Database cleaning')

            self.progress_percentage = 1
            self.progress_trigger.setText(str(self.progress_percentage))
            # remove tables 
            sqlcon.drop_all_tab()
            sqlcon.init_all_tab()

            self.progress_percentage = 20
            self.progress_trigger.setText(str(self.progress_percentage))
            # remove files
            self.progress_percentage = 20
            list_tools = sqlcon.get_all_tool()
            print('list_tools=', list_tools)

            for tool in list_tools:
                folder_path = os.path.join(self.root_folder, tool) 
                # os.rmdir(folder_path)

            self.progress_percentage = 100
            self.progress_trigger.setText(str(self.progress_percentage))

            self.showdialog_information('The database is cleaned and reset successfully!!')

    def do_task(self):
        for k in range(98):      
            x = k**2
        self.progress_trigger.setText(str(self.progress_percentage))

    def showdialog_info_only(self, msg_txt, title_txt):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(msg_txt)
        msg.setWindowTitle(title_txt)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval

    def showdialog_warn_only(self, msg_txt, title_txt):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(msg_txt)
        msg.setWindowTitle(title_txt)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval
     
    def get_selected_path(self, index):
        """ for test """
        if not index.isValid():
            return
        model = index.model()
        path = model.fileInfo(index).absoluteFilePath()

        msg = 'Uploading in process .... \n file = ' + path
        print(msg);self.ui.msg_label.setText(msg)

        self.file_before = path
        msg = ''

        type_file = self.file_image_or_volume(path)


        if os.path.isfile(path): 
            if type_file == '2D':
                # plot te selected image
                self.Plot_image(self.file_before, self.plot_1)

            elif type_file=='txt':
                # display the selected text 
                self.edit_1.setPlainText(open(self.file_before ).read())

            elif type_file == '3D':
                # display the selected Rek volume 
                volume_arr, vol0  = self.get_volume(self.file_before)
                volume_segments = np.array(np.unique(volume_arr.ravel()))
                self.Plot_volume(self.vtk_widget_1 , vol0, volume_segments=volume_segments)
            else: 

                msg=  '- Error (1): Unrecognized format!!. Cannot open the selected file!!\n - [[' + path + ']'

        else:
            msg=  '- Warning (1):  You selected a folder not a file!!\n - ' + self.path_end(path)
                    
            
        msg = msg + self.msg
        print(msg);self.ui.msg_label.setText(msg)
   
    def SQL_add_new_tool(self, tool_sample):     
        
        tool_name = tool_sample['tool_name']
        tool_location = tool_sample['tool_location']
        cad_path = tool_sample['cad_path']
        cad_available = tool_sample['cad_available']
        numbers_components = tool_sample['numbers_components']
        names_components = tool_sample['names_components']

        if tool_name in sqlcon.get_all_tool():
            repeat_tool_name = "Selected tool name " + "\"" + str(tool_name) + "\"" + " already exists!"
            repeat_tool_name_title = "Tool Already Exists"
            self.showdialog_warn_only(repeat_tool_name, repeat_tool_name_title)
            return

        elif tool_name in os.listdir(self.root_folder):
            repeat_tool_name = '''Selected tool name " + "\"" + str(tool_name) + "\"" + " is in file structure, but not database, folder is added manually!
                                \n Please remove the tool folder or change the tool name!'''
            repeat_tool_name_title = "Tool Already Exists"
            self.showdialog_warn_only(repeat_tool_name, repeat_tool_name_title)
            return
            
        elif tool_name != "":
            os.mkdir(tool_location)
            # adding tool data into the DB
            if cad_available == True:
                print("Changing the path of CAD data...")
                cad_folder_name = cad_path.split('/')[-1]
                new_cad_path = os.getcwd() + '/' + tool_location + '/' + cad_folder_name
                print("CAD present:", cad_folder_name, new_cad_path)

                shutil.copytree(cad_path, new_cad_path)
                new_cad_name = tool_location + '/' + 'STL'
                os.rename(new_cad_path, new_cad_name)
                cad_path = new_cad_name
                tool_sample['cad_path'] = cad_path
                
                # save component  names txt file 
                txt_filename = tool_location + '/' + 'names_components.txt'
                print('names_components = ', names_components)
                Autils.write_string_list(txt_filename, names_components)
                tool_sample['names_components'] = txt_filename

            sqlcon.sql_add_new_tool(tool_sample)
            print("New tool added to File Structure and Database...")
            
            new_tool_added = "New tool " + "\"" + str(tool_name) + "\"" + " has been created!"
            new_tool_added_title = "Adding New Tool"
            self.showdialog_info_only(new_tool_added, new_tool_added_title)
            
            # SQL load all tools in  DB
            self.tool_list_names = [""]
            self.tool_list_names.extend(sqlcon.get_all_tool())
            #print('loaded list of tool ===', self.tool_list_names)
            # update the tool list into the tool dropbox
            self.list_tools.clear()
            self.list_tools.addItems(self.tool_list_names)

        else:
            print("No tool name provided, hence no new tool can be created...")
    
    def showdialog_scan_name_exists(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("The scan name already exits. Do you want to overwite the scan data")
        msg.setInformativeText("Info: If you select yes, the scan will be overritten. \n    If you select yes, Otherwise the scan will be replicated")
        msg.setWindowTitle("Existing scan warning")
        msg.setDetailedText("If you select yes, that data will be saved to database.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()

        return retval
    
    def auto_load_scan(self, scan_sample):
        defect_file = scan_sample['defect_file']
        image_location = scan_sample['image_location']
        volume_file = scan_sample['volume_location']
        volume_mask_file = scan_sample['mask_location']
        scan_usage = scan_sample['scan_usage']
        scan_location = scan_sample['scan_location']

        # add data obtained into new directory, else create empty file
        if defect_file != "":
            new_fault_txt = 'fault.txt'
            new_fault_path = scan_location + '/' + new_fault_txt
            if defect_file != new_fault_path:
                shutil.copy(defect_file, new_fault_path)
            scan_sample['defect_file'] = new_fault_path
        else:
            scan_sample['defect_file'] = ""
            
        if image_location != "":
            new_img_folder = 'images'
            new_img_path = scan_location + '/' + new_img_folder
            if image_location != new_img_path:
                shutil.copytree(image_location, new_img_path)
            scan_sample['image_location'] = new_img_path
        else:
            scan_sample['image_location'] = ""

        if volume_file != "":
            ext = volume_file.split('.')[-1]
            new_vol_file = 'volume.' + ext
            new_vol_path = scan_location + '/' + new_vol_file
            if volume_file != new_vol_path:
                shutil.copy2(volume_file, new_vol_path)
            scan_sample['volume_location'] = new_vol_path
        else:
            scan_sample['volume_location'] = ""

        if volume_mask_file != "":
            ext = volume_mask_file.split('.')[-1]
            new_vol_file = 'volume_mask.' + ext
            new_vol_path = scan_location + '/' + new_vol_file
            if volume_mask_file != new_vol_path:
                shutil.copy2(volume_mask_file, new_vol_path)
            scan_sample['mask_location'] = new_vol_path
        else:
            scan_sample['mask_location'] = ""

        return scan_sample 
     
    def SQL_add_new_scan(self, scan_sample):
        tool_name = scan_sample['tool_name']
        scan_name = scan_sample['scan_name']
        scan_usage = scan_sample['scan_usage']
        defect = scan_sample['defect']
        scan_type = scan_sample['scan_type']
        scan_full_location = scan_sample['data_directory']
        image_location = scan_sample['image_location']
        volume_location = scan_sample['volume_location']

        print('save the new scan to MySQL')
        scan_part_id = 0
        scan_location = self.root_folder + tool_name + '/' + scan_name
        
        if defect == "defective-unacceptable":
            bad_scan = "Scan has fault that is unacceptable, hence it cannot be used in the reactor!\n Plese make sure you fix the tool before the next usage."
            bad_scan_title = "Scan with Unacceptable Fault for the reactor"
            self.showdialog_warn_only(bad_scan, bad_scan_title)
        
        if scan_name == "":
            no_scan = "Please enter a name for the new scan data!"
            no_scan_title = "No Scan Name Provided"
            self.showdialog_info_only(no_scan, no_scan_title)
            return

        if os.path.exists(scan_location):
            scan_exist_msg = "Scan already exists in directory! \nPlease manually delete the scan or change scan name during upload."
            scan_exist_title = "Scan Exists Already"
            self.showdialog_warn_only(scan_exist_msg, scan_exist_title)
            return
        else:
            os.mkdir(scan_location)

        scan_sample['scan_location'] = scan_location
        scan_sample['scan_part_id'] = scan_part_id
            
        if scan_type == "Single Part":
            # load data automatically from the scan folder
            update_scan_sample = self.auto_load_scan(scan_sample)
            sqlcon.sql_add_new_scan(update_scan_sample)
            print("New single scan " +  update_scan_sample['scan_name'] + " added to File Structure and Database...")

            new_scan_added = "New single part scan " + "\"" + str(scan_name) + "\"" + " has been added under tool " + str(tool_name) + "\""
            new_scan_added_title = "Adding New Single Part Scan"
            
        else:
            scan_name_list = [x for x in os.listdir(scan_full_location) if os.path.isdir(os.path.join(scan_full_location, x)) == True]
            scan_path_list = [os.path.join(scan_full_location, x) for x in os.listdir(scan_full_location) if os.path.isdir(os.path.join(scan_full_location, x)) == True]

            # repeat for different scan data requirements - images, volume, meta data, etc.
            scan_vol_list = [[os.path.join(x, f) if f.endswith(('nrrd', 'rek')) else "" for f in os.listdir(x)][0] for x in scan_path_list]
            scan_img_list = [[os.path.join(x, f) if os.path.isdir(os.path.join(x, f)) else "" for f in os.listdir(x)][0] for x in scan_path_list]
            scan_txt_list = [[os.path.join(x, f) if f.endswith('txt') else "" for f in os.listdir(x)][0] for x in scan_path_list]

            if volume_location == "":
                scan_vol = np.concatenate([self.get_volume(x)[0] for x in scan_vol_list], axis=1)

                # should new volume also be written to original data directory?
                # written to original data directory
                volume_location = os.path.join(scan_full_location, 'volume.nrrd')
                # written to new data location directly
                #volume_location = os.path.join(scan_location, 'volume.nrrd')
                
                nrrd.write(volume_location, scan_vol)
                scan_sample['volume_location'] = volume_location
                print("Updated multiple parts volume:", scan_vol.shape)
                       
            # load data automatically from the scan folder
            update_scan_sample = self.auto_load_scan(scan_sample)
            sqlcon.sql_add_new_scan(update_scan_sample)
            print("New multiple scan " +  update_scan_sample['scan_name'] + " added to File Structure and Database...")

            for sc in range(len(scan_name_list)):
                new_scan_sample = scan_sample.copy()
                #new_scan_sample['scan_name'] = scan_name_list[sc]
                new_scan_sample['scan_location'] = os.path.join(scan_location, scan_name_list[sc])
                new_scan_sample['volume_location'] = scan_vol_list[sc]
                new_scan_sample['image_location'] = scan_img_list[sc]
                new_scan_sample['defect_file'] = scan_txt_list[sc]
                new_scan_sample['mask_location'] = ""
                new_scan_sample['scan_part_id'] = sc + 1

                if os.path.exists(new_scan_sample['scan_location']) == False:
                    os.mkdir(new_scan_sample['scan_location'])
                else:
                    print("Existing scan part folder, potential conflict!")
                    
                # load data automatically from the scan folder
                update_scan_sample = self.auto_load_scan(new_scan_sample)
                sqlcon.sql_add_new_scan(update_scan_sample)
                print(update_scan_sample)
                scan_inner_name = scan_name_list[sc]
                new_scan_part = "New multiple parts scan " + "\"" + str(scan_inner_name) + "\"" + " has been added under scan " + "\"" + str(scan_name) + "\""
                new_scan_added = new_scan_part + " under tool " + "\"" + str(tool_name) + "\""
                print(new_scan_added)

            print("New multiple scan part " +  new_scan_sample['scan_name'] + " added to File Structure and Database...")
            new_scan_added = "New multiple parts scan " + "\"" + str(scan_name) + "\"" + " has been added under tool " + str(tool_name) + "\""
            new_scan_added_title = "Adding New Scan"

        sql_new_scan_data = sqlcon.get_all_scan_table(tool_name)
        #self.table = self.update_table(self.table, sql_new_scan_data)
        self.update_table(self.table, sql_new_scan_data)
        self.showdialog_info_only(new_scan_added, new_scan_added_title)
        

    def SQL_update_tool(self, tool_sample):     

        tool_name = tool_sample['tool_name']
        tool_location = tool_sample['tool_location']
        cad_path = tool_sample['cad_path']
        cad_available = tool_sample['cad_available']
        names_components = tool_sample['names_components']
        previous_tool_row = sqlcon.get_tool_row(tool_name)
        #print(tool_sample)
        
        existing_stl = tool_location + "/STL"
        txt_filename = tool_location + "/names_components.txt"
        
        if cad_available == False:
            if 'STL' in os.listdir(tool_location):
                print("Previous available CAD has been removed!")
                shutil.rmtree(existing_stl)
            if 'names_components.txt' in os.listdir(tool_location):
                os.remove('names_components.txt')

            tool_sample['cad_path'] = ""
            tool_sample['names_components'] = ""
            tool_sample['numbers_components'] = 0
        else:
            tool_sample['cad_path'] = existing_stl
            tool_sample['names_components'] = txt_filename
                
        if cad_path != previous_tool_row['cad_path']:
            if previous_tool_row['cad_path'] != "":
                print("Previous available CAD has been changed!")
                shutil.rmtree(existing_stl)
            else:
                print("Newly added CAD for this tool!")
            shutil.copytree(cad_path, existing_stl)
            print("Change detected in CAD path:")
            print(cad_path, existing_stl)
            
            # save component names txt file 
            print('names_components = ', names_components)
            Autils.write_string_list(txt_filename, names_components)
        
        sqlcon.sql_update_tool(tool_sample)
        update_tool_added = "Existing tool " + "\"" + str(tool_name) + "\"" + " has been updated!"
        update_tool_added_title = "Updated Existing Tool"
        self.showdialog_info_only(update_tool_added, update_tool_added_title)
        self.show_scans_data("")

    def SQL_update_scan(self, scan_sample):
        
        tool_name = scan_sample['tool_name']
        scan_name = scan_sample['scan_name']
        defect_file = scan_sample['defect_file']
        image_location = scan_sample['image_location']
        volume_file = scan_sample['volume_location']
        volume_mask_file = scan_sample['mask_location']
        
        previous_scan_row = sqlcon.get_scan_row(scan_name)
        scan_sample['scan_part_id'] = previous_scan_row['scan_part_id']
            
        scan_location = self.root_folder + tool_name + '/' + scan_name
        scan_sample['scan_location'] = scan_location
        print("Updated data added to scan location:", scan_location)
        #os.mkdir(scan_location)

        # add data obtained into new directory, else create empty file
        new_fault_txt = 'fault.txt'
        new_fault_path = scan_location + '/' + new_fault_txt
        if new_fault_path == previous_scan_row['defect_file_location']:
            scan_sample['defect_file'] = new_fault_path
        else:
            if '.' not in defect_file:
                scan_sample['defect_file'] = new_fault_path
            else:
                if os.path.isfile(new_fault_path):
                    os.remove(new_fault_path)
                if defect_file == "":
                    scan_sample['defect_file'] = ""
                else:
                    shutil.copy2(defect_file, new_fault_path)
                    scan_sample['defect_file'] = new_fault_path
                    print("Change detected in fault txt:")
                    print(defect_file, new_fault_path)

        new_img_folder = 'images'
        new_img_path = scan_location + '/' + new_img_folder
        if image_location != previous_scan_row['image_location']:
            if previous_scan_row['image_location'] != "" and os.path.isdir(previous_scan_row['image_location']):
                shutil.rmtree(previous_scan_row['image_location'])
            if image_location == "":
                scan_sample['image_location'] = ""
            else:
                shutil.copytree(image_location, new_img_path)
                scan_sample['image_location'] = new_img_path
            print("Change detected in image folder:")
            print(image_location, new_img_path)
        
        new_vol_file = 'volume.rek'
        new_vol_path = scan_location + '/' + new_vol_file
        if volume_file != previous_scan_row['volume_location']:
            if previous_scan_row['volume_location'] != "" and os.path.isfile(previous_scan_row['volume_location']):
                os.remove(previous_scan_row['volume_location'])
            if volume_file == "":
                scan_sample['volume_location'] = ""
            else:
                shutil.copy2(volume_file, new_vol_path)
                scan_sample['volume_location'] = new_vol_path
            print("Change detected in volume file:")
            print(volume_file, new_vol_path)

        volume_ext = volume_mask_file.split('.')[-1]
        new_vol_mask_file = 'volume_mask' + volume_ext
        new_vol_mask_path = scan_location + '/' + new_vol_mask_file
        if volume_mask_file != previous_scan_row['mask_location']:
            if previous_scan_row['mask_location'] != "" and os.path.isfile(previous_scan_row['mask_location']):
                os.remove(previous_scan_row['mask_location'])
            if volume_mask_file == "":
                scan_sample['mask_location'] = ""
            else:
                shutil.copy2(volume_mask_file, new_vol_mask_path)
                scan_sample['mask_location'] = new_vol_path
            print("Change detected in mask volume file:")
            print(volume_mask_file, new_vol_mask_path)

        sqlcon.sql_update_scan(scan_sample)
        print("Select scan updated in File Structure and Database...")
        sql_update_scan_data = sqlcon.get_all_scan_table(tool_name)
        
        update_scan_added = "Data in scan " + "\"" + str(scan_name) + "\"" + " has been updated under tool " + str(tool_name)
        update_scan_added_title = "Updating Existing Scan"
        self.showdialog_info_only(update_scan_added, update_scan_added_title)
        #self.table = self.update_table(self.table, sql_new_scan_data)

        print('save the updated scan to SQL')
        #self.update_table(self.table, sql_update_scan_data)
        self.show_scans_data("")

    ## Show/Hide windows
    def visualize_dataset(self):
        print('visualize the data stored in : \n', self.file_path)
        Dataset_Win.show()
        Algorithms_Win.hide()
        main_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()

    def Fault_prediction(self):
        print('Run faut prediction algorithms  \n' )
        prediction_Win.show()
        prediction_Win.update_tool_list()
        Algorithms_Win.hide()
        main_Win.hide()
        Dataset_Win.hide()
        AutoInspection_Win.hide()
    
    def Registration(self):
        print('Run registration  algorithms\n' )
        Algorithms_Win.show()
        Algorithms_Win.model_path_list.show()
        Algorithms_Win.ui.threshold_slider.hide()
        Algorithms_Win.ui.Amin_label.hide()
        

        Algorithms_Win.current_algorithm = -1
        Algorithms_Win.load_vars()

        main_Win.hide()
        Dataset_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()

    def Fault_localization(self):
        print('Run faut localization algorithms\n' )
        Algorithms_Win.show()
        Algorithms_Win.update_tool_list()
        Algorithms_Win.model_path_list.show()
        Algorithms_Win.ui.threshold_slider.show()
        Algorithms_Win.ui.Amin_label.show()
        Algorithms_Win.ui.verify_save.hide()

        #Algorithms_Win.current_algorithm = 2
        #Algorithms_Win.load_vars()

        main_Win.hide()
        Dataset_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()

    def Automated_framework(self):
        a=0
        print('Run automated frameworkalgorithms\n' )
        Algorithms_Win.hide()
        AutoInspection_Win.show()
        AutoInspection_Win.update_tool_list()
        # AutoInspection_Win.load_vars()
        main_Win.hide()
        Dataset_Win.hide()
        prediction_Win.hide()


    def Fault_characterization(self):
        print('Run faut characterization algorithms\n' )
        Algorithms_Win.show()
        # Algorithms_Win.model_path_list.hide()
        Algorithms_Win.ui.threshold_slider.hide()
        Algorithms_Win.ui.Amin_label.hide()
        Algorithms_Win.ui.verify_save.hide()


        Algorithms_Win.current_algorithm = 3
        Algorithms_Win.load_vars()
        main_Win.hide()
        Dataset_Win.hide()
        prediction_Win.hide()
        AutoInspection_Win.hide()

    def close_vtk_widget(self, vtk_widget):
        if not vtk_widget == None:
            ren = vtk_widget.interactor.GetRenderWindow()
            iren = ren.GetInteractor()
            ren.Finalize()
            iren.TerminateApp()

    def closeEvent(self, e):
        print('Closing the windows')
        self.close_vtk_widget(self.vtk_widget_1)
        self.close_vtk_widget(self.vtk_widget_2)
        self.close_vtk_widget(self.vtk_widget_3)
        

## 3D Viewer:
class QGlyphViewer(QFrame):
    arrow_picked = pyqtSignal(float)

    def __init__(self):
        super(QGlyphViewer,self).__init__()
        # Make tha actual QtWidget a child so that it can be re parented
        interactor = QVTKRenderWindowInteractor(self)
        layout = QHBoxLayout()
        layout.addWidget(interactor)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        self.text=None
       
        # Setup VTK environment
        renderer = vtk.vtkRenderer()
        render_window = interactor.GetRenderWindow()
        render_window.AddRenderer(renderer)

        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        render_window.SetInteractor(interactor)
        renderer.SetBackground(0.2,0.2,0.2)

        # set Renderer
        self.renderer = renderer
        self.interactor = interactor
        self.render_window = render_window
        # self.picker = vtk.vtkCellPicker()
        # self.picker.AddObserver("EndPickEvent", self.process_pick)
        # self.interactor.SetPicker(self.picker)

    def start(self):
        self.interactor.Initialize()
        self.interactor.Start()
        # self.interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.click_to_pick, 10)

    def volume_properties_setup(self, volume, volume_segments=0, max_segment=300):
        print('len(volume_segments) = ', len(volume_segments))
        # volume_segments = np.linspace(volume_segments.min(), volume_segments.max(), num=max_segment)
        if len(volume_segments)>1 and len(volume_segments)<=max_segment:
            #volume property
            volume_property = vtk.vtkVolumeProperty()
            volume_color = vtk.vtkColorTransferFunction()

            # The opacity 
            volume_scalar_opacity = vtk.vtkPiecewiseFunction()
            # The gradient opacity 
            volume_gradient_opacity = vtk.vtkPiecewiseFunction()
            volume_color = vtk.vtkColorTransferFunction()
            for i in volume_segments:
                if i==0:
                    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
                else:
                    color_i = tuple(np.random.randint(255, size=3))
                    volume_color.AddRGBPoint(i, color_i[0]/255.0, color_i[1]/255.0, color_i[2]/255.0)
                    # volume_scalar_opacity.AddPoint(i, 0.15)
                    # volume_gradient_opacity.AddPoint(0i, 0.5)
            volume_property.SetColor(volume_color)
            # volume_property.SetScalarOpacity(volume_scalar_opacity)
            # volume_property.SetGradientOpacity(volume_gradient_opacity)

            volume_property.ShadeOn()
            volume_property.SetAmbient(0.4)
            # volume_property.SetDiffuse(0.6)
            # volume_property.SetSpecular(0.2)

            # # setup the properties
            volume.SetProperty(volume_property)

        return volume

    def update_volume(self, volume, volume_segments):
        
        # assign disply propoerties
        volume = self.volume_properties_setup(volume, volume_segments)
        # Finally, add the volume to the renderer
        self.renderer.RemoveAllViewProps()
        # self.renderer.AddViewProp(volume)
        self.renderer.AddVolume(volume)

        # setup the camera
        camera = self.renderer.GetActiveCamera()
        c = volume.GetCenter()
        camera.SetViewUp(0, 0, -1)
        camera.SetPosition(c[0], c[1]-1000, c[2])
        camera.SetFocalPoint(c[0], c[1], c[2])
        camera.Azimuth(0.0)
        camera.Elevation(90.0)

        # # Set a background color for the renderer
        # colors = vtk.vtkNamedColors()
        # colors.SetColor('BkgColor', [51, 77, 102, 255])
        # self.renderer.SetBackground(colors.GetColor3d('BkgColor'))

        # message = ' Volume loaded successfully!!'
        # self.showdialog_information(message)

        

    def disp_fault_volume(self, volume,fault):
        # self.renderer.RemoveAllViewProps()
        # self.renderer.AddVolume(volume)
        ##-------------------------------------------------------

        # # This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
        # # to be of the colors red green and blue.
        # colorFunc = vtk.vtkColorTransferFunction()
        # colorFunc.AddRGBPoint(5, 1, 0.0, 0.0)  # Red

        # # The previous two classes stored properties. Because we want to apply these properties to the volume we want to render,
        # # we have to store them in a class that stores volume properties.
        # volumeProperty = vtk.vtkVolumeProperty()
        # volumeProperty.SetColor(colorFunc)
        # # color the faults
        # fault.SetProperty(volumeProperty)
        self.renderer.AddVolume(fault)

    def showdialog_information(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Information message")
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()

    def showdialog_warnning(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Error message")
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval

    '''
    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        self.interactor.Finalize()
    '''


## Preprocessing
class DatasetApp(MainApp):
    def __init__(self):
        #Parent constructor
        super(DatasetApp,self).__init__()
        self.ui = None
        self.path = self.root_folder
        self.setup()
        self.msg ='\n\n - Please click on the appropriate CT image/volume to be visualized or text to be edited . '
        self.ui.msg_label.setText(self.msg)
        self.filename =  None
        self.file_before = 'files/reference.jpg'
        self.createMenu()
        self.windows_format()
        self.plot_sample_image()


    def setup(self):
        
        import UI.VisualizationUI 
        self.ui = UI.VisualizationUI.Ui_Run_Visualization()
        self.ui.setupUi(self)
        self.plot_1 = self.ui.Label_plot_1
        self.edit_1 = self.ui.plainTextEdit_1
        self.save_edit_1 = self.ui.pushButton_save_1
        self.save_edit_1.clicked.connect(self.save_txt)
        self.add_scan = self.ui.add_scan
        self.add_scan.clicked.connect(self.add_new_scan)

        # drop box of tools
        self.list_tools = self.ui.comboBox
        print("List of tools:", sqlcon.get_all_tool())
        self.list_tool_names = [""]
        self.list_tool_names.extend(sqlcon.get_all_tool())
        self.list_tools.addItems(self.list_tool_names)
        self.list_tools.currentIndexChanged.connect(self.show_scans_data)
                                
        # table visual
        self.table = self.ui.SQL_table
        #list_scans = sqlcon.get_all_scan()
        # self.table = self.update_table(self.table, list_scans)
        
        # list tree
        self.path = self.root_folder + self.list_tools.currentText()
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        
        # VTK rendrer 1
        self.vtk_plot_1 = self.ui.Frame_plot_1
        self.vtk_widget_1 = None
        path0 ='files/volume_after.rek'
        self.vtk_widget_1, self.vtk_plot_1 = self.create_vtk_widget(self.vtk_widget_1, self.vtk_plot_1, path0)

    def plot_sample_image(self):
        self.Plot_image(self.file_before, self.plot_1)

    def upload_image(self):           
        self.Plot_image(self.file_before, self.plot_1)
 
    def save_txt(self):
        with open(self.file_before, "w") as f:
            f.write(self.edit_1.toPlainText())
        self.edit_1.document().setModified(False)

    def change_name(self, index):
        """ rename """
        if not index.isValid():
            return

        model = index.model()
        old_name = model.fileName(index)
        path = model.fileInfo(index).absoluteFilePath()

        # ask new name
        name, ok = QInputDialog.getText(self, "New Name", "Enter a name", QLineEdit.Normal, old_name)
        if not ok or not name:
            return
        
        # rename
        model = index.model()
        wasReadOnly = model.isReadOnly()
        model.setReadOnly(False)
        model.setData(index, name)
        model.setReadOnly(wasReadOnly)

    def show_scans_data(self, val):

        tool_name = self.list_tools.currentText()
        #self.treeView.setRootIndex(self.treeModel.index(self.root_folder + tool_name))
        self.path = self.root_folder + tool_name
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        #self.treeView.setRootIndex(self.treeModel.index(self.path))

        # SQL load details on list of tools
        if tool_name == "":
            self.table.clear()
            return
        else:
            # SQL load details on list of scans of select tool
            table_scans = sqlcon.get_all_scan_table(tool_name)

        #self.table = self.update_table(self.table, table_scans)
        self.update_table(self.table, table_scans)

        
    def get_selected_scan(self, index):
        """ for test """
        if not index.isValid():
            return
        model = index.model()
        raw = model.fileInfo(index).absoluteFilePath()

        msg = 'Uploading in process .... \n file = ' + raw
        print(msg);self.ui.msg_label.setText(msg)

        self.treeModel.setRootPath(self.path)

    def onStateChanged(self):
        ch = self.sender()
        print(ch.parent())
        ix = self.table.indexAt(ch.pos())
        print(ix.row(), ix.column(), ch.isChecked())


## RUN CT inspection algorithms  
class VerifyRegistration(MainApp):
    def __init__(self):
        #Parent constructor
        super(VerifyRegistration,self).__init__()
        self.config_algo = 0
        self.ui = None
        self.setup()
        self.filename =  None
        self.createMenu()
        self.windows_format()
        self.scan_sample = None
        ref_vol_array, _ = self.get_volume('xx')
        self.ref_vol_array = ref_vol_array
        vol_array, _ = self.get_volume('xx')
        self.vol_array = vol_array
        self.update_flag = 0

        self.plot_random_slice()

    def setup(self):
        
        import UI.Compare_registrationUI 
        self.ui = UI.Compare_registrationUI.Ui_Run_compare_reg()
        self.ui.setupUi(self)

        self.file_before = 'files/img_before.tif'
        self.file_after = 'files/img_after.tif'

        self.plot_1 = self.ui.Label_plot_1
        self.plot_2 = self.ui.Label_plot_2

        self.vtk_plot_1 = self.ui.Frame_plot_1
        self.vtk_plot_2 = self.ui.Frame_plot_2

        # Initialize the 3D plots
        path0 ='data/demo_data/reg.nrrd'
        self.vtk_widget_1, self.vtk_plot_1 = self.create_vtk_widget(self.vtk_widget_1, self.vtk_plot_1, path0)
        self.plot_3D_volume(path0, self.vtk_widget_1)
        path1 ='data/demo_data/vol_reg.nrrd'
        self.vtk_widget_2, self.vtk_plot_2 = self.create_vtk_widget(self.vtk_widget_2, self.vtk_plot_2, path1)
        self.plot_3D_volume(path1, self.vtk_widget_2)

        # Show 2D images first
        self.plot_1.show();self.plot_2.show()
        self.vtk_plot_1.show();self.vtk_plot_2.show()
   
        self.ui.accept_results.clicked.connect(self.accept_registration)
        self.ui.reject_results.clicked.connect(self.reject_registration)
        self.ui.get_new_slice.clicked.connect(self.plot_random_slice)    
        self.ui.go_back.clicked.connect(self.go_back_windows) 

    def plot_3D_volume(self, path, vtk_widget):
        # display the selected Rek volume 
        volume_arr, vol0  = self.get_volume(path)
        volume_segments = np.array(np.unique(volume_arr.ravel()))
        self.Plot_volume(vtk_widget , vol0, volume_segments=volume_segments)
        print(' vizualized volume path: ', path)
         
    def Plot_volume(self, vtk_widget, volume, volume_segments=[0]):
        # update the plot
        vtk_widget.update_volume(volume, volume_segments=volume_segments)
        vtk_widget.start()

    def save_array_image(self, img_array, file_path = "output/image.jpg"):
        img = Image.fromarray(img_array);  
        img.mode = 'I'; img.point(lambda i:i*(1./256)).convert('L').save(file_path) 
    

    def upload_new_volume(self, scan_sample, update_flag):
        self.show(); Dataset_Win.hide()
        self.update_flag = update_flag
        self.scan_sample = scan_sample

        # update register the 3D volume from: 
        vol_path = scan_sample['volume_location']
        print('vol_path from scan sample: ' , vol_path)

        
        # update get the refetence volume 
        list_scans_df = sqlcon.get_all_scan_table(scan_sample['tool_name'])
        print(list_scans_df.size)
        dst_nrrd = ""
        
        if list_scans_df.size>0:
            # ref vol is present
            select_scan = 0
            scan_sample_ref = list_scans_df.iloc[select_scan]
            ref_scan_folder = list_scans_df['scan_location'].iloc[select_scan]
            ref_vol_path = self.get_supported_volume(ref_scan_folder)
            # which folder to choose from the list. defaut 0
            if len(ref_vol_path)>=1:
                ref_vol_path = ref_vol_path[0]
                
            print('reference volume : ' , ref_vol_path)
            ref_vol_array, _ = self.get_volume(ref_vol_path)
            self.ref_vol_array = ref_vol_array
            vol_array, _ = self.get_volume(vol_path)
            self.vol_array = vol_array


            # apply the registration 
            ref_vol_reg, vol_reg, save_ref_volume = self.apply_volumes_registration_flowchart(self.ref_vol_array, self.vol_array) 

            splt = len(vol_path.split('.')[-1])
            dst_nrrd = vol_path [:-splt-1] + '_reg.' + vol_path[-splt:]
            import nrrd
            nrrd.write(dst_nrrd, vol_reg)
            self.scan_sample['volume_location'] = dst_nrrd

            # PLot the 3D volumes
            self.plot_3D_volume(vol_path, self.vtk_widget_1)
            self.plot_3D_volume(dst_nrrd, self.vtk_widget_2)

            # start registratio inspection by slices
            self.plot_random_slice()
        
        else:
            #input itself is ref vol
            # need to handel self registration and pre_proc
            
            mask_available = sqlcon.get_cad_avail_flag(scan_sample['tool_name'])

            if mask_available:
                
                # get mask path from DB for tool
                #mask_path='data/Tool2/STL/Volume_label.nrrd'
                mask_path = sqlcon.get_tool_volume_location(scan_sample['tool_name'])

                ref_vol_array, _ = self.get_volume(mask_path)
                self.ref_vol_array = ref_vol_array
                vol_array, _ = self.get_volume(vol_path)
                self.vol_array = vol_array

                # apply the registration 
                ref_vol_reg, vol_reg, save_ref_volume = self.apply_volumes_registration_flowchart(self.ref_vol_array, self.vol_array) 

                # save registred volume to by co
                splt = len(vol_path.split('.')[-1])
                dst_nrrd = vol_path [:-splt-1] + '_reg.' + vol_path[-splt:]
                import nrrd
                nrrd.write(dst_nrrd, vol_reg)
                self.scan_sample['volume_location'] = dst_nrrd

                # start registratio inspection by slices
                self.plot_random_slice()                

            else:
                #Dataset_Win.SQL_add_new_scan(self.scan_sample)
                self.hide(); Dataset_Win.show()

        return dst_nrrd

    def apply_volumes_registration_flowchart(self, ref_vol_array, vol_array, isref=False):
        # vol_reg = vol_array
        # ref_vol_reg = ref_vol_array
        save_ref_volume = isref

        #resize ratio
        res_ratio=ref_vol_array.shape[1]/vol_array.shape[1]

        ref_vol_reg, vol_reg, estimated_angle= Jutils_3D_pre_proc.pre_proc_3d(ref_vol_array,vol_array,bg_remove_mode=2,resize_ratio=res_ratio,crop_point=0, pre_proc_ref=False, angle_step=5)

        self.vol_array=vol_reg
        return ref_vol_reg, vol_reg, save_ref_volume

    def plot_random_slice(self):
        self.ui.get_new_slice.hide()

        msg = '\n - Reference volume =  '  + str(self.ref_vol_array.shape) + '\n - Input volume =  '  + str(self.vol_array.shape) 
        print(msg)

        if self.ref_vol_array.shape != self.vol_array.shape:
            self.close_verification_error(msg)
        else:    

            import random
            idx = random.randint(0,self.vol_array.shape[1]-1)
            file_path = "output/image.jpg"
            self.save_array_image(self.ref_vol_array[:,idx,:], file_path) 
            # file_path =  'files/img_before.tif'
            self.Plot_image(file_path, self.plot_1)

            file_path = "output/image_reg.jpg"
            self.save_array_image(self.vol_array[:,idx,:], file_path) 
            # file_path =  'files/img_before.tif'
            self.Plot_image(file_path , self.plot_2)
            print('plotted a new slice')

        self.ui.get_new_slice.show()

    def accept_registration(self):

        self.showdialog_information('The registration is accepted. \nThe registred data will saved in the database.')
        '''
        if self.update_flag ==0:
            Dataset_Win.SQL_add_new_scan(self.scan_sample)

        else:
            Dataset_Win.SQL_update_scan(self.scan_sample)
        '''

        self.hide(); Dataset_Win.show()

    def reject_registration(self):
        self.showdialog_information('The registration was not accepted. \nThe Process will be repeated with other parameters. \
                                     \n The process will take few minutes. Please wait :)')
        self.upload_new_volume(self.scan_sample, self.update_flag)

    def close_verification_error(self, msg):
        self.hide(); Dataset_Win.show()
        self.showdialog_warnning('The selected scans does not have the same size: \n' + msg + \
                                  '\nPlease recheck the data or upload it as a new scan in a new tool!!')

    def go_back_windows(self):
        self.hide(); Dataset_Win.show()


## DIALOG Windows 
class Dialog_ToolForm(QDialog):

    def __init__(self, root_folder, update_flag=0):
        super(Dialog_ToolForm, self).__init__()
        self.dict_tool= None
        #self.root_folder = root_folder
        self.update_flag = update_flag
        # the form attributs.
        self.tool_name = QLineEdit()
        self.tool_location = QLineEdit()
        self.designer_name = QLineEdit()
        self.cad_path =  QComboBox()
        self.cad_path = QPushButton('Click')# browse file
        self.cad_path.clicked.connect(self.get_CAD_folder)
        #self.names_components = QLineEdit()
        #self.numbers_components = QLineEdit()
        
        # setting up fields in case of tool update
        if update_flag == 1:
            self.root_folder = Dataset_Win.root_folder
            tool_name = Dataset_Win.list_tools.currentText()
            tool_row = sqlcon.get_tool_row(tool_name)
            #print(tool_row)
            self.tool_name.setText(tool_row['tool_name'])
            self.tool_name.setReadOnly(True) # set uneditable text
            self.designer_name.setText(tool_row['designer_name'])
            self.cad_path.setText(tool_row['cad_path'] if tool_row['cad_path'] != "" else "Click")
            #self.names_components.setText(tool_row['names_components'])
            #self.numbers_components.setText(tool_row['numbers_components'])
            
        else:
            self.root_folder = root_folder
            
        # create the form .
        self.createFormGroupBox()
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.get_form_data)
        buttonBox.rejected.connect(self.cancel_saving)
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)
        if self.update_flag == 0:
            self.setWindowTitle("Add New tool")
        else:
            self.setWindowTitle("Update Existing tool")

    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("Tool form")
        layout = QFormLayout()
        layout.addRow(QLabel("Tool name:"), self.tool_name)
        layout.addRow(QLabel("Designed by:"), self.designer_name)
        layout.addRow(QLabel("CAD directory:"), self.cad_path)
        #layout.addRow(QLabel("Names of CAD Components:"), self.names_components)
        #layout.addRow(QLabel("Number of CAD Components:"), self.numbers_components)
        self.formGroupBox.setLayout(layout)

    def get_CAD_folder(self):
        # open select folder dialog
        fname = QFileDialog.getExistingDirectory(self, 'Select the CAD/STL directory', '~')
        print(fname)

        if os.path.isdir(fname):
            self.cad_path.setText(fname)
            message = 'The CAD folder  is selected successfully! '
            print(message)
            #self.showdialog_information(message)

        else: 
            message = 'You did not select a folder. Please seelect folder where the CAD/STL Files are saved '
            print(message)
            #self.showdialog_warnning(message)

    def get_form_data(self):
        tool_name = self.tool_name.text() 
        tool_location = self.root_folder + tool_name
        designer_name = self.designer_name.text()
        cad_path = self.get_path_button(self.cad_path)
        #names_components = self.names_components.text()
        #numbers_components = self.numbers_components.text()

        if self.update_flag == 0:
            tool_name_list = sqlcon.get_all_tool()
            if tool_name in tool_name_list:
                mssg = "Tool Name already exists, please use another name!"
                self.showdialog_warnning(mssg)
                return
        
        print('save the new tool in MySQL')
        print('tool_name:',tool_name)
        print('designer_name:',designer_name)
        print('cad_path:',cad_path)
        #print('names_components:', names_components)
        #print('numbers_components:', numbers_components)
        self.close()

        if cad_path == '':
            cad_available = False 
            numbers_components = 0
            names_components = ''

        else:
            cad_available = True 
            # additional fields:
            from glob import glob
            STL_filepaths = glob(cad_path + '/*.STL')
            names_components = [os.path.basename(x) for x in STL_filepaths]
            numbers_components = len(names_components)
 
        self.dict_tool = {
            'tool_location':tool_location, 'tool_name':tool_name,
            'designer_name':designer_name, 'cad_available':cad_available,
            'cad_path':cad_path, 'names_components':names_components, 'numbers_components':numbers_components
        }

        if self.update_flag == 0:
            # save new data to SQL 
            Dataset_Win.SQL_add_new_tool(self.dict_tool)
        else:
            # update existing data in SQL
            Dataset_Win.SQL_update_tool(self.dict_tool)        

    def cancel_saving(self):

        print('cancel the saving of the new tool in MySQL')
        self.close()
       
    def showdialog_information(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Information message")
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval
        
    def showdialog_warnning(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Error message")
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval
        
    def get_path_button(self, var_name):
        path = var_name.text()
        if path == 'Click':
            path = ''
        return path

class Dialog_ScanForm(QDialog):

    def __init__(self,  root_folder, scan_name="", user='', update_flag=0):
        super(Dialog_ScanForm, self).__init__()
        self.user = user
        self.dict_tool= None
        self.root_folder = root_folder
        self.file_path = ""
        self.update_flag = update_flag
        
        # the form attributs.
        self.tool_name = os.path.basename(root_folder)
        self.scan_name = QLineEdit()
        #self.scan_part_id = 0
        self.before_after_status = QCheckBox()
        self.data_verification = QCheckBox()
        self.data_verification.toggled.connect(self.Expert_verification_options)

        self.data_directory = QPushButton('Click')# browse full data folder
        self.data_directory.clicked.connect(self.get_data_directory)
        
        self.defect = QPushButton('Click')# browse file
        self.defect.clicked.connect(self.get_defect_file)

        self.image_folder = QPushButton('Click')# browse file
        self.image_folder.clicked.connect(self.get_image_folder)

        self.volume_file = QPushButton('Click')# browse file
        self.volume_file.clicked.connect(self.get_volume_file)

        self.volume_mask_file = QPushButton('Click')# browse file
        self.volume_mask_file.clicked.connect(self.get_volume_mask_file)

        self.object_material =  QLineEdit()
        self.filter_type = QLineEdit()
        self.filter_size = QLineEdit()
        self.technician_name = QLineEdit()
        self.technician_name.setText(self.user)
        self.scan_data_label = QComboBox()
        
        # setting up fields in case of tool update
        if self.update_flag == 1:
            scan_row = sqlcon.get_scan_row(scan_name)
            self.scan_name.setText(scan_row['scan_name'])
            self.scan_name.setReadOnly(True) # set uneditable text
            #self.scan_part_id.setValue(int(scan_row['scan_part_id']))
            self.before_after_status.setChecked(True if scan_row['before_after_status'] == 1 else False)
            self.data_verification.setChecked(True if scan_row['data_verification'] == 1 else False)

            self.defect.setText(scan_row['defect_file_location'] if scan_row['defect_file_location'] != '' else 'Click')
            self.image_folder.setText(scan_row['image_location'] if scan_row['image_location'] != '' else 'Click')
            self.volume_file.setText(scan_row['volume_location'] if scan_row['volume_location'] != '' else 'Click')
            self.volume_mask_file.setText(scan_row['mask_location'] if scan_row['mask_location'] != '' else 'Click')

            self.object_material.setText(scan_row['object_material'])
            self.filter_type.setText(scan_row['filter_type'])
            self.filter_size.setText(scan_row['filter_size'])
            self.technician_name.setText(scan_row['technician_name'])

            self.data_labels = [scan_row['defect']]
            self.scan_data_label.addItems(self.data_labels)
            
        else:
            self.root_folder = root_folder
            # fill optional fileds
            self.object_material.setText("lead")
            self.filter_type.setText("copper")
            self.filter_size.setText("2.0mm")
            self.data_labels = ["I don't know"]
            self.scan_data_label.addItems(self.data_labels)
            
        # create the form .
        self.createFormGroupBox()
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.get_form_data)
        buttonBox.rejected.connect(self.cancel_saving)
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)
        if self.update_flag == 0:
            self.setWindowTitle("Add new scan")
        else:
            self.setWindowTitle("Update existing scan")


    def Expert_verification_options(self):
        if not self.data_verification.isChecked():
            self.data_labels = ["I don't know"]
        else:
            self.data_labels = ["Good", "Faulty Acceptable", "Faulty Unacceptable", "I don't know"]
        self.scan_data_label.clear()
        self.scan_data_label.addItems(self.data_labels)

    def get_data_directory(self):
        # open select folder dialog
        fname = QFileDialog.getExistingDirectory(self, 'Select the folder containing all scan data', '~')
        print(fname)

        if os.path.isdir(fname):
            self.data_directory.setText(fname)
            message = 'The data folder  is selected successfully!'
            print(message)
            self.showdialog_information(message)
            scan_name = os.path.basename(fname)
            self.scan_name.setText(scan_name)
            print('the scan name is  :  ', scan_name)

        else: 
            message = 'You did not select a folder. Please select folder containing all required scan data'
            print(message)
            self.showdialog_warnning(message)
            
    def get_defect_file(self):
        filepath = QFileDialog.getOpenFileName(self, "Open defect file", "~", "Text Files (*.txt)")[0]
        print(filepath)
        self.defect.setText(filepath)

        message = 'The defect file is selected successfully!'
        print(message)
        self.showdialog_information(message)

    def get_image_folder(self):
        # open select folder dialog
        fname = QFileDialog.getExistingDirectory(self, 'Select the tif images directory', '~')

        print(fname)

        if os.path.isdir(fname):
            self.image_folder.setText(fname)
            message = 'The scan image folder  is selected successfully!'
            print(message)
            self.showdialog_information(message)

        else: 
            message = 'You did not select a folder. Please seelect folder where the tif images are saved '
            print(message)
            self.showdialog_warnning(message)

    def get_volume_file(self):
        filepath = QFileDialog.getOpenFileName(self, "Open 3D volume file", "~", "3D volume Files (*.rek *.nrrd *.nii *.bd)")[0]
        print(filepath)
        self.volume_file.setText(filepath)

        message = 'The 3D volume file is selected successfully!'
        print(message)
        self.showdialog_information(message)

    def get_volume_mask_file(self):
        filepath = QFileDialog.getOpenFileName(self, "Open 3D mask volume file", "~", "3D volume Files (*.rek *.nrrd *.nii *.bd)")[0]
        print(filepath)
        self.volume_mask_file.setText(filepath)

        message = 'The 3D volume mask file is selected successfully!'
        print(message)
        self.showdialog_information(message)

    def showdialog_information(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Information message")
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)

    def showdialog_warnning(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Error message")
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval

    def showdialog_question(self, message):
        reply = QMessageBox.question(self, 'Quit', message,
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply
        
    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("Scan form [tool = " + self.tool_name+ "]")
        layout = QFormLayout()   
        
        if self.update_flag == 0:
            layout.addRow(QLabel("Data directory:"), self.data_directory)

        else:
            layout.addRow(QLabel("Defect file:"), self.defect)
            layout.addRow(QLabel("Tif images folder:"), self.image_folder)
            layout.addRow(QLabel("Volume file:"), self.volume_file)
            layout.addRow(QLabel("Volume mask file:"), self.volume_mask_file)
        
        layout.addRow(QLabel("Was the scan verified by an expert?:"), self.data_verification)
        layout.addRow(QLabel("Was the scan before the tool usage?:"), self.before_after_status)
        layout.addRow(QLabel("Tool status:"), self.scan_data_label)

        layout.addRow(QLabel("Scan name:"), self.scan_name)       
        layout.addRow(QLabel("Object material type:"), self.object_material)
        layout.addRow(QLabel("Filter type:"), self.filter_type)
        layout.addRow(QLabel("Filter size:"), self.filter_size)
        layout.addRow(QLabel("Technician name:"), self.technician_name)
        self.formGroupBox.setLayout(layout)

    def get_path_button(self, var_name):

        path = var_name.text()

        if path == 'Click':
            path = ''

        return path

    def isImg(self, fname):
        img_ext = ['png', 'bmp', 'jpg', 'tif', 'tiff']
        if fname.split('.')[-1] in img_ext:
            return True
        else:
            return False
    
    def get_supported_volume(self, root, ext_list = ['.rek' , '.nrrd', '.nii']):
        from glob import glob 
        vol_list = []
        for ext in ext_list:
            vol_list = vol_list + glob( os.path.join(root, '*' + ext) )

        return vol_list

    def get_form_data(self):
        tool_name = self.tool_name
        scan_name = self.scan_name.text()
        before_after_status = self.before_after_status.isChecked()
        data_verification = self.data_verification.isChecked()
            
        if self.update_flag == 0:
            scan_data_label = self.scan_data_label.currentText()
            data_directory = self.get_path_button(self.data_directory)
            if data_directory in ["", "Click"]:
                msg = "Path to data directory has not been provided!!"
                Dataset_Win.showdialog_warnning(msg)
                return
            
            list_files = [os.path.join(data_directory, x) for x in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, x))]
            list_folders = [os.path.join(data_directory, x) for x in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, x)) and len(os.listdir(os.path.join(data_directory, x))) > 0]
            list_txt = [x for x in list_files if '.txt' in x]
            list_img = [x for x in list_folders if self.isImg(os.listdir(x)[0])]
            list_vol = self.get_supported_volume(data_directory)

            # flag : which folder/files to choose from the list. defaut 0
            if len(list_folders) > 1: 
                scan_type = "Multiple Parts"
            else:
                scan_type = "Single Part"
                
            defect_file = list_txt[0] if len(list_txt) > 0 else ""
            image_folder = list_img[0] if len(list_img) > 0 else ""
            volume_file = list_vol[0] if len(list_vol) > 0 else ""
            volume_mask_file = ""

            scan_name_list = sqlcon.get_all_scan()
            if scan_name in scan_name_list:
                mssg = "Scan Name already exists, please use another name!"
                self.showdialog_warnning(mssg)
                return
            
        else:
            data_directory = None
            scan_data_label = None
            scan_type = None
            defect_file = self.get_path_button(self.defect)
            image_folder = self.get_path_button(self.image_folder)
            volume_file = self.get_path_button(self.volume_file)
            volume_mask_file = self.get_path_button(self.volume_mask_file)
            
        
        object_material =  self.object_material.text()
        filter_type = self.filter_type.text()
        filter_size = self.filter_size.text()
        technician_name = self.technician_name.text()
            
        if scan_data_label == "Good":
            scan_usage = "This tool is safe for reuse in the reactor."
            defect = 'defect-free'
        elif scan_data_label == "Faulty Acceptable":
            scan_usage = "This tool is defective but it still safe for reuse in the reactor."
            defect = 'defective-acceptable'
        elif scan_data_label == "Faulty Unacceptable":
            scan_usage = "Do not use it to rector. Need fixing/visual inspection "
            defect = 'defective'
        elif scan_data_label == "I don't know":
            scan_usage = "Unknown, needs inspection (software + visual verification)"
            defect = 'Unknown'
        else:
            scan_usage = None
            defect = ""

        self.dict_scan = {
             'tool_name':tool_name, 'scan_name':scan_name, 'scan_type':scan_type, 'data_directory':data_directory,  'before_after_status':before_after_status,
             'data_verification':data_verification, 'defect_file':defect_file, 'defect':defect , 'image_location':image_folder, 'volume_location':volume_file,
             'mask_location':volume_mask_file, 'object_material':object_material, 'filter_type':filter_type, 'filter_size':filter_size,
             'technician_name':technician_name, 'scan_usage':scan_usage, 'scan_data_label':scan_data_label,
        }

        # to complete - volume update rules to be followed:
        # 1. new volume file - regenerate mask file
        # 2. new mask file - test for compatability with existing volume file
        # note: look into 'accept' and 'reject' of registration result

        # get mask volume
        try:
            volume_mask_file = VerifyReg_Win.upload_new_volume(self.dict_scan, self.update_flag)
        except:
            print("[Exception] Volume mask file unable to be generated!")
            volume_mask_file = ""
        self.dict_scan['mask_location'] = volume_mask_file
        
        print(self.dict_scan)
        #print(list_files)
        
        if self.update_flag == 1:
            # Scan sanity check
            err, mssg = self.check_scan_data()
            if err == 0  :
                self.close()
            else:
                Dataset_Win.showdialog_warnning(mssg)
                return
        else:
            self.close()

        if self.update_flag == 0:
            # save new data to SQL 
            Dataset_Win.SQL_add_new_scan(self.dict_scan)
        else:
            # update existing data in SQL
            Dataset_Win.SQL_update_scan(self.dict_scan) 
        

    def check_scan_data(self): 
        print('==> save the new scan of tool [%s] in MySQL : \n'%(self.tool_name))
        mssg = ''
        err = 0
        print('scan_name:',self.scan_name.text())
        if self.scan_name.text() == '' :
            mssg = mssg + '- Please type a non-empty  scan name !\n'
            err = err + 1

        defect = self.get_path_button(self.defect)
        print('defect:',defect)
  
        if  os.path.isfile(defect ) == False:
            mssg = mssg + '- Please select the defect file !\n'
            err = err + 1
        
        import glob 
        image_folder = self.get_path_button(self.image_folder)
        print('image_folder :',image_folder)
        if  len(glob.glob( os.path.join(image_folder , "*.tif" ) ) ) == 0 and len(glob.glob( os.path.join(image_folder , "*.tiff" ) ) ) == 0 :
            mssg = mssg + '- Please select folder containing tif images!\n'
            err = err + 1

        volume_file = self.get_path_button(self.volume_file)
        print('volume_file :',volume_file)

        if  os.path.isfile(volume_file) == False :
            mssg = mssg + '- Please select the volume file !\n'
            err = err + 1

        print('volume_mask_file :',self.get_path_button(self.volume_mask_file))
        print('object_material :',self.object_material.text())
        print('filter_type:',self.filter_type.text())
        print('filter_size :',self.filter_size.text())
        print('technician_name :',self.technician_name.text())

        return err, mssg


    def cancel_saving(self):

        print('cancel the saving of the new tool in MySQL')
        self.close()


class ProgressBar(QProgressBar):

   def __init__(self, *args, **kwargs):
        super(ProgressBar, self).__init__(*args, **kwargs)
        self.setValue(0)
   def increment(self):
      self.setValue(self.value() + 1)

class Progress_bar_windows(QWidget):

    def __init__(self, *args, **kwargs):
        super(Progress_bar_windows, self).__init__(*args, **kwargs)
        self.resize(600, 100)
        self.bar1 = ProgressBar(self, minimum=0, maximum=100, objectName="RedProgressBar")
        layout = QVBoxLayout(self)
        layout.addWidget( self.bar1 )

        layout.addWidget(  
            ProgressBar(self, minimum=0, maximum=0, objectName="GreenProgressBar"))

      #   layout.addWidget(  
      #       ProgressBar(self, minimum=0, maximum=100, textVisible=False,
      #                   objectName="GreenProgressBar"))
      #   layout.addWidget(  
      #       ProgressBar(self, minimum=0, maximum=0, textVisible=False,
      #                   objectName="GreenProgressBar"))

      #   layout.addWidget(  
      #       ProgressBar(self, minimum=0, maximum=100, textVisible=False,
      #                   objectName="BlueProgressBar"))
      #   layout.addWidget(  
      #       ProgressBar(self, minimum=0, maximum=0, textVisible=False,
      #                   objectName="BlueProgressBar"))


## RUN fault preduction algorithms  
class OneImageIspectionApp(MainApp):
    def __init__(self):
        #Parent constructor
        super(OneImageIspectionApp,self).__init__()
        self.ui = None
        self.filename =  None
        self.model_path = None
        self.setup()
        self.createMenu()
        self.windows_format()
        self.plot_sample_image()
        #self.update_tool_list()
  
    def plot_sample_image(self):
        self.Plot_image(self.file_before, self.plot_1)

    def setup(self):
        
        import UI.Prediction 
        self.ui = UI.Prediction.Ui_Run_prediction()
        self.ui.setupUi(self)
        self.file_before = 'files/img_after.tif'
        self.msg = '\n\nPlease follow these steps: \n 1- Select the appropriate CT image (.tif) to be classified.  \n 2- Selct the classification algorithm  \n 3- Click Run '
        print(self.msg);self.ui.msg_label.setText(self.msg)

        self.plot_1 = self.ui.Label_plot_1
        self.plot_3 = self.ui.Label_outputs

        # adding list of available tools (from DB)
        self.tool_list = self.ui.tool_list
        #self.tool_name = self.ui.tool_list
        self.tool_list.currentIndexChanged.connect(self.load_select_tool_data)
        #self.update_tool_list()
        self.scan_list = self.ui.scan_list
        self.scan_list.currentIndexChanged.connect(self.load_select_scan_data)
        
        # Algorithm/Model
        self.select_algo = self.ui.select_algo
        self.select_model = self.ui.select_model
        self.select_train = self.ui.select_train
        self.select_train.clicked.connect(self.show_train_scans) 

        self.select_deploy = self.ui.select_deploy
        self.select_deploy.clicked.connect(self.show_train_scans) 

        self.SQL_table = self.ui.SQL_table
        self.SQL_table.hide()

        # algorithm message
        self.current_algorithm = 'Fault prediction'
        msg0 = 'Select the classification algorithm:'; self.select_algo.setText(msg0)
        msg0 = 'Select the  model:';self.select_model.setText(msg0)

        # adding list of items to combo box (from DB)
        self.algorithm = self.ui.comboBox
        self.list_classification_algorithms = sqlcon.get_all_classify_model()
        self.algorithm.addItems(self.list_classification_algorithms)
        self.algorithm.currentIndexChanged.connect(self.load_trained_models)

        # model path list 
        self.model_path_list = self.ui.model_path_list
        #self.model_path_list.addItems(self.models_list_classification_algorithms)
        #self.load_itegrated_classification_algorithms()

        # load and setup initial trained models
        #self.load_trained_models()
        self.ui.pushButton_3.clicked.connect(self.Run_Prediction) 

        # Amin
        self.ui.threshold_slider.setMaximum(200)
        self.ui.threshold_slider.setValue(20)
        self.update_param()
        self.ui.threshold_slider.valueChanged.connect(self.update_param)

        # root folder
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.root_folder)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.root_folder))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        #self.treeView.doubleClicked.connect(self.get_select_folder)
   
    def show_train_scans(self):
        tool_name = self.tool_list.currentText()
        
        if self.select_train.isChecked():
            self.SQL_table.show()
            self.scan_list.hide()

            # Matthew: select the aprppriat scan  not all depending on: selected tool name, type of model, classes,  (2D/3D)  
            training_scans = sqlcon.get_all_scan_table(self.tool_list.currentText() ) 
            #self.SQL_table = self.update_table_training(self.SQL_table, training_scans)
            self.update_table(self.SQL_table, training_scans) # awitch to training table
            self.path = os.path.join(self.root_folder, tool_name) if tool_name != "" else self.root_folder
            
        elif self.select_deploy.isChecked():
            self.SQL_table.hide()
            self.scan_list.show()
            scan_name = self.scan_list.currentText()
            self.path = os.path.join(self.root_folder, tool_name, scan_name) if scan_name != "" else os.path.join(self.root_folder, tool_name)

        else:
            print("Possible error in selecting between train and deploy")

        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self)
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        
            
    def update_tool_list(self):
        # Update tools list:
        self.tool_list.clear()
        self.tool_list_names = [""]
        self.tool_list_names.extend(sqlcon.get_all_tool())
        #print('loaded list of tool ===', self.tool_list_names)
        # update the tool list into the tool dropbox
        self.tool_list.clear()
        self.tool_list.addItems(self.tool_list_names)
        #print(' updated tool list ===', self.tool_list_names)

    def load_select_tool_data(self):
        tool_name = self.tool_list.currentText()
        #self.treeView.setRootIndex(self.treeModel.index(self.root_folder + tool_name))
        self.path = self.root_folder + tool_name
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        #self.treeView.setRootIndex(self.treeModel.index(self.path))

        self.scan_list.clear()
        scan_names = [""]
        if tool_name != "":
            scan_names.extend(sqlcon.get_all_tool_scan(tool_name))
        self.scan_list.addItems(scan_names)

        list_tool_scans = sqlcon.get_all_scan_table(tool_name)
        self.update_table(self.SQL_table, list_tool_scans)
        self.load_trained_models()

        # clean existing text
        mssg = ""
        self.plot_3.setText(mssg)
        # clean existing image
        

    def load_select_scan_data(self):
        tool_name = self.tool_list.currentText()
        scan_name = self.scan_list.currentText()
        #self.treeView.setRootIndex(self.treeModel.index(self.root_folder + tool_name))

        if tool_name == "":
            self.path = self.root_folder
        elif scan_name == "":
            self.path = self.root_folder + tool_name
        else:
            self.path = self.root_folder + tool_name + '/' + scan_name
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        #self.treeView.setRootIndex(self.treeModel.index(self.path))

    def load_itegrated_classification_algorithms(self):

        type_data = self.file_image_or_volume(self.file_before)
        if self.algorithm.currentText() == "":
            return

        # get the path of model (from DB)
        query_models = self.root_classification + self.algorithm.currentText() +  "/" + type_data +"*.pth"
        print(query_models)
        self.models_list_classification_algorithms = self.get_trained_models(query_models)
        
        self.model_path_list.clear()
        self.model_path_list.addItems(self.models_list_classification_algorithms)

    def load_trained_models(self):
        tool_name = self.tool_list.currentText()
        model_name = self.algorithm.currentText()
        
        list_trained = sqlcon.get_best_model(model_name, tool_name)
        if type(list_trained) == str:
            train_paths = ["- no trained models -"]
        else:
            train_paths = [list_trained['model_version_name'].iloc[-1]]
            
        self.model_path_list.clear()
        self.model_path_list.addItems(train_paths)
        print("Model trained versions:", train_paths)
        
    def get_select_folder(self, signal):
        folder=self.treeView.model().filePath(signal)
        if self.radio_before.isChecked() :
             self.folder_before = folder
             msg = '- Selected folder root before is \n folder = ' + folder
        if self.radio_after.isChecked() :
             self.folder_after = folder
             msg = '- Selected folder root after is \n folder = ' + folder

        print(msg)
        print(msg);self.ui.msg_label.setText(msg)
        
    def change_name(self, index):
        """ rename """
        if not index.isValid():
            return

        model = index.model()
        old_name = model.fileName(index)
        path = model.fileInfo(index).absoluteFilePath()

        # ask new name
        name, ok = QInputDialog.getText(self, "New Name", "Enter a name", QLineEdit.Normal, old_name)
        if not ok or not name:
            return
        
        # rename
        model = index.model()
        wasReadOnly = model.isReadOnly()
        model.setReadOnly(False)
        model.setData(index, name)
        model.setReadOnly(wasReadOnly)

    def get_selected_path(self, index):
        """ for test """
        if not index.isValid():
            return
        model = index.model()
        path = model.fileInfo(index).absoluteFilePath()

        if os.path.isfile(path): 
            self.file_before = path
            msg = self.Plot_image(self.file_before, self.plot_1)

        else:
            msg=  '- Warning (1):  You selected a folder not a file!!\n - ' + self.path_end(path)
                      
            
        msg = msg + self.msg
        print(msg);self.ui.msg_label.setText(msg)
        
    def get_selected_scan(self, index):
        """ for test """
        if not index.isValid():
            return
        model = index.model()
        raw = model.fileInfo(index).absoluteFilePath()

        msg = 'Uploading in process .... \n file = ' + raw
        print(msg);self.ui.msg_label.setText(msg)

        self.treeModel.setRootPath(self.path)
        
    def update_param(self):

        msg= 'param = ' + str(self.ui.threshold_slider.value())
        Min_object_area = self.ui.Amin_label.setText(msg)

    def path_end(self, location):
        if os.path.exists(location) == True:
            sample_pth = "/".join([x for x in location.split('/')[-3:]])
            return sample_pth
        else:
            return location
        
    def run_CNN_deploy_A(self):

        print( 'selected model = ', self.models_list_classification_algorithms)
        # Run ALgorithm 1
        msg= 'Running the algorithm :'+ str(self.algorithm.currentText()) + ' with param=' 
        print(msg); self.ui.msg_label.setText(msg)
        # Timing start
        tic = time.perf_counter()
        # # Load the images
        # img = cv2.imread(self.file_before,0)

        ## run the algorithm 
        # get model_path (from DB)
        model_path = self.root_classification  + self.algorithm.currentText() + '/' +  self.model_path_list.currentText()
        
        
        if self.file_image_or_volume(self.file_before)=='2D': #
           
            size=(128,128)                                # resizinf images for prediction
            classes = ['CS02 - Missing #1 Simulation Tool dataset',
            'CS03 - Missing #2 Simulation Tool dataset',
            'CS04 - missingparts-1 2D',
            'CS05 - missingparts-2 2D',
            'CS06 - missingparts-3 2D',
            'CS07 - missingparts-4 2D']

            fault = Autils_fault_prediction.deploy_model(model_path, classes, self.file_before, size=size)
            print("Class fault:", fault)
            # fault type
            if fault[:4] == 'CS01':
                msg=  "The tool is defect-free"

            else :
                msg=  "The tool is defective  \n\n  Fault type = " + fault 

            self.plot_3.setText(msg)
            # Timing end
            toc = time.perf_counter() - tic

            # toc = int(toc*100)/100

            # # save image_show
            # import numpy as np
            # from PIL import Image
            # im = Image.fromarray(img_box)
            # filename = "files/detection.jpg"
            # im.save(filename)
            # self.Plot_image( filename, self.plot_3)

            msg= '\n- Execution time = ' + str(toc) + ' seconds'

        elif self.file_image_or_volume(self.file_before)=='3D': #
            print('No algoithm is found ', model_path )

        else: 
            msg = '\n the selected file is not supported!! \n - file path = '+ self.path_end(self.file_before)

        msg = msg + self.msg
        print(msg);self.ui.msg_label.setText(msg)

    def run_CNN_train_A(self):

        print( 'selected model = ', self.models_list_classification_algorithms)

        if self.file_before[-4:]=='.tif':# or self.file_before[-4:]=='.jpg' or self.file_before[-4:]=='.png':
            # Run ALgorithm 1
            msg= 'Running the algorithm :'+ str(self.algorithm.currentText()) 
            print(msg); self.ui.msg_label.setText(msg)
            # Timing start
            tic = time.perf_counter()
            # # Load the images
            # img = cv2.imread(self.file_before,0)

            ## run the algorithm 
            #Matthew: get model_path
            model_path = self.root_classification  + self.algorithm.currentText() + '/' +  self.model_path_list.currentText()
            size=(128,128)                                # resizinf images for prediction
            classes = ['CS02 - Missing #1 Simulation Tool dataset',
            'CS03 - Missing #2 Simulation Tool dataset',
            'CS04 - missingparts-1 2D',
            'CS05 - missingparts-2 2D',
            'CS06 - missingparts-3 2D',
            'CS07 - missingparts-4 2D']

            TRAIN_FOLDER = 'data/train'

            # Matthew: Load input of this algorithm from the  DB as following
            # input_dic = {"model_path":' ', "classes":[' ', ' '], "TRAIN_FOLDER":' ', "size":(2,2)}

            
            performance = Autils_fault_prediction.train_model(model_path, classes, TRAIN_FOLDER, size=size)

            self.output_dic = {"model_path":' ', "classes":[' ', ' '], "TRAIN_FOLDER":' ', "size":(2,2), "performance": performance}

            # fault type
            msg=  "The model " + model_path  + " was trained with the new dataset folder " + TRAIN_FOLDER

            self.plot_3.setText(msg)
            # Timing end
            toc = time.perf_counter() - tic

            # toc = int(toc*100)/100

            # # save image_show
            # import numpy as np
            # from PIL import Image
            # im = Image.fromarray(img_box)
            # filename = "files/detection.jpg"
            # im.save(filename)
            # self.Plot_image( filename, self.plot_3)

            msg=  '\n- Execution time = ' + str(toc) + ' seconds'

        else: 
            msg = '\n the selected file is not supported!! \n - file path = '+ self.path_end(self.file_before)

        msg = msg + self.msg
        print(msg);self.ui.msg_label.setText(msg)
        main_Win.showdialog_train_verify(self.output_dic)


    def multiclass_classification_train_A(self):

        print( 'selected model = ', self.models_list_classification_algorithms)
        
        from scripts import Autils_classification
        from scripts.Autils_classification import Net
        project = 'CNN-MClss' 
        experimert = 'defect-free_defective'#1-2-3
        type_algo = 'classification'
        RAW_DATA_ROOT =  'data/workspace/train_model' # contains folders : defective + defect-free 

        # Matthew: need to copy the selected scan by fault in the <RAW_DATA_ROOT> folder
        #------------------------------ Raw Data  --------------------------------------
        raw_data_folder = 'Yes'#'No'#
        sys.path.append(RAW_DATA_ROOT); 
        ext_list = ['.tif', 'tiff', 'jpg']   # extention of raw data
        #--------------------------- Structured Data  ----------------------------------
        size = (128,128)#(256,256)#          # Create resized copies of all of the source images in the workspace
        ext = '.jpg'                         # extension to to be used for AI-learning in the workspace
        WORKSPACE_folder = 'data/workspace/'

        #----------------------------- Output model  ----------------------------------
        # input data folders 
        data_TAG = '_'.join(RAW_DATA_ROOT.split('/')[-3:-1])
        DIR_WORKSPACE = WORKSPACE_folder + data_TAG + '/'
        DIR_TRAIN = DIR_WORKSPACE + 'train/'
        DIR_TEST = DIR_WORKSPACE + 'test/'
        DIR_DEPLOY = DIR_WORKSPACE + 'deploy/'

        # copy select images to raw data root
        tool_name = self.tool_list.currentText()
        image_path_list = sqlcon.get_all_scan_paths(tool_name)
        print(image_path_list)
        # test if image path is present, if not use dialog box
        # test if both class data is present (any two scans for now), if not use dialog box
        
        # # output model 
        import torch
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_folder = "models/" + type_algo + "/" + data_TAG +  '/'
        model_path = model_folder +  '2D_' + project + '_' + str(device)+ experimert +  '.pth'

        # SQL load dataset info from DB the following
        # dataset_info_dic = {"raw_data_folder":' ', "RAW_DATA_ROOT":' ', "ext_list":' ', "size":(x,y), DIR_TRAIN, DIR_TEST, DIR_DEPLOY}
        model_name = self.algorithm.currentText()
        model_name_version = self.model_path_list.currentText()

        if model_name_version == "- no trained models -":
            print("Initiating training new model from scratch...")
            model_path = sqlcon.get_model_path(model_name)
        else:
            print("Initiating training over existing model version...", model_name_version)
            model_version = sqlcon.get_best_model(model_name, tool_name)
            model_path = model_version['model_pkl_location'].iloc[-1]
            model_path = sqlcon.get_model_path(model_name)
            
        workspace_paths = []
        for idx in range(image_path_list.shape[0]):
            path_in_scan = image_path_list['image_location'].iloc[idx]
            scan_name = path_in_scan.split('/')[-2]
            path_in_workspace = RAW_DATA_ROOT + '/' + scan_name + '_images'
            if os.path.isdir(path_in_workspace) == False:
                shutil.copytree(path_in_scan, path_in_workspace)
            else:
                print("Potential data left uncleaned from previous errors!")
            workspace_paths.append(path_in_workspace)
        image_path_list['workspace_paths'] = workspace_paths
        
        print('Workspace =', DIR_WORKSPACE)
        print('model path =', model_path)
        print('data folder flag =', raw_data_folder)
        print('data folder =', RAW_DATA_ROOT)
        print('model folder =', model_folder)
        print('device =', device)

        #raw_data_folder = image_path_list['workspace_paths'].tolist()
        Autils_classification.build_dataset_workspace(raw_data_folder, RAW_DATA_ROOT, ext_list, size, DIR_TRAIN, DIR_TEST, DIR_DEPLOY)

        epochs = 100                                        # number of epochs 
        split_size = 0.7                                    # train/val split
        batch_size=50                                       # batch size
        num_workers=0                                       # num workers
        optimizer = 'adam'                                  # optimizer to adjust thr Network weights
        lr = 0.001                                          # learning rate
        transfer_learning = 'Yes'#'No'#                     # enable  transfer learning
        es_patience = 2                                     #This is required for early stopping, the number of epochs we will wait with no improvement before stopping
        loss_critereon = 'crossEntropy'                      # loss criteria


        # Matthew: load from DB the following
        # input_dic = {"DIR_TRAIN":' ', "model_path":' ',  paramer: [epochs=epochs, lr=lr, optimizer=optimizer,loss_criteria=loss_critereon, split_size=split_size, batch_size=batch_size,num_workers=num_workers]}

        # make a copy of the model before training, otherwise overlap occurs
        # Run the training algorithm
        model, classes, epoch_nums, training_loss, validation_loss = Autils_classification.train_model(DIR_TRAIN, model_path, epochs=epochs, lr=lr, optimizer=optimizer,\
                                                                                loss_criteria=loss_critereon, split_size=split_size, batch_size=batch_size,\
                                                                                num_workers=num_workers)
        #list_training_scans = ['scan1', 'scan5', 'scan67']
        list_training_scans = image_path_list['image_location'].tolist()
        #self.output_dic = {"model": model, "classes": classes, "model_path": model_path, "list_training_scans": list_training_scans}
                           #"performance_index":0.5, "performance_evaluation":[]}
        #performance_index: value between 0 and 1 with 1 being best model

        # run model on testing set
        truelabels, predictions, TS_sz = Autils_classification.test_model(model_path, DIR_TEST, classes)
        # show perform ance
        Autils_classification.classification_performance(classes, truelabels, predictions, TS_sz= TS_sz)#, TR_sz= len(train_loader.dataset) + len(val_loader.dataset))
        save_flag = main_Win.showdialog_train_verify(self.output_dic)

        print("Model:", model)
        print("Classes:", classes)
        print("Epoch Nums:", epoch_nums)
        print("Training Loss:", training_loss)
        print("Validation Loss:", validation_loss)
        
        train_parameters = {"list_training_scans": list_training_scans, "classes": classes, "epochs": epochs, "split_size": split_size,
                            "batch_size": batch_size, "num_workers": num_workers, "optimizer": optimizer, "lr": lr,
                            "transfer_learning": transfer_learning, "es_patience": es_patience, "loss_critereon": loss_critereon}

        performance_res = {"epochs": epoch_nums, "train_loss": training_loss, "valid_loss": validation_loss}

        if save_flag == 65536:
            print("Provide more data")
            main_Win.showdialog_request_more_data()
        else: 
            print("Saving the model")
            # saving model, save input images, classes to new model version
            # save output_dict as pkl file
            # (later) saving results (confusion matrix, performance graph)
            self.model_results = {"model_name": model_name, "model_path": model_path, "tool_name": tool_name, "learning_meta_data": train_parameters,
                                  "performance_meta_data": performance_res, "model_object": model}
            print(self.model_results)
            
            # save model based data to DB
            sqlcon.sql_save_trained_model(self.model_results)
            main_Win.showdialog_operation_success("The model is saved for future use.")    
            self.load_trained_models()
            
        # clean up existing train data from workspace
        collect_remove = [shutil.rmtree(WORKSPACE_folder + x) for x in os.listdir(WORKSPACE_folder)]
         
    
    def multiclass_classification_deploy_A(self):

        print( 'selected model = ', self.models_list_classification_algorithms)

        from scripts import Autils_classification
        from scripts.Autils_classification import Net
        project = 'CNN-MClss' 
        type_algo = 'classification'
        #------------------------------ Raw Data  --------------------------------------
        raw_data_folder = 'Yes'#'No'#
        RAW_DATA_ROOT =  'data/NVS-proj/'
        sys.path.append(RAW_DATA_ROOT); 
        ext_list = ['.tif', 'tiff', 'jpg']   # extention of raw data

        #--------------------------- Structured Data  ----------------------------------
        size = (128,128)#(256,256)#          # Create resized copies of all of the source images in the workspace
        ext = '.jpg'                         # extension to to be used for AI-learning in the workspace
        WORKSPACE_folder = 'data/workspace/'

        #----------------------------- Output model  ----------------------------------
        # input data folders 
        data_TAG = '_'.join(RAW_DATA_ROOT.split('/')[-3:-1])
        DIR_WORKSPACE = WORKSPACE_folder + data_TAG + '/'
        DIR_TRAIN = DIR_WORKSPACE + 'train/'
        DIR_TEST = DIR_WORKSPACE + 'test/'
        DIR_DEPLOY = DIR_WORKSPACE + 'deploy/'
        
        # # output model 
        import torch
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_folder = "models/" + type_algo + "/" + data_TAG +  '/'
        model_path = model_folder +  '2D_' + project + '_' + str(device)+ '.pth'

        img_to_test = self.file_before
        classes =  ['CS01 - Simulation Tool dataset', 'CS02 - Missing #1 Simulation Tool dataset', 'CS03 - Missing #2 Simulation Tool dataset', 'CS04 - missingparts-1 2D', 'CS06 - missingparts-3 2D']

        # get model name and class names (from DB)
        tool_name = self.tool_list.currentText()
        model_name = self.algorithm.currentText()

        model_name_version = self.model_path_list.currentText()
        model_version = sqlcon.get_best_model(model_name, tool_name)
        
        if model_name_version == "- no trained models -":
            mssg = "Note:\nThere are no trained model versions for the selected tool and model!\nPlease choose a classification algorithm and train a model."
            self.plot_3.setText(mssg)
            return
        
        model_path = model_version['model_pkl_location'].iloc[-1]
        model_path = sqlcon.get_model_path(model_name)
        classes_path = sqlcon.get_model_class_path(model_name)
        with open(classes_path, 'r') as oup:
            classes = list(json.load(oup).keys())

        print('Workspace =', DIR_WORKSPACE)
        print('model_path =', model_path)
        print('data folder =', RAW_DATA_ROOT)
        print('model folder =', model_folder)
        print('device =', device)
        print('classes =', classes)
        from glob import glob
        import random
        import time
        # list_images_paths = glob(DIR_DEPLOY + "*.jpg" )
        # random.shuffle(list_images_paths)

        # deploy the classification model  
        # input_dic={"model_path":' ', "classes": '',"img_to_test":''}

        
        tic = time.perf_counter()         # Timing start
        fault = Autils_classification.predict_image(model_path, classes, img_to_test)
        toc = time.perf_counter() - tic; toc = int(toc*100)/100  # Timing end

        print("predicted fault:", fault)
        # fault type
        if fault[0] == classes[0]:
            msg=  "The tool is defect-free"

        else :
            msg=  "The tool is defective  \n\n  Fault type = " + fault[0] 

        self.plot_3.setText(msg)
        
        msg= '\n- Execution time = ' + str(toc) + ' seconds'
        
        # self.about_actionoutput_dic = {"img_to_test": img_to_test, "fault": fault}
        # main_Win.showdialog_fault_prediction_verify(self.output_dic)
        

    def RCNN_RESNET50_train_A(self):
        # import sys
        # from scripts import Autils_Object_detection
        # from scripts.Autils_Object_detection import Net
        project = 'RCNN-RESNET50-obj' 
        type_algo = 'localization'

        #------------------------------ Raw Data  --------------------------------------
        raw_data_folder = 'Yes'# 'No'#
        RAW_DATA_ROOT =  'data/raw-data/'
        sys.path.append(RAW_DATA_ROOT); 
        ext_list = ['.tif', 'tiff', 'jpg']   # extention of raw data

        #--------------------------- Structured Data  ----------------------------------
        size = (128,128)#(256,256)#          # Create resized copies of all of the source images
        WORKSPACE_folder = 'data/workspace/'

        #----------------------------- Output model  ----------------------------------
        # input data folders 
        data_TAG = 'obj_' + '_'.join(RAW_DATA_ROOT.split('/')[-3:-1])
        DIR_WORKSPACE = WORKSPACE_folder + data_TAG + '/'


        # # output model 
        import torch
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        model_path =  "models/localization/" + project +"/2D_" + data_TAG +  '_' + str(device)+ '.pth'     # model path 
        print('Workspace =', DIR_WORKSPACE)
        print('model_path =', model_path)
        print('data_folder =', RAW_DATA_ROOT)
        print('device =', device)

        Autils_Object_detection.build_dataset_workspace(raw_data_folder, RAW_DATA_ROOT, DIR_WORKSPACE)

        Autils_Object_detection.split_dataset_train_test(DIR_WORKSPACE)

        transfer_learning = 'Yes'#'No'#                     # enable transfer learning
        epochs = 2                                         # number of epochs 
        es_patience = 2                                     #This is required for early stopping, the number of epochs we will wait with no improvement before stopping
        N_split = 2                                        # number of times the epochs will run
        lr = 0.005                                          # learning rate
        momentum=0.9                                        # learning momentum
        weight_decay=0.0005                                 # learning decay weight
        split_size = 0.7                                    # train/val split
        batch_size=4                                        # batch size
        num_workers=0                                       # num workers

        # Run the training algorithm
        model, CLASS_MAPPING, epoch_nums, training_loss, validation_loss = Autils_Object_detection.train_model(DIR_WORKSPACE, model_path, epochs=epochs, lr=lr, momentum=momentum, weight_decay=weight_decay,\
                                                                                es_patience=es_patience, N_split=N_split, batch_size=batch_size, num_workers=num_workers, transfer_learning=transfer_learning)

        # model_path = 'models/localization/RCNN-RESNET50-obj/2D_obj_data_raw-data_RCNN-RESNET50-obj_cpu.pth'
        batch_size=4                                        # batch size
        num_workers=0                                       # num workers

        # Run the training algorithm
        model, CLASS_MAPPING, image_list, val_iou_list = Autils_Object_detection.test_model_performance(DIR_WORKSPACE, model_path, batch_size=batch_size, num_workers=num_workers,)

        self.output_dic = {"model":model, "CLASS_MAPPING":CLASS_MAPPING, "model_path":model_path}

        main_Win.showdialog_train_verify(self.output_dic)   


    def RCNN_RESNET50_deploy_A(self):

        project = 'RCNN-RESNET50-obj' 
        type_algo = 'localization'

        #------------------------------ Raw Data  --------------------------------------
        raw_data_folder = 'Yes'# 'No'#
        RAW_DATA_ROOT =  'data/raw-data/'
        sys.path.append(RAW_DATA_ROOT); 
        ext_list = ['.tif', 'tiff', 'jpg']   # extention of raw data

        #--------------------------- Structured Data  ----------------------------------
        size = (128,128)#(256,256)#          # Create resized copies of all of the source images
        WORKSPACE_folder = 'data/workspace/'

        #----------------------------- Output model  ----------------------------------
        # input data folders 
        data_TAG = 'obj_' + '_'.join(RAW_DATA_ROOT.split('/')[-3:-1])
        DIR_WORKSPACE = WORKSPACE_folder + data_TAG + '/'


        # # output model 
        import torch
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        model_path =  "models/localization/" + project +"/2D_" + data_TAG +  '_' + str(device)+ '.pth'     # model path 
        print('Workspace =', DIR_WORKSPACE)
        print('model_path =', model_path)
        print('data_folder =', RAW_DATA_ROOT)
        print('device =', device)

        # model_path = 'models/classification/data_aRTist-tool2/2D_CNN-MClss_cpu.pth'
        CSV_DATA_FILE, CSV_TRAIN_FILE, CSV_TEST_FILE, CSV_DEPLOY_FILE, DIR_TRAIN, DIR_TEST, DIR_DEPLOY  = Autils_Object_detection.get_directories(DIR_WORKSPACE)
        # run model on testing set
        pred_images,  TS_sz = Autils_Object_detection.deploy_model(model_path, DIR_TEST)
      
    
    def save_trained_model(model_data):
        return
    

    def Run_Prediction(self):
#----------------------------------------------# TRAINING  #----------------------------------------------#

        if self.select_train.isChecked():
            
            if self.algorithm.currentText() == '2-CNN':
                self.run_CNN_train_A()

            elif self.algorithm.currentText() == '3-new-algorithm':
                a=1; 
                #self.new__train_algorithm_G()  
            
            elif self.algorithm.currentText() == '2D_CNN-MClss_cpu':#'data_NVS-proj':
                self.multiclass_classification_train_A()

            elif self.algorithm.currentText() == 'RCNN-RESNET50-obj':
                self.RCNN_RESNET50_train_A()

#----------------------------------------------# DEPLOYMENT #----------------------------------------------#

        elif self.select_deploy.isChecked():

            #if self.algorithm.currentText() == '2D_CNN-MClss_cpu':
            #elif self.algorithm.currentText() == '2D_CNN-MClss_cuda':
            #if self.algorithm.currentText() == '2-CNN':
            #if self.algorithm.currentText() == 'data_NVS-proj':
            
            if self.algorithm.currentText() == '2-CNN':
                self.run_CNN_deploy_A()

            elif self.algorithm.currentText() == '3-new-algorithm':
                a=1; 
                #self.new_deploy_algorithm_G()  

            elif self.algorithm.currentText() == '2D_CNN-MClss_cpu':#'data_NVS-proj':
                self.multiclass_classification_deploy_A()

            elif self.algorithm.currentText() == 'RCNN-RESNET50-obj':
                self.RCNN_RESNET50_deploy_A()

        else:
            print("No model selected in connection from GUI")

## RUN CT inspection algorithms  
class InspectionApp(MainApp):
    def __init__(self):
        #Parent constructor
        super(InspectionApp,self).__init__()
        self.config_algo = 0
        self.ui = None
        self.setup()
        self.filename =  None
        self.createMenu()
        self.windows_format()
        self.plot_sample_image()

    def setup(self):
        
        import UI.RunAlgorithmsUI 
        self.ui = UI.RunAlgorithmsUI.Ui_Run_inspection()
        self.ui.setupUi(self)

        self.file_before = 'files/img_before.tif'
        self.file_after = 'files/img_after.tif'

        self.msg ='\n\n - Please follow these steps: \n 1- Select the appropriate CT data.  \n 2- Selct the desired algorithm  \n 3- Click Run Algorithm'
        self.ui.msg_label.setText(self.msg)
        self.verify_save = self.ui.verify_save
        self.verify_save.clicked.connect(self.Verify_Save_KnowledgeBase) 
        self.verify_save.hide()

        self.plot_1 = self.ui.Label_plot_1
        self.plot_2 = self.ui.Label_plot_2
        self.plot_3 = self.ui.Label_plot_3
        self.Label_outputs = self.ui.Label_outputs

        self.vtk_plot_1 = self.ui.Frame_plot_1
        self.vtk_plot_2 = self.ui.Frame_plot_2
        self.vtk_plot_3 = self.ui.Frame_plot_3

        # Initialize the 3D plots
        path0 ='files/volume_before.rek'
        self.vtk_widget_1, self.vtk_plot_1 = self.create_vtk_widget(self.vtk_widget_1, self.vtk_plot_1, path0)
        path0 ='files/volume_after.rek'
        self.vtk_widget_2, self.vtk_plot_2 = self.create_vtk_widget(self.vtk_widget_2, self.vtk_plot_2, path0)
        path0 ='files/volume_after0.rek'
        self.vtk_widget_3, self.vtk_plot_3 = self.create_vtk_widget(self.vtk_widget_3, self.vtk_plot_3, path0)

        # Show 2D images first
        self.plot_1.show();self.plot_2.show();self.plot_3.show()
        self.vtk_plot_1.hide();self.vtk_plot_2.hide();self.vtk_plot_3.hide()
        # Algorithm/Model
        self.select_algo = self.ui.select_algo
        self.select_model = self.ui.select_model

        # adding list of available tools (from DB)
        self.tool_list = self.ui.tool_list
        self.tool_list.currentIndexChanged.connect(self.load_select_tool_data)
        #self.update_tool_list()
        self.scan_list = self.ui.scan_list
        self.scan_list.currentIndexChanged.connect(self.load_select_scan_data)
        
        # adding list of items to combo box
        self.algorithm = self.ui.comboBox
        self.algorithm.clear()
        self.list_localization_algorithms = [""]
        self.list_localization_algorithms.extend(sqlcon.get_all_localize_model())
        self.algorithm.addItems(self.list_localization_algorithms)
        self.algorithm.currentIndexChanged.connect(self.load_itegrated_inspection_algorithms)
        self.current_algorithm = self.algorithm.currentText()
        
        # model path list 
        self.model_path_list = self.ui.model_path_list
        #self.load_itegrated_inspection_algorithms()
        #self.model_path_list.currentIndexChanged.connect(self.load_itegrated_inspection_models)

        # self.ui.pushButton_1.clicked.connect(self.upload_image)
        # self.ui.pushButton_2.clicked.connect(self.upload_image)
        self.ui.pushButton_3.clicked.connect(self.run_localisation)     

        # Amin
        self.ui.threshold_slider.setMaximum(200)
        self.ui.threshold_slider.setValue(20)
        self.update_Amin()
        self.ui.threshold_slider.valueChanged.connect(self.update_Amin)

        # select type input
        self.radio_after = self.ui.radio_after
        self.radio_before = self.ui.radio_before
        self.radio_before.setChecked(1)
        self.radio_before.toggled.connect(self.radio_change_init)
        self.radio_after.toggled.connect(self.radio_change_init)

        # root folder
        # self.treeView = self.ui.treeView_1
        # # model = QDirModel()
        # # self.view.setModel(model)
        # # self.view.setRootIndex(model.index(self.root_folder))
        self.path = None
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.root_folder)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.root_folder))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.upload_selected_dataset_item(index))
        self.treeView.doubleClicked.connect(self.get_select_folder)

    def radio_change_init(self):
        tool_name = self.tool_list.currentText()
        before_after = ['after', 'before']
        scan_names = []
        if self.radio_before.isChecked():
             self.scan_list.clear()
             if tool_name != "":
                before_after_data = sqlcon.get_scan_before_after(tool_name)
                scan_samples = [x + " " + '[' + before_after[int(y)] + ']' for x, y in zip(before_after_data['scan_name'], before_after_data['before_after_status'])]
                scan_before = [x for x in scan_samples if 'before' in x]
                scan_names.extend(scan_before)
                    
        elif self.radio_after.isChecked():
            self.scan_list.clear()
            if tool_name != "":
                before_after_data = sqlcon.get_scan_before_after(tool_name)
                scan_samples = [x + " " + '[' + before_after[int(y)] + ']' for x, y in zip(before_after_data['scan_name'], before_after_data['before_after_status'])]
                scan_after = [x for x in scan_samples if 'after' in x]
                scan_names.extend(scan_after)
                
        else:
            self.scan_list.clear()
            if tool_name != "":
                before_after_data = sqlcon.get_scan_before_after(tool_name)
                scan_samples = [x + " " + '[' + before_after[int(y)] + ']' for x, y in zip(before_after_data['scan_name'], before_after_data['before_after_status'])]
                scan_names.extend(scan_samples)
        
        self.scan_list.addItems(scan_names)
        print("Scan reset on before after radio button click...")

    def update_tool_list(self):
        # Update tools list:
        self.tool_list.clear()
        self.tool_list_names = [""]
        self.tool_list_names.extend(sqlcon.get_all_tool())
        print('loaded list of tool ===', self.tool_list_names)
        # update the tool list into the tool dropbox
        self.tool_list.clear()
        self.tool_list.addItems(self.tool_list_names)
        print(' updated tool list ===', self.tool_list_names)

    def load_select_tool_data(self):
        tool_name = self.tool_list.currentText()
        #self.treeView.setRootIndex(self.treeModel.index(self.root_folder + tool_name))
        self.path = self.root_folder + tool_name
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        #self.treeView.setRootIndex(self.treeModel.index(self.path))

        before_after = ['after', 'before']
        scan_names = []
        if self.radio_before.isChecked():
             self.scan_list.clear()
             if tool_name != "":
                before_after_data = sqlcon.get_scan_before_after(tool_name)
                scan_samples = [x + " " + '[' + before_after[int(y)] + ']' for x, y in zip(before_after_data['scan_name'], before_after_data['before_after_status'])]
                scan_before = [x for x in scan_samples if 'before' in x]
                scan_names.extend(scan_before)
                    
        elif self.radio_after.isChecked():
            self.scan_list.clear()
            if tool_name != "":
                before_after_data = sqlcon.get_scan_before_after(tool_name)
                scan_samples = [x + " " + '[' + before_after[int(y)] + ']' for x, y in zip(before_after_data['scan_name'], before_after_data['before_after_status'])]
                scan_after = [x for x in scan_samples if 'after' in x]
                scan_names.extend(scan_after)
                
        else:
            self.scan_list.clear()
            if tool_name != "":
                before_after_data = sqlcon.get_scan_before_after(tool_name)
                scan_samples = [x + " " + '[' + before_after[int(y)] + ']' for x, y in zip(before_after_data['scan_name'], before_after_data['before_after_status'])]
                scan_names.extend(scan_samples)
        self.scan_list.addItems(scan_names)

    def load_select_scan_data(self):
        tool_name = self.tool_list.currentText()
        scan_name = self.scan_list.currentText()
        #self.treeView.setRootIndex(self.treeModel.index(self.root_folder + tool_name))

        if tool_name == "":
            self.path = self.root_folder
        elif scan_name == "":
            self.path = self.root_folder + tool_name
        else:
            scan_name = " ".join([x for x in scan_name.split()[:-1]])
            print(scan_name)
            self.path = self.root_folder + tool_name + '/' + scan_name
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        #self.treeView.setRootIndex(self.treeModel.index(self.path))

    def Verify_Save_KnowledgeBase(self):
         # Matthew :  save the results of the methods/functions in knowledge base
        if self.current_algorithm == 3: #fault characterization
            self.showdialog_FullInspection_results_saving() 
        elif self.current_algorithm == 2: # fault localizatoin 
            self.showdialog_Localization_results_saving()
        elif self.current_algorithm == -1: # registration
            self.showdialog_registration() 
        else:
            print( 'Error :  processing method undefined!!!')        
        # Hide button after compition 
        self.verify_save.hide()

    def load_vars(self):
        if self.current_algorithm == 3: #fault characterization 
            msg0 = 'Select the fault characterization algorithm:'; self.select_algo.setText(msg0)
            msg0 = 'Select the model:';self.select_model.setText(msg0)
            
            self.algorithm.clear()
            self.algorithm.addItems(self.get_subfolders(self.root_charachterization))
            self.load_itegrated_inspection_algorithms()
        

        elif self.current_algorithm == 2: # fault localizatoin 
            msg0 = 'Select the fault localization algorithm:'; self.select_algo.setText(msg0)
            msg0 = 'Select the model:';self.select_model.setText(msg0)
            self.algorithm.clear()
            self.algorithm.addItems(self.get_subfolders(self.root_localization))
            #self.load_itegrated_inspection_algorithms()

        elif self.current_algorithm == -1: # registration 

            # algorith message
            msg0 = 'Select the registration algorithm:'; self.select_algo.setText(msg0)
            msg0 = 'Select the model:';self.select_model.setText(msg0)
            self.algorithm.clear()
            self.algorithm.addItems(self.get_subfolders(self.root_registration))
            self.load_itegrated_inspection_algorithms()

        else:
            print( 'Error :  processing method undefined!!!')

    def load_itegrated_inspection_algorithms(self):
        print('Algo', self.current_algorithm)
        if self.current_algorithm!=0: 
            data_type = self.file_image_or_volume(self.file_before)

        if self.current_algorithm==-1:
            query_models = self.root_registration+ self.algorithm.currentText() +  "/" + data_type + "*.pth"
            self.models_list_registration_algorithms =self.get_trained_models(query_models)
            self.model_path_list.clear()
            self.model_path_list.addItems(self.models_list_registration_algorithms)

        elif self.current_algorithm==3:
            query_models = self.root_charachterization+ self.algorithm.currentText() +  "/" + data_type + "*.pth"
            self.models_list_charachterization_algorithms =self.get_trained_models(query_models)
            self.model_path_list.clear()
            self.model_path_list.addItems(self.models_list_charachterization_algorithms)

        elif self.current_algorithm==2:
            query_models = self.root_localization + self.algorithm.currentText() +   "/" + data_type + "*.pth"
            self.models_list_localization_algorithms =self.get_trained_models(query_models)
            self.model_path_list.clear()
            self.model_path_list.addItems(self.models_list_localization_algorithms)

        else:
            print('Error : Undefined algorithm!!!', self.current_algorithm)

    def load_itegrated_inspection_models(self):

        self.algorithm_inputs_param()
        if self.type_algo == '2D':
            self.Plot_image( self.file_after, self.plot_3)

        elif self.type_algo == '3D':
            volume_arr, vol1 = self.get_volume( self.file_after )
            self.Plot_volume(self.vtk_widget_3, vol1, volume_arr)
    
    def get_select_folder(self, signal):
        folder=self.treeView.model().filePath(signal)
        if self.radio_before.isChecked() :
             self.folder_before = folder
             msg = '- Slected folder root before is \n folder = ' + folder
        if self.radio_after.isChecked() :
             self.folder_after = folder
             msg = '- Slected folder root after is \n folder = ' + folder

        print(msg)
        msg = msg = self.msg
        print(msg);self.ui.msg_label.setText(msg)
        
    def plot_sample_image(self):

        self.Plot_image(self.file_before, self.plot_1)
        self.Plot_image(self.file_after , self.plot_2)
        self.Plot_image(self.file_after, self.plot_3)

    def upload_selected_dataset_item(self, index):
        print('file before: %s \nfile before: %s '%(self.file_before, self.radio_after))

        if not index.isValid():
            return
        model = index.model()
        path = model.fileInfo(index).absoluteFilePath()

        msg = ' The file [' + os.path.basename(self.file_before) + '] was loaded successfully!!\n'
        if self.radio_before.isChecked() :
            self.file_before = path
            self.ui.msg_label.setText('Uploading in process of CT before.... \n file = ' + os.path.basename(self.file_before) )

            if self.file_image_or_volume(self.file_before)=='3D':
                self.plot_1.hide();self.plot_3.hide()
                self.vtk_plot_1.show();self.vtk_plot_3.show()
                volume_arr, vol1 = self.get_volume( self.file_before )
                self.Plot_volume(self.vtk_widget_1, vol1)

            elif self.file_image_or_volume(self.file_before)=='2D':
                self.plot_1.show();self.plot_3.show()
                self.vtk_plot_1.hide();self.vtk_plot_3.hide()
                self.Plot_image(self.file_before, self.plot_1)

            else: 
                msg = '- The selected item (CT before) is not suppoted: \n' +  os.path.basename(self.file_before)



        if self.radio_after.isChecked() :
            self.file_after = path

            self.ui.msg_label.setText('Uploading in process of CT after.... \n file = ' + os.path.basename( self.file_before) )
            if self.file_image_or_volume(self.file_after)=='3D':
                self.plot_2.hide();self.plot_3.hide()
                self.vtk_plot_2.show();self.vtk_plot_3.show()
                volume_arr, vol2 = self.get_volume( self.file_after )
                self.Plot_volume(self.vtk_widget_2, vol2)
            elif self.file_image_or_volume(self.file_after)=='2D':
                self.plot_2.show();self.plot_3.show()
                self.vtk_plot_2.hide();self.vtk_plot_3.hide()
                self.Plot_image(self.file_after, self.plot_2)

            else: 
                msg = '- The selected item (CT after) is not suppoted: \n' +  os.path.basename(self.file_after)


        msg = msg + self.msg
        print(msg); self.ui.msg_label.setText(msg)

    def change_name(self, index):
        """ rename """
        if not index.isValid():
            return

        model = index.model()
        old_name = model.fileName(index)
        path = model.fileInfo(index).absoluteFilePath()

        # ask new name
        name, ok = QInputDialog.getText(self, "New Name", "Enter a name", QLineEdit.Normal, old_name)
        if not ok or not name:
            return
        
        # rename
        model = index.model()
        wasReadOnly = model.isReadOnly()
        model.setReadOnly(False)
        model.setData(index, name)
        model.setReadOnly(wasReadOnly)

    def get_selected_path(self, index):
        """ for test """
        if not index.isValid():
            return
        model = index.model()
        path = model.fileInfo(index).absoluteFilePath()

        if self.radio_before.isChecked() :
             self.file_before = path

        if self.radio_after.isChecked() :
             self.file_after = path
             
        print(' path =',path)

    def path_end(self, location):
        if os.path.exists(location) == True:
            sample_pth = "/".join([x for x in location.split('/')[-3:]])
            return sample_pth
        else:
            return location
        
    def update_Amin(self):
        msg= 'Amin = ' + str(self.ui.threshold_slider.value())
        Min_object_area = self.ui.Amin_label.setText(msg)


    def run_full_inspection_A(self):
        # Get the parameter
        Min_object_area = self.ui.threshold_slider.value()

        msg= 'Running the algorithm :'+ str(self.algorithm.currentText()) + ' with Amin=' + str(Min_object_area)
        print(msg); self.ui.msg_label.setText(msg)

        if  self.type_algo == '2D' :
            # Timing start
            tic = time.perf_counter()

            # img_after =  self.get_image(self.file_after)
            # img_before = self.get_image(self.file_before)
            
            ## run registration of 2D images 
            print('\n - run registration of the after image')
            img_before, img_after0,img_after, rot_angle = Jutil_2D_registration.Image_registration([self.file_before], [self.file_after])
            # Plot results
            import numpy as np
            from PIL import Image
            im = Image.fromarray(img_before); filename = "files/registration.jpg"; im.save(filename)
            self.Plot_image( filename, self.plot_1)

            im = Image.fromarray(img_after); filename = "files/registration.jpg"; im.save(filename)
            self.Plot_image( filename, self.plot_2)
            
            ## run the algorithm using 2D images 
            img_box, nb_box, mask = Autils_fault_detection_DSP.fault_detection_algorithm1(img_after, img_before , Min_object_area=Min_object_area)
            # Timing end
            toc = time.perf_counter() - tic ; toc = int(toc*100)/100
            msg=  '- Amin=' + str(Min_object_area) + \
                '\n- number of fault detected = ' + str(nb_box) + '  ! ' \
                '\n- Execution time = ' + str(toc) + ' seconds'

            if nb_box <=0 :
                msg_out = 'The tool is defect-free!'

            elif nb_box==1:
                msg_out = 'The tool is a defective tool' + \
                           '\n- ' + str(nb_box) + ' fault is detected! ' 

            else: 
                msg_out = 'The tool is a defective tool' + \
                           '\n- ' + str(nb_box) + ' fault are detected! '             
                           
                           
            # Plot results
            import numpy as np
            im = Image.fromarray(img_box); filename = "files/detection.jpg"; im.save(filename)
            self.Plot_image( filename, self.plot_3)

            self.output_dic={"img_type":'2D',"img_after": img_after, "img_box":img_box,"nb_box":nb_box, "mask":mask, "rot_angle":rot_angle}

        elif self.type_algo == '3D':
            Min_object_area=100; mask_th=0.5; tol_list = [6,4,2]; fault_ratio_th=0.1
            # Timing start
            tic = time.perf_counter()
            # Preparation
            img3D_ref_prep, img3D_faulty_prep = Autils_fault_detection_DSP_3D.preprocessing_checklist(self.file_before, self.file_after, default=True)
            # Registration [Shift]
            img3D_faulty_prep = Autils_fault_detection_DSP_3D.registration_3D_shift(img3D_ref_prep, img3D_faulty_prep)
            
            # # Load preprocesse data
            # img3D_ref_prep, _ = self.get_volume(self.file_before)
            # img3D_faulty_prep, _ = self.get_volume(self.file_after)

            # Run the algorithm using 3D volume
            save_folder = 'output/'

            mask_fault = Autils_fault_detection_DSP_3D.Run_fault_detection_algorithm1_3D(img3D_ref_prep, img3D_faulty_prep, Min_object_area, mask_th=mask_th, save_folder= save_folder)
            print('\n\n\nsegment= ', np.unique(mask_fault))
 
            # Timing end
            toc = time.perf_counter() - tic
            toc = int(toc*100)/100

            # save the make results
            Autils_fault_detection_DSP_3D.save_3D_masks(mask_fault, save_folder)
            print('mask_fault size=', mask_fault.shape)
            # Load image
            fault_cathegory, fault_names = Autils.categorize_faults(mask_fault, tol_list, method=1)
            fault_cathegory.insert(0,img3D_faulty_prep)
            fault_names.insert(0,'Input scan')
            # visualize the results
            Autils.napari_view_volume(fault_cathegory, fault_names, inspection=True)
            # quatify the faulty error
            fault_size, msg, predicted_class = Autils_fault_detection_DSP_3D.quatify_fault(img3D_faulty_prep, mask_fault,fault_ratio_th=fault_ratio_th)
            mask_fault_segments = np.unique(mask_fault)
            msg =  msg + '\n- Amin=' + str(Min_object_area) + \
                    '\n- fault_size=' + str(fault_size) + \
                    '\n- predicted_class=' + str(predicted_class) + \
                    '\n- mask_fault_segments=' + str(mask_fault_segments) + \
                    '\n- Execution time = ' + str(toc) + ' seconds' + msg

                    
            if fault_size <=0 :
                msg_out = 'The tool is defect-free!'

            else: 
                msg_out = 'The tool is a defective tool' + \
                           '\n\n More details: \n- ' + str(fault_size) + ' %' + ' of  its volume is diagnosed to be  defective! ' 

     

            # # # Plot results
            # mask_fault_vtk_vol = Autils.get_vtk_volume_from_3d_array(mask_fault)
            # tool_vtk_vol = Autils.get_vtk_volume_from_3d_array(img3D_faulty_prep)

            # # mask_segments = np.array(np.unique(mask_fault.ravel()))
            # # self.Plot_volume(self.vtk_widget_3, mask_fault_vtk_vol, volume_segments=mask_segments)

            # print('\n\n\nsegment= ', np.unique(mask_fault))

            # self.vtk_widget_3.disp_fault_volume(tool_vtk_vol, mask_fault_vtk_vol)
            # self.vtk_widget_3.start()

    
            # # charachterization 
            # import nrrd
            # mask_fault= 'data/fault_volume.nrrd'
            # # nrrd.write(fault_volume, mask_fault)

            # # Matthew: get these paths from DB
            # volume_masks='data/Volume_mask.nrrd'
            # component_name_file= 'data/label_table.txt'

            # fault_list=Jutils_3D_characterization.fault_characteriziation(volume_masks, mask_fault, component_name_file )

            # msg_out = msg_out + '\n\n- The fault charachteristics :\n'
            # for d in fault_list :
            #     # msg_out = msg_out + '\n    -> ' + d[:-2] + ' is defective (' +  str(int( fault_list[d]) )   + ' %)'
            #     msg_out = msg_out + '\n    -> ' + d[:-2] + ' is missing'

            # self.output_dic={"img_type":'3D',"img_after": self.file_after, "fault_list":fault_list, "mask_fault":mask_fault}



        # disply results and performance
        msg = msg + self.msg 
        print(msg);self.ui.msg_label.setText(msg)
        self.Label_outputs.setText(msg_out)

    def run_DSP_algorithm_A(self):
        # Get the parameter
        Min_object_area = self.ui.threshold_slider.value()
        msg= 'Running the algorithm :'+ str(self.algorithm.currentText()) + ' with Amin=' + str(Min_object_area)
        print(msg); self.ui.msg_label.setText(msg)

        if  self.type_algo == '2D' :
            # Timing start
            tic = time.perf_counter()

            # img_after =  self.get_image(self.file_after)
            # img_before = self.get_image(self.file_before)
            
            ## run registration of 2D images 
            print('\n - run registration of the after image')
            img_before, img_after0,img_after, rot_angle = Jutil_2D_registration.Image_registration([self.file_before], [self.file_after])
            # Plot results
            import numpy as np
            from PIL import Image
            im = Image.fromarray(img_before); filename = "files/registration.jpg"; im.save(filename)
            self.Plot_image(filename, self.plot_1)

            im = Image.fromarray(img_after); filename = "files/registration.jpg"; im.save(filename)
            self.Plot_image(filename, self.plot_2)
            
            ## run the algorithm using 2D images 
            img_box, nb_box, mask = Autils_fault_detection_DSP.fault_detection_algorithm1(img_after, img_before , Min_object_area=Min_object_area)
            # Timing end
            toc = time.perf_counter() - tic ; toc = int(toc*100)/100
            msg=  '- Amin=' + str(Min_object_area) + \
                '\n- number of fault detected = ' + str(nb_box) + '  ! ' \
                '\n- Execution time = ' + str(toc) + ' seconds'

            if nb_box <=0 :
                msg_out = 'The tool is defect-free!'

            elif nb_box==1:
                msg_out = 'The tool is a defective tool' + \
                           '\n- ' + str(nb_box) + ' fault is detected! ' 

            else: 
                msg_out = 'The tool is a defective tool' + \
                           '\n- ' + str(nb_box) + ' fault are detected! '             
                           
                           
            # Plot results
            import numpy as np
            im = Image.fromarray(img_box); filename = "files/detection.jpg"; im.save(filename)
            self.Plot_image( filename, self.plot_3)

            self.output_dic={"img_type":'2D',"img_after": img_after, "img_box":img_box,"nb_box":nb_box, "mask":mask, "rot_angle":rot_angle}



        elif self.type_algo == '3D':

            mask_th=0.5; tol_list = [6,4,2]; fault_ratio_th=0.1
            # Timing start
            tic = time.perf_counter()
            # # Preparation
            img3D_ref_prep, img3D_faulty_prep = Autils_fault_detection_DSP_3D.preprocessing_checklist(self.file_before, self.file_after, default=True)
            # # Registration [Shift]
            img3D_faulty_prep = Autils_fault_detection_DSP_3D.registration_3D_shift(img3D_ref_prep, img3D_faulty_prep)

            # Run the algorithm using 3D volume
            save_folder = 'output/'
            # img3D_ref_prep, _ = self.get_volume(self.file_before)
            # img3D_faulty_prep, _ = self.get_volume(self.file_after)
            print( ' volume ', img3D_ref_prep)
            print( ' volume type', type(img3D_ref_prep))

            mask_fault = Autils_fault_detection_DSP_3D.Run_fault_detection_algorithm1_3D(img3D_ref_prep, img3D_faulty_prep, Min_object_area, mask_th=mask_th, save_folder= save_folder)

            # Timing end
            toc = time.perf_counter() - tic
            toc = int(toc*100)/100

            # save the make results
            Autils_fault_detection_DSP_3D.save_3D_masks(mask_fault, save_folder)
            print('mask_fault size=', mask_fault.shape)

            # Load image
            fault_cathegory, fault_names = Autils.categorize_faults(mask_fault, tol_list, method=1)
            fault_cathegory.insert(0,img3D_faulty_prep)
            fault_names.insert(0,'Input scan')
            # visualize the results
            Autils.napari_view_volume(fault_cathegory, fault_names, inspection=True)

            # quatify the faulty error
            fault_size, msg, predicted_class  = Autils_fault_detection_DSP_3D.quatify_fault(img3D_faulty_prep, mask_fault,fault_ratio_th=fault_ratio_th)
            mask_fault_segments = np.unique(mask_fault)
            msg =  '- Amin=' + str(Min_object_area) + \
                    '\n- fault_size=' + str(fault_size) + \
                    '\n- predicted_class=' + str(predicted_class) + \
                    '\n- mask_fault_segments=' + str(mask_fault_segments) + \
                    '\n- Execution time = ' + str(toc) + ' seconds' + msg

            if fault_size <=0 :
                msg_out = 'The tool is defect-free!'

            else: 
                msg_out = 'The tool is a defective tool' + \
                           '\n- ' + str(fault_size) + ' %' + ' of  its volume is diagnosed to be  defective! ' 

            mask_fault_vtk_vol = Autils.get_vtk_volume_from_3d_array(mask_fault)
            tool_vtk_vol = Autils.get_vtk_volume_from_3d_array(img3D_faulty_prep)

            # Plot results
            mask_fault_vtk_vol = Autils.get_vtk_volume_from_3d_array(mask_fault)
            tool_vtk_vol = Autils.get_vtk_volume_from_3d_array(img3D_faulty_prep)

            self.vtk_widget_3.disp_fault_volume(tool_vtk_vol, mask_fault_vtk_vol)
            self.vtk_widget_3.start()

            self.output_dic={"img_type":'3D',"img_after": self.file_after,  "mask_fault":mask_fault}


        # disply results and performance
        msg = msg + self.msg 
        print(msg);self.ui.msg_label.setText(msg)

        self.Label_outputs.setText(msg_out)

    def run_CNN_Object_detection_A(self):

        # Get the parameter
        msg= 'Running the localization algorithm   :'+ str(self.algorithm.currentText()) 
        print(msg); self.ui.msg_label.setText(msg)

        if  self.type_algo == '2D' :

            # Timing start
            tic = time.perf_counter()
            ## run the algorithmusing 2D images 
            img_box, nb1_box = 1,0 # function 
            # Timing end
            toc = time.perf_counter() - tic ; toc = int(toc*100)/100
            msg= '\n- number of fault detected = ' + str(nb1_box) + '  ! ' \
                '\n- Execution time = ' + str(toc) + ' seconds'

            # Plot results
            import numpy as np
            from PIL import Image
            im = Image.fromarray(img_box); filename = "files/detection.jpg"; im.save(filename)
            self.Plot_image( filename, self.plot_3)

        elif self.type_algo == '3D':
            # Timing start
            tic = time.perf_counter()
                    # Run the algorithm using 3D volume
            mask_fault = 2#

            # Timing end
            toc = time.perf_counter() - tic
            toc = int(toc*100)/100

        
            msg= '\n- Execution time = ' + str(toc) + ' seconds'

            # Plot results

    
        # disply results and performance
        msg = msg + self.msg 
        print(msg);self.ui.msg_label.setText(msg)

    def run_registration_DIPY_J(self):
        # Get the parameter
        param =  '' #self.ui.threshold_slider.value()
        msg= 'Running the algorithm :'+ str(self.algorithm.currentText()) + ' with Amin=' + str(param)
        print(msg); self.ui.msg_label.setText(msg)

        if self.type_algo == '2D' :

            # Timing start
            tic = time.perf_counter()
            ## run the algorithm using 2D images 


            # Run the  fault detection algorithm1 using 2D images
            img_before, img_after,img_after_reg, rot_angle = Jutil_2D_registration.Image_registration([self.file_before], [self.file_after])
            # Timing end
            toc = time.perf_counter() - tic ; toc = int(toc*100)/100
            msg=  '- param=' + str(param) + \
                '\n- Execution time = ' + str(toc) + ' seconds'

            # Plot results
            import numpy as np
            from PIL import Image
            im = Image.fromarray(img_after_reg); filename = "files/registration.jpg"; im.save(filename)
            self.Plot_image( filename, self.plot_3)
            
            # self.Label_outputs.setText('Registration completed.')


        elif self.type_algo == '3D':
            # Timing start
            tic = time.perf_counter()
            ## run the algorithm using 3D volume 
            model_path = 'files/UNET-CTSegmentation_496_496.h5'


            # Run the registratin algorithm
            volume_before, vol_reg, mse = Jutils_3D_registration.Volume_registration(self.file_before, self.file_after, model_path)
            # Timing end
            toc = time.perf_counter() - tic
            toc = int(toc*100)/100

            if self.file_after[-4:]=='.nii':
                # vol_reg = vol_reg.get_data()
                vol_reg = vol_reg

            
            vol_reg_vtk_vol = Autils.get_vtk_volume_from_3d_array(vol_reg)

            msg=  '- param=' + str(param) + \
                '\n- Execution time = ' + str(toc) + ' seconds'

            # Plot results
            vol_reg_vtk_vol = Autils.get_vtk_volume_from_3d_array(vol_reg)
    
            self.Plot_volume(self.vtk_widget_3, vol_reg_vtk_vol)

            # self.Label_outputs.setText('Registration completed.')


    def algorithm_inputs_param(self):

        self.Label_outputs.setText('')

        if self.file_image_or_volume(self.file_before)=='3D' and self.file_image_or_volume(self.file_after)=='3D' :
            self.type_algo = '3D'
                
        elif self.file_image_or_volume(self.file_before)=='2D' and self.file_image_or_volume(self.file_after)=='2D' :
            self.type_algo = '2D'
            
        else: 
            msg=  '- The selected data is not compatible!!:' + \
                '\n- Data before = ' + self.file_before + \
                '\n- Data after = ' + self.file_after 

            msg = msg + self.msg 
            print(msg);self.ui.msg_label.setText(msg)

        # self.type_algo can be updated from model table in database
        algo_name = self.algorithm.currentText()
        self.type_data = sqlcon.get_model_typedata(algo_name)
        if self.type_data != self.type_algo:
                print("Different data dimension and model input dimensions")
        else:
            print("Data dimension and model input dimension matches!")
    
    def new_algorithm_G(self):
        #inputs: 
        model_path = self.root_localization  + self.algorithm.currentText() + '/' +  self.model_path_list.currentText()
        print('inputs:', [self.file_before, self.file_after, self.model_path_list])
        print('model_path:',model_path)

        # Timing start
        tic = time.perf_counter()
        ## run the algorithm using 3D volume 
        run_function = 1
        # Timing end
        toc = time.perf_counter() - tic
        toc = int(toc*100)/100

        msg=  '- model_path=' + str(model_path) + \
                '\n- Execution time = ' + str(toc) + ' seconds'


        msg_out=  '- the decive is defective with missing part \n - dflkl;dk;ld \n - dlkfjlkdflkseconds' 
        
        
        # disply results instruction display
        msg = msg + self.msg 
        print(msg);self.ui.msg_label.setText(msg)
        # disply results tool diagnosis display
        self.Label_outputs.setText(msg_out)


    def run_localisation(self):

        msg ='. The algorithm ' + self.algorithm.currentText() + ' is running ...\nIt take few minutes :):)'  
        print(msg);self.ui.msg_label.setText(msg)

        # get the algorithm inputs and parameters
        self.algorithm_inputs_param()

        # fault localization
        if self.algorithm.currentText() in ['2D-1-DSP-REG', '2D-2-DIPY-REG']:
            self.run_DSP_algorithm_A()

        elif self.algorithm.currentText() == '2-CNN':
            self.run_CNN_Object_detection_A()

        elif self.algorithm.currentText() == '3-new-algorithm':
            self.new_algorithm_G()  

        else:
            msg=  'Error : algorithm undefined : ' + self.algorithm.currentText() + '\n' + self.msg
            print(msg); self.ui.msg_label.setText(msg)
            return 0

        msg=  'Algorithm excuted successfully : '+ self.algorithm.currentText() + '\n' + self.msg             
        print(msg); self.ui.msg_label.setText(msg)
        
        # show the verification message
        self.verify_save.show()

        return 1
    
    def Run_algorithms(self):

        msg ='. The algorithm ' + self.algorithm.currentText() + ' is running ...\nIt take few minutes :):)'  
        print(msg);self.ui.msg_label.setText(msg)

        # get the algorithm inputs and parameters
        self.algorithm_inputs_param()

#----------------------------------------------# fault localization #----------------------------------------------#
        if self.current_algorithm == 2: # fault localization

            if self.algorithm.currentText() == '1-DSP':
                self.run_DSP_algorithm_A()

            elif self.algorithm.currentText() == '2-CNN':
                self.run_CNN_Object_detection_A()

            elif self.algorithm.currentText() == '3-new-algorithm':
                self.new_algorithm_G()  

#----------------------------------------------# fault characterization #----------------------------------------------#
            
        elif self.current_algorithm == 3:# fault characterization
            self.run_full_inspection_A()
#----------------------------------------------# Registration #----------------------------------------------#

        elif self.current_algorithm == -1: # Registration
            # self.Label_outputs.setText('Registering the image/volume data.')

            if self.algorithm.currentText() == '1- DIPY':

                self.run_registration_DIPY_J()


        else:
            msg=  'Error : algorithm undefined : ' + self.algorithm.currentText() + '\n' + self.msg
            print(msg); self.ui.msg_label.setText(msg)
            return 0

        msg=  'Algorithm excuted successfully : '+ self.algorithm.currentText() + '\n' + self.msg             
        print(msg); self.ui.msg_label.setText(msg)

        
        # show the verification message
        self.verify_save.show()

        return 1 

## RUN CT inspection algorithms  
class AutoInspectionApp(InspectionApp):
    def __init__(self):
        #Parent constructor
        super(AutoInspectionApp,self).__init__()
        self.config_algo = 0
        self.ui = None
        self.setup()
        self.filename =  None
        self.createMenu()
        self.windows_format()
        self.plot_sample_image()

    def setup(self):
        
        import UI.RunAutomated_frameworkUI 
        self.ui = UI.RunAutomated_frameworkUI.Ui_Run_inspection()
        self.ui.setupUi(self)
        self.scan_path = ''
        self.type_algo = '3D'
        self.file_before = 'files/volume_before.rek'
        self.file_after = 'files/volume_after.rek'

        self.msg ='\n\n - Please follow these steps: \n 1- Select the appropriate CT data.  \n 2- Selct the desired algorithm  \n 3- Click Run Algorithm'
        self.ui.msg_label.setText(self.msg)
        self.verify_save = self.ui.verify_save
        self.verify_save.clicked.connect(self.Verify_Save_KnowledgeBase) 
        self.verify_save.hide()

        self.plot_1 = self.ui.Label_plot_1
        self.plot_2 = self.ui.Label_plot_2
        self.plot_3 = self.ui.Label_plot_3
        self.Label_outputs = self.ui.Label_outputs

        self.vtk_plot_1 = self.ui.Frame_plot_1
        self.vtk_plot_2 = self.ui.Frame_plot_2
        self.vtk_plot_3 = self.ui.Frame_plot_3

        # Initialize the 3D plots
        path0 ='files/volume_before.rek'
        self.vtk_widget_1, self.vtk_plot_1 = self.create_vtk_widget(self.vtk_widget_1, self.vtk_plot_1, path0)
        self.plot_3D_volume(path0, self.vtk_widget_1)
        path0 ='files/volume_after.rek'
        self.vtk_widget_2, self.vtk_plot_2 = self.create_vtk_widget(self.vtk_widget_2, self.vtk_plot_2, path0)
        self.plot_3D_volume(path0, self.vtk_widget_2)
        path0 ='files/volume_after0.rek'
        self.vtk_widget_3, self.vtk_plot_3 = self.create_vtk_widget(self.vtk_widget_3, self.vtk_plot_3, path0)
        self.plot_3D_volume(path0, self.vtk_widget_3)



        # Show 3D Volume first
        self.vtk_plot_1.show();self.vtk_plot_2.show();self.vtk_plot_3.show()
        self.plot_1.hide();self.plot_2.hide();self.plot_3.hide()
        
        # Algorithm/Model
        self.select_algo = '3D'
        self.select_model = ''

        # adding list of available tools (from DB)
        self.tool_list = self.ui.tool_list
        self.tool_list.currentIndexChanged.connect(self.load_selected_tool_scans)

        self.scan_list = self.ui.scan_list
        self.scan_list.currentIndexChanged.connect(self.load_select_scan_data_volume)
        self.ui.pushButton_3.clicked.connect(self.run_auto_inspection)     

        # select type input
        self.radio_after = self.ui.radio_after
        self.radio_before = self.ui.radio_before
        self.radio_before.toggled.connect(self.radio_change_value)
        self.radio_after.toggled.connect(self.radio_change_value)
        self.radio_before.setChecked(1)

        # root folder
        self.path = None
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.root_folder)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.root_folder))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.upload_selected_dataset_item(index))
        self.treeView.doubleClicked.connect(self.get_select_folder)   

    def load_selected_tool_scans(self):
        self.radio_before.setChecked(1)
        tool_name = self.tool_list.currentText()
        #self.treeView.setRootIndex(self.treeModel.index(self.root_folder + tool_name))
        self.scan_path = self.root_folder + tool_name
        
        self.scan_list.clear()
        if tool_name != "":
            scan_names = (sqlcon.get_all_tool_scan(tool_name))
            self.scan_list.addItems(scan_names)

    def radio_change_value(self):

        self.verify_save.hide()
        tool_name = self.tool_list.currentText()
        before_after = ['after', 'before']
        scan_names = []
        if self.radio_before.isChecked():
             self.scan_list.clear()
             if tool_name != "":
                before_after_data = sqlcon.get_scan_before_after(tool_name)
                scan_samples = [x + " " + '[' + before_after[int(y)] + ']' for x, y in zip(before_after_data['scan_name'], before_after_data['before_after_status'])]
                scan_before = [x for x in scan_samples if 'before' in x]
                scan_names.extend(scan_before)
                    
        elif self.radio_after.isChecked():
            self.scan_list.clear()
            if tool_name != "":
                before_after_data = sqlcon.get_scan_before_after(tool_name)
                scan_samples = [x + " " + '[' + before_after[int(y)] + ']' for x, y in zip(before_after_data['scan_name'], before_after_data['before_after_status'])]
                scan_after = [x for x in scan_samples if 'after' in x]
                scan_names.extend(scan_after)        
        self.scan_list.addItems(scan_names)

    def load_select_scan_data_volume(self):
        self.verify_save.hide()
        tool_name = self.tool_list.currentText()
        scan_name = self.scan_list.currentText()
        #self.treeView.setRootIndex(self.treeModel.index(self.root_folder + tool_name))

        if scan_name == "":
            scan_name = sqlcon.get_first_scan_name(tool_name)
            if scan_name == "":
                self.scan_path = self.root_folder + tool_name
                self.treeView.setRootIndex(self.treeModel.index(self.scan_path))
                self.plot_3D_empty()
                return

        # plot the loaded scan data
        self.scan_path = self.root_folder + tool_name + '/' + scan_name
        self.scan_path = self.scan_path.removesuffix(' [before]')
        self.scan_path = self.scan_path.removesuffix(' [after]')
        print("3D selected scan path:", self.root_folder, tool_name, scan_name, self.scan_path)
         
        # update the data disply root
        self.update_data_root()
        # Plot the selected scan volume
        self.plot_loaded_scan()

    def update_data_root(self):
        # update the data disply root 
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.scan_path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.scan_path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        #self.treeView.setRootIndex(self.treeModel.index(self.path))

    def plot_loaded_scan(self):
        # plot the 3D volume of the scan     
        print('\n\n\n\n self.scan_path: %s'%(self.scan_path))

        if self.radio_before.isChecked():
            # Getting scan volume path 
            self.file_before = self.get_scan_volume_path(self.scan_path)
            self.plot_3D_volume(self.file_before, self.vtk_widget_1)
        else: 
            self.file_after = self.get_scan_volume_path(self.scan_path)
            # print('\n\n\n\nfile after: %s  '%( self.file_after))
            self.plot_3D_volume(self.file_after, self.vtk_widget_2)

    def get_scan_volume_path(self, scan_folder):
        # get the volume:
        msg = ''
        from glob import glob
        path = glob(os.path.join(scan_folder, 'volume.*')  )[0]
        print('\n\nGetting volume_path =', path )

        if os.path.isfile(path): 
            self.select_algo = self.file_image_or_volume(path)
        else:
            msg=  '- Warning (1):  You found file is not a 3D volume!!\n - ' + self.path_end(path)
        msg = msg + self.msg
        print(msg);self.ui.msg_label.setText(msg)
        return path
    
    def plot_3D_volume(self, path, vtk_widget):
        # display the selected Rek volume 
        volume_arr, vol0  = self.get_volume(path)
        volume_segments = np.array(np.unique(volume_arr.ravel()))
        self.Plot_volume(vtk_widget , vol0, volume_segments=volume_segments)
        print(' vizualized volume path: ', path)

    def plot_3D_empty(self):
        path0 ='files/volume_before.rek'
        path1 ='files/volume_after.rek'
        path2 ='files/volume_after0.rek'
        self.plot_3D_volume(path0, self.vtk_widget_1)
        self.plot_3D_volume(path1, self.vtk_widget_2)
        self.plot_3D_volume(path2, self.vtk_widget_3)
        print("Cleaning up visualized 3D volumes")
                  
    def run_auto_inspection(self):

        ## Automated framework
        # output_dic_dsp = self.run_DSP_flowchart()
        # print('Final Result:')
        # print(output_dic_dsp)

        # flag Matthew: labeled_data availability should be got from the DB
        # need function to check if CAD available for the tool, if avilable make labelled data True, otherwise false. 
        labeled_data  = sqlcon.get_cad_avail_flag(self.tool_list.currentText())

        if labeled_data == True:
            output_dic_DNN = self.run_DNN_flowchart()
            print('Final Result:')
            print(output_dic_DNN)

        else:
            ## Automated framework
            output_dic_dsp = self.run_DSP_flowchart()
            print('Final Result:')
            print(output_dic_dsp)
        


        # show the verification message
        self.verify_save.show()

    def run_DSP_flowchart(self):
        # show the verification message
        if self.file_image_or_volume(self.file_before)=='3D' and self.file_image_or_volume(self.file_after)=='3D' :
            self.type_algo = '3D'
        else:
            self.type_algo = '2D' 
        
        # Method 1 : DSP A
        output_dic = self.run_DSP_algorithm_A()
        # flag : add DSP detection methods

        return output_dic

    def run_DSP_algorithm_A(self):
        print('\n\n\n\n file before: %s \nfile after: %s '%(self.file_before, self.file_after))
        # Get the parameter
        Min_object_area=100
        mask_th=0.5; tol_list = [6,4,2]; fault_ratio_th=0.1

        msg= 'Running the ' + self.type_algo + '[DSP algorithm  with Amin=' + str(Min_object_area) + ']'
        print(msg); self.ui.msg_label.setText(msg)

        # Timing start
        tic = time.perf_counter()
        # Run the algorithm using 3D volume
        img3D_ref_prep, _ = self.get_volume(self.file_before)
        img3D_faulty_prep, _ = self.get_volume(self.file_after)

        #results distination
        save_folder =  'output/' 
        # Run the  fault detection algorithm1 using 3D volume
        mask_fault = Autils_fault_detection_DSP_3D.Run_fault_detection_algorithm1_3D(img3D_ref_prep, img3D_faulty_prep,\
                                            Min_object_area, mask_th=mask_th, denoise=False, disp=False, save_folder= save_folder)



        # # show in Napari
        # #visualise 
        # viewer = Autils_fault_detection_DSP_3D.napari_viewer(img3D_ref_prep[:,2:-2,:], img3D_faulty_prep[:,2:-2,:])
        # viewer.add_image(mask_fault[:,2:-2,:], name='Inspection results ', colormap='red', opacity=0.6)

        # Timing end
        toc = time.perf_counter() - tic
        toc = int(toc*100)/100


        # save the make results
        Autils_fault_detection_DSP_3D.save_3D_masks(mask_fault, save_folder)
        print('mask_fault size=', mask_fault.shape)
        # Load image
        fault_cathegory, fault_names = Autils.categorize_faults(mask_fault, tol_list, method=1)
        fault_cathegory.insert(0,img3D_faulty_prep)
        fault_names.insert(0,'Input scan')
        # visualize the results
        Autils.napari_view_volume(fault_cathegory, fault_names, inspection=True)
        # quatify the faulty error
        fault_size, msg_out, predicted_class = Autils_fault_detection_DSP_3D.quatify_fault(img3D_faulty_prep, mask_fault,fault_ratio_th=fault_ratio_th)
        mask_fault_segments = np.unique(mask_fault)
        msg =  msg + '\n- Amin=' + str(Min_object_area) + \
                '\n- fault_size=' + str(fault_size) + \
                '\n- predicted_class=' + str(predicted_class) + \
                '\n- mask_fault_segments=' + str(mask_fault_segments) + \
                '\n- Execution time = ' + str(toc) + ' seconds' + msg

        msg=  '- Amin=' + str(Min_object_area) + \
            '\n- Execution time = ' + str(toc) + ' seconds'


        #fault characterization
        # volume_masks='data/demo_data/mask_front.nrrd'
        # component_name_file= 'data/demo_data/label_table.txt'
        # fault_list=Jutils_3D_characterization.fault_characteriziation(volume_masks, mask_fault, component_name_file)
        # msg_out=msg_out+'\n\nMissing Parts:\n'

        # percentage_threshold=25
        # for part_name, percentage in fault_list.items():
        #     if percentage>percentage_threshold:
        #         msg_out=msg_out+'\n'+part_name

        # # Save/Plot the 3D fault volume results
        # output_volume_path = os.path.join(save_folder , 'fault_volume.nrrd')
        # import nrrd
        # nrrd.write(output_volume_path, mask_fault)
        # self.plot_3D_volume(output_volume_path, self.vtk_widget_3)

        # Plot the mask fault volume results
        mask_fault_vtk_vol = Autils.get_vtk_volume_from_3d_array(mask_fault)
        tool_vtk_vol = Autils.get_vtk_volume_from_3d_array(img3D_faulty_prep)

        mask_fault_vtk_vol = Autils.get_vtk_volume_from_3d_array(mask_fault)
        tool_vtk_vol = Autils.get_vtk_volume_from_3d_array(img3D_faulty_prep)

        self.Vizualize_3D_inspection_fault(self.vtk_widget_3, tool_vtk_vol, mask_fault_vtk_vol, mask_fault_segments)
        self.output_dic={"img_type":'3D',"img_after": self.file_after,  "mask_fault":mask_fault}

        # disply results and performance
        msg = msg + self.msg 
        print(msg);self.ui.msg_label.setText(msg)
        self.Label_outputs.setText(msg_out)

        output_dic = {'report': msg_out}

        return output_dic
    

    def run_DNN_flowchart(self):
        # run predictions

        # Need to develop function
        # predicted_class = self.run_prediction_flowchart()
        predicted_class = ''

        # run Localization
        # faulty_mask = self.run_localization_flowchart()
        faulty_mask = ''


        # run Charachterization
        faulty_components_df = self.run_charachterization_flowchart()

        output_dic = {'report': 'The tool is diagnosed to be ' + predicted_class  + ' + results summary {  prediction, localaization, charachterization} ',\
                            'fault_list':faulty_components_df, 'suggestion_to_technician': 'please recheck these parts ....',\
                            'incremental_leaning': 'do we need futher training:Yes'}

        return output_dic

    def run_prediction_flowchart(self):
        # run predictions

        # predicted_class='defect-free'
        # flag : run prediction flowchart
        predicted_class = self.run_prediction_algorithm()

        return predicted_class  # 'defect-free'/'defective'

    def run_prediction_algorithm(self):

        # Get the parameter
        Min_object_area = 100
        mask_th = 0.3

        msg= 'Running the ' + self.type_algo + '[DSP algorithm  with Amin=' + str(Min_object_area) + ']'
        print(msg); self.ui.msg_label.setText(msg)

        # Run the algorithm using 3D volume
        img3D_ref_prep, _ = self.get_volume(self.file_before)
        img3D_faulty_prep, _ = self.get_volume(self.file_after)

        #results distination
        save_folder =  'output/' 
        # Run the  fault detection algorithm1 using 3D volume
        mask_fault = Autils_fault_detection_DSP_3D.Run_fault_detection_algorithm1_3D(img3D_ref_prep, img3D_faulty_prep,\
                                            Min_object_area, mask_th=mask_th, denoise=False, disp=False, save_folder= save_folder)

        # quatify the faulty error
        fault_size = self.quatify_fault(img3D_faulty_prep, mask_fault)
        mask_fault_segments = np.unique(mask_fault)
        predicted_class=''

        if fault_size <=0 :
            msg_out = 'The tool is defect-free!'
            predicted_class='defect-free'

        else: 
            msg_out = 'The tool is a defective tool' + \
                        '\n- ' + str(fault_size) + ' %' + ' of  its volume is diagnosed to be  defective! ' 
            predicted_class='defective'


        return predicted_class

    def run_localization_flowchart(self):
        faulty_mask= self.run_localization_algorithm()
        # flag : run localization flowchart
        #example_faulty_components = self.run_localization_algorithm()
        return faulty_mask

    def run_localization_algorithm(self):

        print('\n\n\n\n file before: %s \nfile after: %s '%(self.file_before, self.file_after))
        # Get the parameter
        Min_object_area = 100
        mask_th = 0.3

        msg= 'Running the ' + self.type_algo + '[DSP algorithm  with Amin=' + str(Min_object_area) + ']'
        print(msg); self.ui.msg_label.setText(msg)

        # Run the algorithm using 3D volume
        img3D_ref_prep, _ = self.get_volume(self.file_before)
        img3D_faulty_prep, _ = self.get_volume(self.file_after)

        #results distination
        save_folder =  'output/' 
        # Run the  fault detection algorithm1 using 3D volume
        mask_fault = Autils_fault_detection_DSP_3D.Run_fault_detection_algorithm1_3D(img3D_ref_prep, img3D_faulty_prep,\
                                            Min_object_area, mask_th=mask_th, denoise=False, disp=False, save_folder= save_folder)

        # show in Napari
        #visualise 
        # viewer = Autils_fault_detection_DSP_3D.napari_viewer(img3D_ref_prep[:,2:-2,:], img3D_faulty_prep[:,2:-2,:])
        # viewer.add_image(mask_fault[:,2:-2,:], name='Inspection results ', colormap='red', opacity=0.6)

        return mask_fault







    def run_charachterization_flowchart(self):
        # risk_priority: is the state level of in the risk priority tree
        # location_centroid: id the centroid of the component
        columns = ['name', 'risk_priority', 'type_fault', 'percentage', 'location_centroid']
        # example_faulty_components = [ ['screw', 0, 'missing', 80, (12,25,36)], ['disk',21 ,'defect-free', 100,(65,704,273)], ['disk',21 ,'missplaced', 40,(695,74,23)], ['disk',3, 'scratch', 0.2,(12,25,78)]]
        # flag : run Charachterization flowchart
        example_faulty_components = self.run_charachterization_algorithm()
        import pandas as pd
        faulty_components_df = pd.DataFrame(example_faulty_components, columns=columns)
        print(faulty_components_df)
        return faulty_components_df 
    
    def run_charachterization_algorithm(self):

        # # Get the parameter
        # Min_object_area = 100
        # mask_th = 0.3

        # msg= 'Running the ' + self.type_algo + '[DSP algorithm  with Amin=' + str(Min_object_area) + ']'
        # print(msg); self.ui.msg_label.setText(msg)

        # # Timing start
        # tic = time.perf_counter()
        # # Run the algorithm using 3D volume
        # img3D_ref_prep, _ = self.get_volume(self.file_before)
        # img3D_faulty_prep, _ = self.get_volume(self.file_after)

        # #results distination
        # save_folder =  'output/' 
        # # Run the  fault detection algorithm1 using 3D volume
        # mask_fault = Autils_fault_detection_DSP_3D.Run_fault_detection_algorithm1_3D(img3D_ref_prep, img3D_faulty_prep,\
        #                                     Min_object_area, mask_th=mask_th, denoise=False, disp=False, save_folder= save_folder)



        # # show in Napari
        # #visualise 
        # viewer = Autils_fault_detection_DSP_3D.napari_viewer(img3D_ref_prep[:,2:-2,:], img3D_faulty_prep[:,2:-2,:])
        # viewer.add_image(mask_fault[:,2:-2,:], name='Inspection results ', colormap='red', opacity=0.6)

        # # Timing end
        # toc = time.perf_counter() - tic
        # toc = int(toc*100)/100

        # # quatify the faulty error
        # fault_size = self.quatify_fault(img3D_faulty_prep, mask_fault)
        # mask_fault_segments = np.unique(mask_fault)
        # predicted_class=''

        # if fault_size <=0 :
        #     msg_out = 'The tool is defect-free!'
        #     predicted_class='defect-free'

        # else: 
        #     msg_out = 'The tool is a defective tool' + \
        #                 '\n- ' + str(fault_size) + ' %' + ' of  its volume is diagnosed to be  defective! ' 
        #     predicted_class='defective'

        # msg=  '- Amin=' + str(Min_object_area) + \
        #     '\n- Execution time = ' + str(toc) + ' seconds'


        # Get the parameter
        Min_object_area=100
        mask_th=0.5; tol_list = [6,4,2]; fault_ratio_th=0.1

        msg= 'Running the ' + self.type_algo + '[DSP algorithm  with Amin=' + str(Min_object_area) + ']'
        print(msg); self.ui.msg_label.setText(msg)

        # Timing start
        tic = time.perf_counter()
        # Run the algorithm using 3D volume
        img3D_ref_prep, _ = self.get_volume(self.file_before)
        img3D_faulty_prep, _ = self.get_volume(self.file_after)

        #results distination
        save_folder =  'output/' 
        # Run the  fault detection algorithm1 using 3D volume
        mask_fault = Autils_fault_detection_DSP_3D.Run_fault_detection_algorithm1_3D(img3D_ref_prep, img3D_faulty_prep,\
                                            Min_object_area, mask_th=mask_th, denoise=False, disp=False, save_folder= save_folder)



        # # show in Napari
        # #visualise 
        # viewer = Autils_fault_detection_DSP_3D.napari_viewer(img3D_ref_prep[:,2:-2,:], img3D_faulty_prep[:,2:-2,:])
        # viewer.add_image(mask_fault[:,2:-2,:], name='Inspection results ', colormap='red', opacity=0.6)

        # Timing end
        toc = time.perf_counter() - tic
        toc = int(toc*100)/100


        # save the make results
        Autils_fault_detection_DSP_3D.save_3D_masks(mask_fault, save_folder)
        print('mask_fault size=', mask_fault.shape)
        # Load image
        fault_cathegory, fault_names = Autils.categorize_faults(mask_fault, tol_list, method=1)
        fault_cathegory.insert(0,img3D_faulty_prep)
        fault_names.insert(0,'Input scan')
        # visualize the results
        Autils.napari_view_volume(fault_cathegory, fault_names, inspection=True)
        # quatify the faulty error
        fault_size, msg_out, predicted_class = Autils_fault_detection_DSP_3D.quatify_fault(img3D_faulty_prep, mask_fault,fault_ratio_th=fault_ratio_th)
        mask_fault_segments = np.unique(mask_fault)
        msg =  msg + '\n- Amin=' + str(Min_object_area) + \
                '\n- fault_size=' + str(fault_size) + \
                '\n- predicted_class=' + str(predicted_class) + \
                '\n- mask_fault_segments=' + str(mask_fault_segments) + \
                '\n- Execution time = ' + str(toc) + ' seconds' + msg

        msg=  '- Amin=' + str(Min_object_area) + \
            '\n- Execution time = ' + str(toc) + ' seconds'

            


        # fault characterization
        # flag matthew: get these 2 paths from DB for tool
        #volume_masks='data/Tool2/STL/Volume_label.nrrd'
        #component_name_file= 'data/Tool2/STL/label_table.txt'
        volume_masks = sqlcon.get_tool_volume_location(self.tool_list.currentText())
        component_name_file = sqlcon.get_tool_labels_location(self.tool_list.currentText())
        
        
        fault_list=Jutils_3D_characterization.fault_characteriziation(volume_masks, mask_fault, component_name_file)
        msg_out=msg_out+'\nError Lists:\n'




        # percentage_threshold=25
        columns = ['name', 0, 'type_fault', 0, (0,0,0)]
        list_fault=[]
        scratch_list=' '
        for part_name, percentage in fault_list.items():
            
            # columns = ['name', 'risk_priority', 'type_fault', 'percentage', 'location_centroid']
            risk_priority=0

            type_fault='scratch'
            if percentage>25:
                type_fault='missing'
            elif percentage>15:
                type_fault='misplaced or broken'
            elif percentage>5:
                type_fault='scratch'
            
            location_centroid=(0,0,0)

            columns = [part_name, risk_priority, type_fault, percentage,location_centroid]
            list_fault.append(columns)

            if type_fault=='missing':
                msg_out=msg_out+'\n'+type_fault+' -- '+part_name
            elif type_fault=='misplaced or broken':
                msg_out=msg_out+'\n'+type_fault+' -- '+part_name
            else:
                scratch_list=part_name+ ',  '+scratch_list

        msg_out=msg_out+'\n'+'Scratches:'+' -- '+scratch_list

        # # Save/Plot the 3D fault volume results
        # output_volume_path = os.path.join(save_folder , 'fault_volume.nrrd')
        # import nrrd
        # nrrd.write(output_volume_path, mask_fault)
        # self.plot_3D_volume(output_volume_path, self.vtk_widget_3)

        # Plot the mask fault volume results
        mask_fault_vtk_vol = Autils.get_vtk_volume_from_3d_array(mask_fault)
        tool_vtk_vol = Autils.get_vtk_volume_from_3d_array(img3D_faulty_prep)

        mask_fault_vtk_vol = Autils.get_vtk_volume_from_3d_array(mask_fault)
        tool_vtk_vol = Autils.get_vtk_volume_from_3d_array(img3D_faulty_prep)

        self.Vizualize_3D_inspection_fault(self.vtk_widget_3, tool_vtk_vol, mask_fault_vtk_vol, mask_fault_segments)
        self.output_dic={"img_type":'3D',"img_after": self.file_after,  "mask_fault":mask_fault}

        # disply results and performance
        msg = msg + self.msg 
        print(msg);self.ui.msg_label.setText(msg)
        self.Label_outputs.setText(msg_out)

        return list_fault


    def Verify_Save_KnowledgeBase(self):        
        self.showdialog_FullInspection_results_saving() 
       
        # Hide button after compition 
        self.verify_save.hide()
   
    def set_fault_volume_properties(self, volume, volume_segments=0, max_segment=300):
        print('len(volume_segments) = ', len(volume_segments))
        # volume_segments = np.linspace(volume_segments.min(), volume_segments.max(), num=max_segment)
        if len(volume_segments)>1 and len(volume_segments)<=max_segment:
            #volume property
            volume_property = vtk.vtkVolumeProperty()
            volume_color = vtk.vtkColorTransferFunction()

            # The opacity 
            volume_scalar_opacity = vtk.vtkPiecewiseFunction()
            # The gradient opacity 
            volume_gradient_opacity = vtk.vtkPiecewiseFunction()
            volume_color = vtk.vtkColorTransferFunction()
            for i in volume_segments:
                if i==0:
                    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
                else:
                    color_i = (255,0,0)#tuple(np.random.randint(255, size=3))
                    volume_color.AddRGBPoint(i, color_i[0]/255.0, color_i[1]/255.0, color_i[2]/255.0)
                    # volume_scalar_opacity.AddPoint(i, 0.15)
                    # volume_gradient_opacity.AddPoint(0i, 0.5)
            volume_property.SetColor(volume_color)
            # volume_property.SetScalarOpacity(volume_scalar_opacity)
            # volume_property.SetGradientOpacity(volume_gradient_opacity)

            volume_property.ShadeOn()
            volume_property.SetAmbient(0.4)
            # volume_property.SetDiffuse(0.6)
            # volume_property.SetSpecular(0.2)

            # # setup the properties
            volume.SetProperty(volume_property)

        return volume

    def Vizualize_3D_inspection_fault(self, vtk_widget, volume,fault, volume_segments):
        vtk_widget.renderer.RemoveAllViewProps()
        vtk_widget.renderer.AddVolume(volume)
        #-------------------------------------------------------
        fault = self.set_fault_volume_properties(fault, volume_segments)
        vtk_widget.renderer.AddVolume(fault)
        vtk_widget.start()

def compile_ui(path_):
    from glob import glob
    list_ui = glob(path_+'.ui')
    print('list_ui=', list_ui)
    for filename in list_ui:
    # Recompile ui
        with open(filename) as ui_file:
            with open(filename.replace('.ui','.py'),"w") as py_ui_file:
                uic.compileUi(ui_file,py_ui_file)

def init_windows():
    # main_Win.show_about_dialog()
    # main_Win.paintEngine()
    main_Win.hide()
    prediction_Win.hide()
    Algorithms_Win.hide()
    Dataset_Win.show()
    progress_bar_WIN.hide()
    AutoInspection_Win.hide()
    VerifyReg_Win.hide()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    compile_ui("UI/*")

    # create pyqt5 app
    App = QApplication(sys.argv)
    # create  Main Window
    main_Win = MainApp()
    # example of showing dialog message
    # main_Win.showdialog_registration()

    # create  visualiation winndow
    Dataset_Win = DatasetApp()
    # create  prediction Window
    prediction_Win = OneImageIspectionApp()
    # create  Fault inspection algorithm
    Algorithms_Win = InspectionApp()
    # create  Fault inspection algorithm
    VerifyReg_Win = VerifyRegistration()
    # create  Fault inspection algorithm
    AutoInspection_Win = AutoInspectionApp()
    # progress bar
    progress_bar_WIN = Progress_bar_windows()

    # show the starting window
    init_windows()
    # start the app
    App.setStyle(QStyleFactory.create('Fusion'))
    sys.exit(App.exec())
