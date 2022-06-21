import PySimpleGUI as sg
import cv2 as cv
import numpy as np
import torch

# FUNCTION
def np_to_byte(img):
    ret, frame_png = cv.imencode('.png', img)
    img_byte = frame_png.tobytes()
    return img_byte

# LAYOUT
sg.theme('SystemDefault')
image_control = sg.Column([[sg.Image(np_to_byte(np.full((480, 640), 0)),k='IMAGE')]])
setting_control = sg.Column([
    [sg.Image(r"D:\Code\Image Processing\FP PCV\BATTERY.png", expand_x=True, pad=(0,30))],
    [sg.Button('START',expand_x=True), sg.Button('STOP',expand_x=True)],
    [sg.Button('BROWSE',expand_x=True, k="BROWSE",disabled = True)],
    [sg.Frame('Setting', layout=[
                                    [sg.Checkbox('Mirror', k='MIRROR'),sg.Checkbox('Use Local File', pad=(20,0),k='LCL')],
                                ],expand_x=True)],
    [sg.Frame('Model Configuration', layout= [
                                        [sg.Text('Class Model')],
                                        [sg.Checkbox('Batarai 9V', k='9V_CHECK'),sg.Checkbox('Batarai AA', k='AA_CHECK')],
                                        [sg.Text('Model Confidence')],
                                        [sg.Slider(range=(0,100),orientation='h',k='CONF',default_value=50, expand_x=True)]
                                    ], expand_x=True )],
])
layout = [[image_control, setting_control]]

# INIT
window = sg.Window('Battery Classification', layout)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='Image Processing/FP PCV/yolov5/runs/train/exp/weights/best.pt', force_reload=True)
cap = cv.VideoCapture(2)
recording = True
file_path = 'D:\Code\Image Processing\Image\gambarITS.jpg'

# MAIN LOOP
while True:
    event, values = window.read(timeout=16)

    if event == sg.WIN_CLOSED:
        break
    
    if event == 'START':
        recording = True
    
    if event == 'STOP':
        recording = False
        window['IMAGE'].update(data=np_to_byte(np.full((480, 640), 0)))

    if values['LCL']:
        window['BROWSE'].update(disabled=False)
    else:
        window['BROWSE'].update(disabled=True)

    if event == 'BROWSE':
        file_path = sg.popup_get_file('open', no_window=True)
        if file_path[-3:] == 'mp4':
            cap2 = cv.VideoCapture(file_path)
        elif file_path == '':
            file_path = 'D:\Code\Image Processing\Image\gambarITS.jpg'

    if recording:
        ret, frame = cap.read()
        frame = cv.resize(frame, (640,480))

        if file_path[-3:] == 'mp4':
            ret2, fill_bg = cap2.read()
            if ret2:
                pass
            else:
                print('no video')
                cap2.set(cv.CAP_PROP_POS_FRAMES, 0)
            fill_bg = cv.resize(fill_bg, (640,480))
        else:
            fill_bg = cv.imread(file_path)
            fill_bg = cv.resize(fill_bg, (640,480))

        # MIRROR
        if values['MIRROR']:
            frame = cv.flip(frame, 1)

        # Input YOLO Model
        model.classes = []
        model.conf = values['CONF'] / 100
        if values['9V_CHECK']:
            model.classes.append(0)
        if values['AA_CHECK']:
            model.classes.append(1)
        
        result = model(frame)
        result.render()

        imgbytes = np_to_byte(result.imgs[0])
        window['IMAGE'].update(data=imgbytes)

window.close()