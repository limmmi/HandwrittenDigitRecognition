import tkinter as tk
import win32gui
import pyautogui as pg
import pretreatment_x
import CNN_2
import torch

# 加载训练好的卷积神经网络best模型
model_cnn_best = CNN_2.CNN()
model_cnn_best.load_state_dict(torch.load('cnn_best_NEW.pkl'))
model_cnn_best.eval()

list_output_num = []

root_size = 1400, 1020
canvas_size = 1300, 1000


class DigitalRecognition_UI:
    def __init__(self):
        root = tk.Tk()
        root.title('手写数字识别')
        root.geometry(f'{root_size[0]}x{root_size[1]}')
        root.configure(bg="blue")  # 背景颜色
        root.resizable(0, 0)  # 防止用户调整尺寸
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        canvas = tk.Canvas(root,width=canvas_size[0], height=canvas_size[1])
        canvas.place(x=0, y=0)

        # 读取画布
        def read():  # 通过截图保存画布内容
            HWND = win32gui.GetFocus()  # 获取当前窗口句柄
            rect1 = win32gui.GetWindowRect(HWND)  # 获取当前窗口坐标
            pg.screenshot('canvas.jpg',
                           region=(rect1[0], rect1[1],
                                   rect1[2] - rect1[0] - (root_size[0]-canvas_size[0]),
                                   rect1[3] - rect1[1] - (root_size[1]-canvas_size[1])))

        def savePosn(event):
            global lastx, lasty
            lastx, lasty = event.x, event.y

        def addLine(event):

            canvas.create_line((lastx, lasty, event.x, event.y), width=3)
            savePosn(event)

        def clearB():
            list_output_num.clear()
            canvas.delete('all')
            show_L['text'] = ''

        def predictB():
            list_output_num.clear()
            read()
            x1 = pretreatment_x.pretreatment('canvas.jpg')
            if x1 == 'ERROR':
                return
            output_cnn_best = model_cnn_best(x1)
            for i in range(len(output_cnn_best)):
                num = torch.argmax(output_cnn_best[i])
                list_output_num.append(str(int(num)))
            ans = "".join(list_output_num)
            show_L['text'] = ans

        canvas.bind("<Button-1>", savePosn)
        canvas.bind("<B1-Motion>", addLine)

        clear_B = tk.Button(master=root, bg='white', fg='black', activebackground='pink',
                            width=10, height=3, text='清除', command=clearB)

        predict_B = tk.Button(master=root, bg='white', fg='black', activebackground='pink',
                              width=10, height=3, text='识别', command=predictB)

        show_L = tk.Label(master=root, text=' ', width=10, height=3, relief='sunken')

        clear_B.place(x=1305, y=10)
        predict_B.place(x=1305, y=74)
        show_L.place(x=1305, y=138)
        root.mainloop()


enter = DigitalRecognition_UI()