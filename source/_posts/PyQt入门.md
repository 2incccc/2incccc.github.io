---
title: PyQt入门
date: 2023-06-22 02:12:52
tags: [Python, 前端, 笔记]
categories:
- 笔记留档
---
## PyQt 简介

包括以下模块：QtCore  QtGui  QtWidgets  QtDBus  QtNetwork  QtHelp  QtXml  QtSvg  Qtsqll QtTest

---

## PyQt6日期与时间

`QDate QTime QDateTime`  分别处理公历日期 时间 和组合 

PyQt6 有 `currentDate`、`currentTime` 和 `currentDateTime` 方法获取当前的日期或时间。

....

---

## PyQt6的第一个程序

#### 简单示例

<!--more-->

```python

import sys
from PyQt6.QtWidgets import QApplication, QWidget
## 导入必要的包

def main():

    app = QApplication(sys.argv)
    ## 当我们把sys.argv 传给QApplication 时，我们是把命令行参数传给Qt。这使得我们可以在应用程序启动时将任何配置设置传递给Qt。
    w = QWidget()
    w.resize(250, 200)
    ## 改变大小
    w.move(300, 300)
    ## 移动位置

    w.setWindowTitle('Simple')
    ## 设置标题

    w.show()
    ##显示一个部件的步骤是首先在内存里创建，然后在屏幕上显示。

    sys.exit(app.exec())
    ## sys.exit 方法确保一个干净的退出。环境将被告知应用程序如何结束。



if __name__ == '__main__':
    main()

```

#### 气泡 QToolTip

```python
# file: tooltip.py
#!/usr/bin/python

"""
ZetCode PyQt6 tutorial

This example shows a tooltip on
a window and a button.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
from PyQt6.QtWidgets import (QWidget, QToolTip,
    QPushButton, QApplication)
from PyQt6.QtGui import QFont


class Example(QWidget):

    def __init__(self):
        super().__init__() ## 调用父类的

        self.initUI()


    def initUI(self):

        QToolTip.setFont(QFont('SansSerif', 10))
        ## 这个静态方法给气泡提示框设置了字体，这里使用了10pt 的 SansSerif 字体。
        
        self.setToolTip('This is a <i>QWidget</i> widget')

        btn = QPushButton('Button', self)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        ## 在气泡提示框上添加了一个按钮部件。
        
        btn.resize(btn.sizeHint())
        btn.move(50, 50)
        ## sizeHint 方法是给按钮一个系统建议的尺寸，然后使用 move 方法移动这个按钮的位置。
        
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Tooltips')
        self.show()


def main():

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

#### 退出按钮

参数 text 是将显示在按钮上的文本。parent 是我们放置按钮的小部件。在我们的例子中，它将是一个QWidget。**应用程序的小部件形成层次结构**，在这个层次结构中，大多数小部件都有父级。没有父级的小部件的父级是顶级窗口。

```python
# file: quit_button.py
#!/usr/bin/python

"""
ZetCode PyQt6 tutorial

This program creates a quit
button. When we press the button,
the application terminates.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
from PyQt6.QtWidgets import QWidget, QPushButton, QApplication

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        qbtn = QPushButton('Quit', self)
        ## 我们创建了一个按钮，它是 QPushButton 类的一个实例。构造函数的第一个参数是按钮的标签。 第二个参数是父级小部件。父小部件是 Example 小部件，它继承自 QWidget。

        qbtn.clicked.connect(QApplication.instance().quit)
        ## PyQt6 的事件处理系统是由信号和插槽机制构成的，点击按钮（事件），会发出点击信号。事件处理插槽可以是 Qt 自带的插槽，也可以是普通 Python 函数
        ## 使用 QApplication.instance 获取的 QCoreApplication 对象包含主事件循环————它处理和分派所有事件。 单击的信号连接到终止应用程序的退出方法。 通信是在两个对象之间完成的：发送者和接收者。 发送者是按钮，接收者是应用程序对象。
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 50)

        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle('Quit button')
        self.show()


def main():

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

#### 弹窗

```python
# file: messagebox.py
#!/usr/bin/python

"""
ZetCode PyQt6 tutorial

This program shows a confirmation
message box when we click on the close
button of the application window.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
from PyQt6.QtWidgets import QWidget, QMessageBox, QApplication


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, 350, 200)
        self.setWindowTitle('Message box')
        self.show()

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                    "Are you sure to quit?", QMessageBox.StandardButton.Yes |
                    QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                    ## 这里创建了一个带有两个按钮的消息框：是和否。第一个参数是标题栏，第二个参数是对话框显示的消息文本，第三个参数是对话框中的按钮组合，最后一个参数是默认选中的按钮。返回值存储在变量 reply 中。

        if reply == QMessageBox.StandardButton.Yes:

            event.accept()
        else:

            event.ignore()


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

#### 窗口居中
```python
# file: center.py
#!/usr/bin/python

"""
ZetCode PyQt6 tutorial

This program centers a window
on the screen.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
from PyQt6.QtWidgets import QWidget, QApplication


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.resize(350, 250)
        self.center()

        self.setWindowTitle('Center')
        self.show()

    def center(self):

        qr = self.frameGeometry()
        ## 这样就可以得到一个矩形的窗口，这里可以放置所有类型的窗口。
        cp = self.screen().availableGeometry().center()
        ## 从屏幕属性里计算出分辨率，然后计算出中心点位置。

        qr.moveCenter(cp)
        self.move(qr.topLeft())
        ## 我们已经知道矩形窗口的宽高，只需要把矩形窗口的中心点放置到屏幕窗口的中心点即可。这不会修改矩形窗口的大小。
        ## 把应用窗口的左上方点坐标移动到矩形窗口的左上方，这样就可以居中显示了。


def main():

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

---

## PyQt6的菜单和工具栏
在这部分教程中，我们创建了一个状态栏、菜单栏和工具栏。菜单是位于菜单栏中的一组命令。工具栏有一些按钮和应用程序中的一些常用命令。状态栏显示状态信息，通常位于应用程序窗口的底部。

### QMainWindow
`QMainWindow`类提供了主程序窗口。在这里可以创建一个具有状态栏、工具栏和菜单栏的经典应用程序框架。

```python
# file: statusbar.py
#!/usr/bin/python

"""
ZetCode PyQt6 tutorial

This program creates a statusbar.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
from PyQt6.QtWidgets import QMainWindow, QApplication


class Example(QMainWindow):
    ## 在这里可以创建一个具有状态栏、工具栏和菜单栏的经典应用程序框架。

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.statusBar().showMessage('Ready')

        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle('Statusbar')
        self.show()


def main():

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

---

>课程来源： Python GUI Development with PyQt6 & Qt Designer


* PyQt Introduction

* PyQt6 Installation and First GUI Window 

* Window Class Types
三种窗口类
    * QMainWindow：包括工具栏、菜单栏、状态栏等
    * QDialog: 对话窗口类
    * QWidget：所有用户界面对象的基类

* Adding Icon & Title to PyQT6 Window

* Introduction to Qt Designer

![](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16825910403231682591039550.png)

* Convert UI File to PY
``` bash
pyuic6 -x windowUI.ui -o windowUI.py
```

* Loading UI
```python
from PyQt6 import uic
```

---

* Working with QLabel
用于显示文字、图像

* Working with QPyshButton

* QVBoxLayout&QHBoxLayout

* QGridLayout 网格布局，用于对齐不见
  ![](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16825972503241682597249498.png)
* Event Handling
    事件处理机制叫做信号和槽
    每个小部件在应用任何事件时（如点击clicked 滑动等）可以发出信号，需要槽连接该信号和方法（method）
    很多部件（如按键等）有内置信号，如btn.clicked 通过btn.clicked.connect(self.clicked_btn) （括号内为槽）相连接起来。
* Event Handling with Qt Designer

---

#### Simple Calculator with Qt Designer
步骤：
1. 确定对象，修改对象命名、属性等
2. 排布对象布局
3. 生成ui文件，并转为py文件
4. 在py文件中编写事件函数（信号、槽）

---

* Working with QRadioButton
单选按钮，一次只能选中一个
* Working with QChexbox 
  复选框
* Creating QSpinBox
  上下调整的框框以修改值
  ![](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16832000343061683200033667.png)
* Creating QDoubleSpinBox