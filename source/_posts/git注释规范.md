---
title: git注释规范
date: 2022-08-29 11:50:04
tags: [git,test]
cover: https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/1661745096140wallhaven-9mjoy1.png
top_img: https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/1661745096140wallhaven-9mjoy1.png
---
``` python
import time

from selenium import webdriver
from selenium.webdriver.common.by import By

class QQMusicPlayer():
    def __init__(self,Account,Password):
        self.driver = webdriver.Chrome()
        self.driver.get("https://y.qq.com/")
        self.account = Account
        self.password = Password

    def Login(self):
        #目前QQ音乐只支持QQ号登录
        account = self.account
        password = self.password
        self.driver.find_element(By.XPATH,"/html/body/div/div/div[1]/div/div[2]/span/a").click()
        time.sleep(1)
        self.driver.find_element(By.XPATH,"/html/body/div[3]/div/div[2]/div/div[2]/div/div/div[1]/h2/a[1]").click()
        #选择使用密码登录
        iframe1 = self.driver.find_element(By.XPATH,"/html/body/div[3]/div/div[2]/div/div[2]/div/div/div[2]/div[1]/iframe")
        self.driver.switch_to.frame(iframe1)
        iframe2 = self.driver.find_element(By.XPATH,"/html/body/div[2]/div[1]/div/iframe")
        self.driver.switch_to.frame(iframe2)
        self.driver.find_element(By.XPATH,"/html/body/div[1]/div[9]/a[1]").click()
        #输入账号密码
        self.driver.find_element(By.XPATH,"/html/body/div[1]/div[5]/div/div[1]/div[3]/form/div[1]/div/input").send_keys(self.account)
        self.driver.find_element(By.XPATH,"/html/body/div[1]/div[5]/div/div[1]/div[3]/form/div[2]/div[1]/input").send_keys(self.password)
        self.driver.find_element(By.XPATH,"/html/body/div[1]/div[5]/div/div[1]/div[3]/form/div[4]/a").click()
        #发起需要验证码页面
        self.PopDialog()


    def PopDialog(self):
        return None

if __name__ == "__main__":
    Player = QQMusicPlayer("2452230995","lyj2003727123.")
    Player.Login()
```