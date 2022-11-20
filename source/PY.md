---
title: links
date: 2013/7/13 20:46:25
layout: py
permalink: PY.html
---
欢迎大家以下方格式添加~
``` python
from datetime import datetime

import aiohttp

from config import SERVER_CHAN_SEND_KEY


async def server_chan_send(dataset):
    """server酱将消息推送"""
    if SERVER_CHAN_SEND_KEY == '':
        return
    
    msg = ("| 账号 | 课程名 | 签到时间 | 签到状态 |\n"
           "| :----: | :----: | :------: | :------: |\n")
    msg_template = "|  {}  |  {}  | {}  |    {}    |"
    
    for datas in dataset:
        if datas:
            for data in datas:
                msg += msg_template.format(data['username'], data['name'], data['date'], data['status'])
                
            params = {
                'title': msg,
                'desp': msg
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method="GET",
                    url="https://sctapi.ftqq.com/{}.send?title=messagetitle".format(SERVER_CHAN_SEND_KEY),
                    params=params
                ) as resp:
                    text = await resp.text()
        else:
            msg = "当前暂无签到任务！\{}".format(datetime.now().strftime('%Y年%m月%d日 %H:%M:%D'))
            break
```