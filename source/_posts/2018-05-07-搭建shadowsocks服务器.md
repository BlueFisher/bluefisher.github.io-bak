---
title: 搭建shadowsocks服务器
mathjax: false
date: 2018-05-07 14:52:38
categories:
- Linux
tags:
- shadowsocks
---

# 安装shadowsocks

```bash
sudo apt-get install python3-pip
sudo pip3 install shadowsocks
```

若出现 `Command "[Python](https://link.jianshu.com/?t=http://lib.csdn.net/base/python) setup.py egg_info" failed with error code 1 in /tmp/pip-build-* ` 错误，则需要安装：

```bash
sudo pip3 install setuptools
```

新建shadowsocks配置文件shadowsocks.json ：

```json
{
  "server": "::",
  "port_password": {
    "PORT1": "PASSWORD1",
    "PORT2": "PASSWORD1"
  },
  "timeout": 300,
  "method": "rc4-md5",
  "fast_open": true
}
```

 `PORT1` ，`PORT2` 为服务器监听的端口号，后面是客户端连接当前端口的密码

开启服务器测试：

```bash
ssserver -c shadowsocks.json
```

如遇到 `AttributeError: /usr/lib/x86_64-Linux-gnu/libcrypto.so.1.1: undefined symbol: EVP_CIPHER_CTX_cleanup` 错误，则：

打开文件 `/usr/local/lib/python3.6/dist-packages/shadowsocks/crypto/openssl.py` 

将 `libcrypto.EVP_CIPHER_CTX_cleanup.argtypes = (c_void_p,)` 改为 `libcrypto.EVP_CIPHER_CTX_reset.argtypes = (c_void_p,)`

将 `libcrypto.EVP_CIPHER_CTX_cleanup(self._ctx)` 改为 `libcrypto.EVP_CIPHER_CTX_reset(self._ctx)`

重新启动 shadowsocks 即可。

<!--more-->

# 加速

## Google BBR 加速

```bash
wget –no-check-certificate https://github.com/teddysun/across/raw/master/bbr.sh
chmod +x bbr.sh
./bbr.sh
reboot
```

## TCP Fast Open 

<https://github.com/shadowsocks/shadowsocks/wiki/TCP-Fast-Open>

# 开机启动

以systemd来配置开机启动

新建 `/usr/lib/systemd/system/shadowsocks.service `

```
[Unit]
Description=Shadowsocks Client Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/ssserver -c /home/xxx/shadowsocks.json

[Install]
WantedBy=multi-user.target
```

或者也可以修改成

```
[Service]
Type=forking
User=root
ExecStart=/usr/local/bin/ssserver -c /home/xxx/shadowsocks.json -d start
```

其中 `ExecStart` 中的内容换成shadowsocks的安装路径和配置文件的安装路径

```bash
sudo systemctl enable /usr/lib/systemd/system/shadowsocks.service
sudo systemctl start /usr/lib/systemd/system/shadowsocks.service
sudo systemctl status /usr/lib/systemd/system/shadowsocks.service
sudo systemctl stop /usr/lib/systemd/system/shadowsocks.service
```

功能分别为应用开机启动、立即启动服务、查看服务状态、停止服务

如果已经应用开机启动，但service文件改变，则需要重新载入service文件

```bash
sudo systemctl daemon-reload
```

查看日志：

```bash
sudo journalctl -u shadowsocks
```
# 开启 ipv6

可能某些 VPS 的 google scholar ipv4 地址被 Google 封了，打开谷歌学术一直是 "We're sorry..." ，所以需要在服务器上开启 ipv6 来访问。

安装 Miredo ，若 VPS 已经设置了 IPV6 则不需要

```bash
sudo apt install miredo
```

查看 `ifconfig` 应该会多出 teredo 的虚拟网卡，可以在服务器上测试下 `ping6 ipv6.google.com` 

在 `/etc/hosts` 中添加

```
## Scholar 学术搜索
2404:6800:4008:c06::be scholar.google.com
2404:6800:4008:c06::be scholar.google.com.hk
2404:6800:4008:c06::be scholar.google.com.tw
2401:3800:4001:10::101f scholar.google.cn #www.google.cn
```

具体可以参考 <https://raw.githubusercontent.com/lennylxx/ipv6-hosts/master/hosts>

最后重启 shadowsocks 即可

```bash
sudo systemctl restart /usr/lib/systemd/system/shadowsocks.service
```