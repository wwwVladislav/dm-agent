import socket
import json

class UDPClient:

    @staticmethod
    def rotate(host,port,alpha, beta):
        data = {}
        data['name'] = 'ROTATE'
        data['arguments'] = (alpha,beta)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((host, port))
        sock.send(json.dumps(data).encode())
        sock.close()

    @staticmethod
    def fire(host,port):
        data = {}
        data['name'] = 'FIRE'

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((host, port))
        sock.send(json.dumps(data).encode())
        sock.close()
