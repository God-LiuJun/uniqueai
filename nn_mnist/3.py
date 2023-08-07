def backward(self,dout):
    dx=dout
    dx[self.x<=0]=0
    return dx