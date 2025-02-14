class Config: # 역전파 활성화 모드
  enable_backprop=True
  #  함수들
import contextlib
import numpy as np
@contextlib.contextmanager
def using_config(name, value):
  old_value=getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield # 컨텍스트 블록 실행 (이 부분에서 실행 흐름이 사용자 코드로 넘어감)
  finally:
    setattr(Config, name, old_value)

def no_grad():
  return using_config('enable_backprop',False)

def as_array(x):
  if np.isscalar(x): #numpy.float64같은 스칼라타입인지 확인
    return np.array(x)
  return x

def as_variable(obj): # np인스턴스와 함꼐 입력되도 오류없이 수행
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)

import weakref
class Variable:
  __array_priority__=200
  def __init__(self, data, name=None):
    if data is not None:
      if not isinstance(data,np.ndarray):
        raise TypeError("{}은 지원 불가".format(type(data)))
    self.data=data
    self.name=name # 많은 변수 처리 -> 이름 필요 -> 인스턴스 변수 추가
    self.grad=None
    self.creator=None
    self.generation=0

  def set_creator(self, func):
    self.creator=func
    self.generation=func.generation+1 # 부모 세대 함수보다 1만큼 큰 값 설정

  def backward(self, retain_grad=False): # retain_grad: 필요 없는 미분값 삭제 / 보통 말단 변수 미분값만이 필요
    if self.grad is None:
      self.grad=np.ones_like(self.data)

    funcs=[]
    seen_set=set()
    def add_func(f):
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        funcs.sort(key=lambda x:x.generation)

    add_func(self.creator)

    while funcs:
      f=funcs.pop()
      gys=[output().grad for output in f.outputs] #약한 참조로 바꿨으므로 ()추가하여 수정
      gxs=f.backward(*gys)
      if not isinstance(gxs,tuple):
        gxs=(gxs,)
      for x, gx in zip(f.inputs, gxs):
        if x.grad is None:
          x.grad=gx
        else:
          x.grad = x.grad + gx
        if x.creator is not None:
          add_func(x.creator) # 수정

      if not retain_grad: # 말단 변수 아닐시 미분값 삭제
        for y in f.outputs:
          y().grad=None

  def cleargrad(self):
    self.grad=None

  # 목표: Variable을 ndarray처럼 보이게 만드는 것
  @property # 인스턴스 변수처럼 사용할수 있게 함 x.shape() -> x.shape
  def shape(self):
    return self.data.shape

  @property
  def ndim(self):
    return self.data.ndim

  @property
  def size(self):
    return self.data.size

  @property
  def dtype(self):
    return self.data.dtype

  def __len__(self): # 특수 메서드
    return len(self.data)

  def __repr__(self): # print함수 출력값 수정하려면  __repr__재정의
    if self.data is None:
      return 'variable(None)'
    p=str(self.data).replace('\n','\n'+''*9)
    return 'variable('+p+')'
  # 함수식 거추장-> 연산자 오버로드
  def __mul__(self, other): #self=a전달, other=b전달
    return mul(self, other)
  
class Function:
  def __call__(self, *inputs):
    inputs=[as_variable(x) for x in inputs]
    xs=[x.data for x in inputs]
    ys=self.forward(*xs)
    if not isinstance(ys, tuple):
      ys=(ys,)
    outputs=[Variable(as_array(y)) for y in ys]

    if Config.enable_backprop: # 역전파 모드에서만 연결 연산 수행
      # 입력 변수 세대중에 가장 큰 세대 값으로 함수의 세대 설정
      self.generation=max([x.generation for x in inputs])
      for output in outputs:
        output.set_creator(self)
      self.inputs=inputs
      self.outputs=[weakref.ref(output) for output in outputs] # 약한 참조로 수정

    return outputs if len(outputs)>1 else outputs[0]

  def forward(self, xs):
    raise NotImplementedError()

  def backward(self, gys):
    raise NotImplementedError()

class Square(Function):
  def forward(self, x):
    y=x**2
    return y
  def backward(self, gy):
    x=self.inputs[0].data
    gx=2*x*gy
    return gx

class Add(Function):
  def forward(self, x0,x1):
    y=x0+x1
    return y
  def backward(self, gy):
    return gy,gy #상류에서 흘러오는 미분값을 그대로 흘려보내는 것이 덧셈의 역전파(미분시, gy*1이라서)

def square(x):
  return Square()(x)
def add(x0,x1):
  x1=as_array(x1) #x1가 float등일 경우 ndarray인스턴스로 변환
  return Add()(x0,x1)

class Mul(Function):
  def forward(self, x0, x1):
    y=x0*x1
    return y

  def backward(self, gy):
    x0,x1=self.inputs[0].data, self.inputs[1].data
    return gy*x1, gy*x0
def mul(x0,x1):
  x1=as_array(x1) #x1가 float등일 경우 ndarray인스턴스로 변환
  return Mul()(x0,x1)



class Neg(Function):
  def forward(self,x):
    return -x
  def backward(self, gy):
    return -gy
def neg(x):
  return Neg()(x)

class Sub(Function):
  def forward(self, x0,x1):
    return x0-x1
  def backward(self, gy):
    return gy,-gy

def sub(x0,x1):
  x1=as_array(x1)
  return Sub()(x0,x1)


def rsub(x0,x1):
  x1=as_array(x1)
  return Sub()(x1,x0)

class Div(Function):
  def forward(self, x0,x1):
    y=x0/x1
    return y
  def backward(self, gy):
    x0,x1=self.inputs[0].data, self.inputs[1].data
    return gy*(1/x1), gy*(-x0/x1**2)

def div(x0,x1):
  x1=as_array(x1)
  return Div()(x0,x1)

def rdiv(x0,x1):
  x1=as_array(x1)
  return Div()(x1,x0)

class Pow(Function):
  def __init__(self, c):
    self.c=c
  def forward(self, x):
    y=x**self.c
    return y
  def backward(self, gy):
    x=self.inputs[0].data
    return gy*(self.c*x**(self.c-1))

def pow(x,c):
  return Pow(c)(x)

def setup_variable():
  Variable.__add__=add
  Variable.__radd__=add
  Variable.__mul__=mul
  Variable.__rmul__=mul
  Variable.__rsub__=rsub
  Variable.__sub__=sub
  Variable.__neg__=neg
  Variable.__truediv__=div
  Variable.__rtruediv__=rdiv
  Variable.__pow__=pow
