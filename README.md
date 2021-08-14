

[TOC]

------

# Python Common

##### 如何用递归进行执行if分支

```python
def set_intersection(self, set1, set2):
    if len(set1) > len(set2):
        return self.set_intersection(set2, set1)
    return [x for x in set1 if x in set2]
```

##### 如何非numpy argsort

```python
#对等长range 排序
#再次理解 key 需要传一个callable（函数或者__call__） 函数参数是索引 返回的是计算出的排序分数 以分数从小到大sort
score = [5,4,3,2,1]
argsort = list(range(len(score)))
argsort.sort(key=lambda i:score[i])
```

##### sort的key如何依据两组数据排序

```python
#sort key的输出可以是一个tuple 表明 第一个维度相等情况下 按照第二个维度排
chinese = [ 90 , 90, 91]
math = [ 99 ,100, 59]
argsort = list(range(len(chinese)))
argsort.sort(key=lambda i:(chinese[i],math[i]))
```

##### KeyboardInterrupt也是一种Exception

```python
#特别地 这里print e不会有反映 
from random import gauss
try:
    idd = 0
    while 1:
        idd += gauss(0,1)
        print('\r%.6f'%idd,end='')
except KeyboardInterrupt as e:
    print('last idd: %.6f'%idd)
```

##### softmax溢出问题

```python
#指数函数的特点正半轴无穷溢出，负半轴无穷0
#举个例子[1,10,100,1000]算分母的时候e**1000会溢出
#所以对softmax上下同时除以e**1000也即是变成求[1-1000,10-1000,100-1000,0]的softmax这样不会溢出
def softmax(X,axis=-1):
    X = X - X.max(axis=axis,keepdims=True)
    X_exp = np.exp(X)
    return X_exp/X_exp.sum(axis=axis,keepdims=True)
```

##### 如何生成batch

```python
#注意这种方式是keep last 若要实现droplast 需要对最后一个batch判断
X is numpy array
for i in range(0,len(X),batch_size):
    x_batch = x[i:i+batch_size]
```

```python
class dataloader:
    def __init__(self,X,batch_size=1):
        self.X = X
        self.batch_size = batch_size
        self.idx = 0
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.X)
    
    def __next__(self):
        if self.idx > self.__len__():
            self.idx = 0  #复位 下一次for的时候 还能iteration 
            raise  StopIteration
            
        clip = self.X[self.idx:self.idx+self.batch_size]
        self.idx += self.batch_size

        return clip

# X = np.arange(10).reshape(5,2)
# X_loader = dataloader(X,2)

# for batch_x in X_loader:
#     print(batch_x)


# for batch_x in X_loader:
#     print(batch_x)
```

##### yield from 的用处

```python
#yield from 后面跟可迭代对象
#相比于yield， yield from 可以在不同生成器函数之间灵活切换
#举一个 二叉树搜索树的例子 我想返回二叉树中所有大于 target值的节点值 相对顺序不变
#假设我用递归地方式 中序遍历 那么可以这么写

def get_larger(node,target):
    if node is None:
        return
    yield from get_larger(node.left,target)
    if node.val > target:
        yield node.val
    yield from get_larger(node.right,target)

print(list(get_larger(root,-1)))

```

##### numpy 实现one-hot几种方式

```python
target = [0,1,2,1,3]  #假设4分类

'''方式一
将target升高一个维度成为一个列向量
第一行 [0] == [0,1,2,3] 得 [TFFF] 转int即one-hot
依次类推
这里看出 == 也可以广播，广播时，维度多者取出第一个维度与维度少者比较
'''
(np.arange(target.min(),target.max()+1)==target[:,None]).astype('int')

'''方式二
華索
注意 fancy indexing 索引一定是array或者列表不是 元组
'''
np.eye(target.max()+1)[target]

'''方式三
在计算交叉熵的时候 还可以省去转换one hot 直接根据one hot的选出对饮特征即可
有点像NLL
这里再次用到了fancy indexing
np.arange(len(y)) 遍历每一个样本
target选出对应的特征
'''
y = np.arange(20).reshape(5,4)
y[np.arange(len(y)),target]
#np.log(y)
```

##### %r万能格式输出

```python
formatter = "%r %r %r %r"
 
print(formatter%(1, 2, 3, 4))
print(formatter% ("o'n'e", "two", "three", "four")) 
print(formatter% (True, False, False, True)) 
print(formatter% (formatter, formatter, formatter, formatter)) 
print(formatter% (
"I had this thing.",
"That you could type up right.",
 "But it didn't sing.",
 "So I said goodnight."
 )) 
```

##### 生成器表达式

+ “(i for i in xxx)” 像这样的式子叫做生成器表达式，类型generator
  + 当用生成器生成list，set或者dict时 （）可以不要。比如[i for i in xxx]
  + 但是要用其**生成元组时，必须显式写出来**，tuple（i for i in xxx）

##### *的特殊作用

```python
#把可迭代拆开作为函数参数传入
t1 = (1,2,3)
max(*t1)
'''
max的定义如下
max(iterable, *[, default=obj, key=func]) -> value
max(arg1, arg2, *args, *[, key=func]) -> value
想一想 这样做调用的是谁
'''
```

```python
#接收参数
In [12]: a,b,*t2 = range(5)
In [13]: a,b,t2
Out[13]: (0, 1, [2, 3, 4])
```

```python
#特别注意 *t本身不是一个东西 甚至不能打印它
In [14]: *t1
  File "<ipython-input-14-3eca0c89f958>", line 1
    *t1
       ^
SyntaxError: can't use starred expression here
```

##### os path split 分割文件名

```python
In [18]: _,f = os.path.split('/home/porn.jpg')

In [19]: f
Out[19]: 'porn.jpg'
```

##### format的对齐与填充

+ 先说写法，｛<xxx>:<?><num>｝

  + xxx为填充内容（默认空格填充）
  + ？是对其方式，左中右分别为<^>(默认左对齐)
  + num填对其的长度

+ ```python
  In [29]: '{:<9}{:♥^9}{person:>9}'.format('I','Love',person='You')
  Out[29]: 'I        ♥♥Love♥♥♥      You'
  ```

##### filter的作用

+ 比如一些年份你筛选出哪些年举办奥运会，那么大可以这么做

  ```python
  year = tuple(range(1800,2022))
  year1 = tuple(item for item in year if item%4==0)
  ```

+ 但是如果要你判断闰年，条件复杂的时候用filter筛选（反过来如果条件简单，写lambda函数也麻烦，就按照上面例子的意思写生成器即可）

  ```python
  def is_runn(y):
      flag = False
      if y%4 == 0:
          flag = True
      if y%100 == 0:
          flag =  False
      if y%400 == 0:
          flag = True
      return flag
  
  year2 = tuple(filter(is_runn,year))
  ```

##### 具名元组

+ collections.namedtuple
+ 类似于dict只不过不可变，非常有用，自行百度

##### slice

+ slice(a,b,c)完全等价于[a : b : c] 

+ 但是slice相当于可以对切片起名字,更具体

  ```python
  #比如有班级学生信息样本
  info = '1997-08-26|M|Larry'
  BIRTH = slice(0,10)
  SEX = slice(11,12)
  NAME = slice(13,None)
  
  birth = info[BIRTH]
  sex = info[SEX]
  name = info[NAME]
  ```

##### numpy 矩阵分块

```python
arr1=np.eye(3)#单位矩阵
arr2=arr1*3

#使用bmat()创建矩阵【合成矩阵】
matr=np.bmat("arr1 arr2;arr1 arr2")
```

##### del 删除列表元素

```python
a = [1,2,3,4,5]
del a[1:3]
```

##### *与列表的 注意事项

```python
#如果列表里items都是不可变元素 那么可以用*进行复制
#如果不是 那么注意 ，不可变对象只相当于复制了一个引用
#举个例子 很多二维dp问题初始化的时候 错误的写法

In [2]: a=[[0]*3]*3

In [3]: a
Out[3]: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

In [4]: a[1][1] = 2

In [5]: a
Out[5]: [[0, 2, 0], [0, 2, 0], [0, 2, 0]]

```

```python
#正确的写法是
In [6]: a = [[0]*3 for i in range(3)]

In [7]: a[0][0] = 13123

In [8]: a
Out[8]: [[13123, 0, 0], [0, 0, 0], [0, 0, 0]]
```

##### 不要往不可变对象里放可变对象

```python
#+=谜题如下所示
In [9]: t = (1,2,[2,3])

In [10]: t[-1] += [4,5]
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-10-725136d4cd8e> in <module>()
----> 1 t[-1] += [4,5]

TypeError: 'tuple' object does not support item assignment

In [11]: t
Out[11]: (1, 2, [2, 3, 4, 5])
```

由此可以得出以下几点：

+ 不要往不可变对象里放可变对象
+ += 首先对列表操作了列表改变了，这时是对可变对象操作，可以执行
+ 改变后的列表重新赋值到元组里报错，元组里元素不可变

##### 字典构造了几种方式

```python
In [2]: dict(one=1,two=2)  
Out[2]: {'one': 1, 'two': 2}

In [3]: {'one':1,'two':2}
Out[3]: {'one': 1, 'two': 2}
    
In [6]: dict(zip(['a','b'],[1,2]))
Out[6]: {'a': 1, 'b': 2}

In [7]: dict([('one',1),('two',2)])
Out[7]: {'one': 1, 'two': 2}
    
In [9]: dict.fromkeys(range(5))
Out[9]: {0: None, 1: None, 2: None, 3: None, 4: None}
    
#注意zip；直接传参；from keys
```

##### 字典setdefault

```python
#setdefault的作用是返回value
#如果没找到对应的key按照给定的value去set并返回value
#注意 它和get最大的区别在于get的default不写进字典里
#setdefault会写进并返回

#比如统计词位置的时候 可以这么写
In [15]: string = "I love you I miss you But you do not love me".split()

In [17]: coll = {}

In [18]: for i,char in enumerate(string,1):
    ...:     coll.setdefault(char,[]).append(i)

In [19]: coll
Out[19]:
{'I': [1, 4],
 'love': [2, 11],
 'you': [3, 6, 8],
 'miss': [5],
 'But': [7],
 'do': [9],
 'not': [10],
 'me': [12]}
```

##### 字符串的字典序

```python
'''
其实python 这部分字符串直接比较大小就是字典序
如果字串长度不等 那么长者字典序大 
如果等 则比较每一位的ord
'''
def DictOrder(str1,str2):
    def cmp(a,b):
        if len(a) == 0 and len(b) == 0:
            return 'eq'
        elif a[0] > b[0]:
            return 'greater'
        elif a[0] < b[0]:
            return 'lower'
        else:
            return cmp(a[1:],b[1:])
    
    if len(str1) > len(str2):
        return 'greater'
    elif len(str1) < len(str2):
        return 'lower'
    else:
        return cmp(str1,str2)
```

##### defaultdict

+ default的作用在于可以帮你默认初始化
+ 传入default_factory必须是一个可调用或者None
+ 举个例子 要模拟10000掷骰子实验记录情况
+ 注意 defaltdict的作用机理是 在_ _ getitem _ _ 里调用我们 _ _ missing _ _ 所以get,in 等其他不会受此作用或者影响

```python
from collections import defaultdict
from random import randint

nums = [randint(1,6) for _ in range(10000)]
counter = defaultdict(int)  #也可以写作 lambda : 0 
for num in nums:
    counter[num] += 1

```

##### cmp to key

+ sort的key只能比较传单一元素比较，如果有2个key需要用cmp to key函数

+ cmp to key的原理是一次传入两个元素并计算

  + 如果返回的是一个大于0的值，那么代表a>b
  + 如果返回的是一个小于0的值，那么代表a<b
  +  如果返回的是一个等于0的值，那么代表a=b

```python
#应用1 牌复原
from functools import cmp_to_key
Meihua = lambda x: '♣'+ x
Heitao = lambda x: '♠'+ x 
Hongtao = lambda x: '♥' + x
Fangpian = lambda x: '♦' + x

def compare(x,y):
    x = x.replace('A','1')
    y = y.replace('A','1')
    if x[1:] != y[1:]:
        return int(x[1:]) - int(y[1:])
    else:
        return ord(x[0]) - ord(y[0])


res = [ flower(item) for flower in [Meihua,Heitao,Hongtao,Fangpian] for item in [str(num) for num in range(2,11)]+['A']]
res.sort(key= cmp_to_key(compare))
print(res)

```

```python
#应用2 排名
from functools import cmp_to_key
persons = [
    {
        'name':'zhangsan',
        'age':20,
        'grade':98
    },
    {
        'name':'lisi',
        'age':18,
        'grade': 88
    },
    {
        'name':'wangwu',
        'age':20,
        'grade': 20
    },
    {
        'name': 'yanqing',
        'age': 15,
        'grade': 20
    },
    {
        'name': 'awu',
        'age': 20,
        'grade': 20
    },
]

def cmp(a,b):
    if a['grade'] > b['grade']:
        return 1

    elif a['grade'] < b['grade']:
        return -1
    else:
        if a['age'] > b['age']:
            return 1
        elif a['age'] < b['age']:
            return -1
        else:
            if a['name'] > b['name']:
                return 1
            else:
                return -1
persons.sort(key=cmp_to_key(cmp))
new_persons = sorted(persons,key=cmp_to_key(cmp))
print(persons)
print(new_persons)

#################也可以写得更简单一点
def cmp(a,b):
    if a['grade'] != b['grade']:
        return a['grade'] - b['grade']
    elif a['age'] != b['age']:
        return a['age'] - b['age']
    else:
        return a['name'] > b ['name']
```

##### 邻域的一种优雅写法

+ 用一维数组direction，每两两个为一组，表示距离中心点的偏移量

  ```python
  import numpy as np
  
  direction4 = (-1,0,1,0,-1)
  direction8 = (-1,0,1,0,-1,1,1,-1,-1)
  direction9 = (-1,0,0,1,0,-1,1,1,-1,-1)
  
  a = np.arange(9).reshape(3,3)
  center = (1,1)
  res4 = 0
  res8 = 0
  res9 = 0
  
  for k in range(len(direction4)-1):
      x = center[0] + direction4[k]
      y = center[1] + direction4[k+1]
      res4 += a[x][y]
  
  for k in range(len(direction8)-1):
      x = center[0] + direction8[k]
      y = center[1] + direction8[k+1]
      res8 += a[x][y]
      
  for k in range(len(direction9)-1):
      x = center[0] + direction9[k]
      y = center[1] + direction9[k+1]
      res9 += a[x][y]
  
  print(a)
  print(res4,res8,res9)
  ```

##### reduce的用法

+ 类似于降维的作用
+ 调用的合并函数传入两个参数

  ```python
  from functools import reduce
  def add(x, y) :            # 两数相加
      return x + y
  sum1 = reduce(add, [1,2,3,4,5])   # 计算列表和：1+2+3+4+5
  sum2 = reduce(lambda x, y: x+y, [1,2,3,4,5])  # 使用 lambda 匿名函数
  print(sum1)
  print(sum2)
  ················································
  #再比如异或 如果你不用reduce 那么很麻烦 首先定义res 然后for循环每个元素对当前res累积操作
  from functools import reduce
  
  def xor(x, y) :            # 两数相加
      return x ^ y
  reduced = reduce(xor, [1,2,3,4,5])   # 
  ```

##### 函数参数类型

+ 以下面一个例子为例 不同种类的参数可有可无单数他们

  + 位置参数: a b 按照位置传进去
  + 可变参数：*args 可以接受任意参数
  + 仅限关键字参数：注意他比如e f 它长得像位置参数 但是它跟在可变参数后面，表明我不想给他默认值，但是你调用的时候必须传值，而且必须以e=xxx f=xxx传值 为了起强调之意
  + 默认参数：kk g 必须写在位置参数后面
  + 可变关键字参数：**kw

  ```python
  def func(a,b, *args, e, kk=2, f, g='xxx', **kw):
      print('a=', a)
      print('b=', b)
      print('args=', args)
      print('e=', e)
      print('f=', f)
      print('g=', g)
      print('**kw', kw)
      
  func(1,2,3,4, e='a',f='b',g='c', m='123', n='456')
  ```

##### 不要写只带第一个函数的类

+ 看一个例子，关于客户买东西有不同促销策略的例子

  ```python
  from abc import ABC, abstractmethod
  from collections import namedtuple
  
  Customer = namedtuple('Customer', 'name fidelity')
  
  
  class LineItem:
  
      def __init__(self, product, quantity, price):
          self.product = product
          self.quantity = quantity
          self.price = price
  
      def total(self):
          return self.price * self.quantity
  
  
  class Order:  # the Context
  
      def __init__(self, customer, cart, promotion=None):
          self.customer = customer
          self.cart = list(cart)
          self.promotion = promotion
  
      def total(self):
          if not hasattr(self, '__total'):
              self.__total = sum(item.total() for item in self.cart)
          return self.__total
  
      def due(self):
          if self.promotion is None:
              discount = 0
          else:
              discount = self.promotion.discount(self)
          return self.total() - discount
  
      def __repr__(self):
          fmt = '<Order total: {:.2f} due: {:.2f}>'
          return fmt.format(self.total(), self.due())
  
  
  class Promotion(ABC):  # the Strategy: an Abstract Base Class
  
      @abstractmethod
      def discount(self, order):
          """Return discount as a positive dollar amount"""
  
  
  class FidelityPromo(Promotion):  # first Concrete Strategy
      """5% discount for customers with 1000 or more fidelity points"""
  
      def discount(self, order):
          return order.total() * .05 if order.customer.fidelity >= 1000 else 0
  
  
  class BulkItemPromo(Promotion):  # second Concrete Strategy
      """10% discount for each LineItem with 20 or more units"""
  
      def discount(self, order):
          discount = 0
          for item in order.cart:
              if item.quantity >= 20:
                  discount += item.total() * .1
          return discount
  
  
  class LargeOrderPromo(Promotion):  # third Concrete Strategy
      """7% discount for orders with 10 or more distinct items"""
  
      def discount(self, order):
          distinct_items = {item.product for item in order.cart}
          if len(distinct_items) >= 10:
              return order.total() * .07
          return 0
  ```

+ 这个例子有很多值得学习的地方

  + 对于只带属性不带方法的"类"可以用具名元组简化，这就好比struct 和 class
  + hasattr 初始化检查

+ 可以优化之处在于这个抽象基类完全没有必要 直接可以用函数代替

  ```python
  from collections import namedtuple
  
  Customer = namedtuple('Customer', 'name fidelity')
  
  
  class LineItem:
  
      def __init__(self, product, quantity, price):
          self.product = product
          self.quantity = quantity
          self.price = price
  
      def total(self):
          return self.price * self.quantity
  
  
  class Order:  # the Context
  
      def __init__(self, customer, cart, promotion=None):
          self.customer = customer
          self.cart = list(cart)
          self.promotion = promotion
  
      def total(self):
          if not hasattr(self, '__total'):
              self.__total = sum(item.total() for item in self.cart)
          return self.__total
  
      def due(self):
          if self.promotion is None:
              discount = 0
          else:
              discount = self.promotion(self)
          return self.total() - discount
  
      def __repr__(self):
          fmt = '<Order total: {:.2f} due: {:.2f}>'
          return fmt.format(self.total(), self.due())
  
  
  def fidelity_promo(order):
      """5% discount for customers with 1000 or more fidelity points"""
      return order.total() * .05 if order.customer.fidelity >= 1000 else 0
  
  
  def bulk_item_promo(order):
      """10% discount for each LineItem with 20 or more units"""
      discount = 0
      for item in order.cart:
          if item.quantity >= 20:
              discount += item.total() * .1
      return discount
  
  
  def large_order_promo(order):
      """7% discount for orders with 10 or more distinct items"""
      distinct_items = {item.product for item in order.cart}
      if len(distinct_items) >= 10:
          return order.total() * .07
      return 0
  
  ```

##### 排列组合与回溯

```python
#排列数
nums = [1,2,3]
k = 2

def combination(arr):
    if len(arr) == k:
        yield arr[:]
    else:
        for i in range(len(nums)):
            arr.append(nums[i])
            yield from combination(arr)
            arr.pop()

print([item for item in combination([])])
```

```python
#组合数
#这里罗列出不同的组合方式 不考虑数字顺序
nums = [1,2,3]
k = 2


def combination(arr,level):
    if len(arr) == k:
        yield arr[:]
    else:
        for i in range(level,len(nums)):
            arr.append(nums[i])
            yield from combination(arr,i+1)
            arr.pop()


print([item for item in combination([],0)])

```

##### 装饰器计时器

```python
import time

def tiktok(func):
    def inner(*args,**kwargs):
        t0 = time.time()
        res = func(*args,**kwargs)
        t1 = time.time()
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s) -> %r' % (t1-t0, name, arg_str, res))
        return res

    return inner

@tiktok
def fibo(n):
    if n <= 2:
        return 1
    v1 = 1
    v2 = 1

    for _ in range(n-2):
        v1,v2 = v2,v1+v2
    
    return v2

if __name__ == '__main__':
    fibo(100)
```

##### locals()

+ 显示函数本身局部变量的key value 
+ [链接](https://www.runoob.com/python/python-func-locals.html)

##### 带参装饰器

+ 分两种写法 函数带参 和 类带参

+ 函数带参装饰器分3层

  + 第一层是装饰器工厂函数，输入修饰参数，输出装饰器
  + 第二层装饰器
  + 第三层wrapper

  ```python
  DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'
  
  def clock(fmt=DEFAULT_FMT):  # <1>
      def decorate(func):      # <2>
          def clocked(*_args): # <3>
              t0 = time.time()
              _result = func(*_args)  # <4>
              elapsed = time.time() - t0
              name = func.__name__
              args = ', '.join(repr(arg) for arg in _args)  # <5>
              result = repr(_result)  # <6>
              print(fmt.format(**locals()))  # <7>
              return _result  # <8>
          return clocked  # <9>
      return decorate  # <10>
  
  @clock
  def snooze(seconds):
      time.sleep(seconds)
  ```

+ 类带参(推荐写成这样)

  + init负责接收参数
  + call负责装饰器函数
  + wrapper封装函数

  ```python
  DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'
  
  class clock:
  
      def __init__(self, fmt=DEFAULT_FMT):
          self.fmt = fmt
  
      def __call__(self, func):
          def clocked(*_args):
              t0 = time.time()
              _result = func(*_args)
              elapsed = time.time() - t0
              name = func.__name__
              args = ', '.join(repr(arg) for arg in _args)
              result = repr(_result)
              print(self.fmt.format(**locals()))
              return _result
          return clocked
  
  @clock
  def snooze(seconds):
      time.sleep(seconds)
  ```

##### 链式调用

+ 类本身不存在的属性getattr捕获，赋与Chain对象

+ Chain调用时返回Chain对象

  ```python
  class Chain(object):
      def __init__(self, path=''):
         self.__path = path
  
      def __getattr__(self, path):
         return Chain('%s/%s' % (self.__path, path))
  
      def __call__(self, path):
         return Chain('%s/%s' % (self.__path, path))
  
      def __str__(self):
         return self.__path
  
      __repr__ = __str__
  
  print(Chain().users('michael').repos)
  ```

##### os判断文件，文件夹，扩展名

+ 尽量用os的函数，不要自己写诸如‘.py’ in xxx这种

  ```python
  [item for item in os.listdir() if os.path.isdir()]
  [item for item in os.listdir() if os.path.isfile() and os.path.splitext(item)[1]=='.py']
  
  ```

##### 贴标签，浅拷贝，深拷贝

+ 总则，**id是唯一数值标注，在对象生命周期内绝不会改变！**（is即是比较id，==是调用__ eq __，）

+ 对于可变对象

  + = 相当于贴标签，id不会变，对于不可变对象也是一样，区别如下

    ```python
    In [1]: a = 1
    
    In [2]: id(a)
    Out[2]: 1382265056
    
    In [3]: b=a
    
    In [4]: id(b)
    Out[4]: 1382265056
    
    In [5]: a += 1
    
    In [6]: id(a)
    Out[6]: 1382265088  #不能原地操作 只能撕掉标签重新开辟空间 重新贴
    
    In [7]: id(b)
    Out[7]: 1382265056
    
        ############################
    
    In [8]: a = [1]
    
    In [9]: b = a
    
    In [10]: id(a),id(b)
    Out[10]: (1382454589768, 1382454589768)
    
    In [11]: a.append(2)
    
    In [12]: id(a),id(b)
    Out[12]: (1382454589768, 1382454589768)  #id没有改变 原地操作
    
    In [13]: a,b
    Out[13]: ([1, 2], [1, 2])
    ```

  + [:],list()，copy.copy()，list.copy等相当于浅copy，对于numpy数组之类，[:]已经足够了

    ```python
    arr = [{'name': 'wcl', 'age': 23}, {'name': 'wjy', 'age': 14}]
    arr2 = arr.copy()
    del arr[1]
    arr[0]['age'] = 18
    print('arr', arr)
    print('arr2', arr2)
    ########输出#########
    arr [{'name': 'wcl', 'age': 18}]
    arr2 [{'name': 'wcl', 'age': 18}, {'name': 'wjy', 'age': 14}]
    ```

  + 如果列表里有列表，[:]和list只是做浅copy，列表中的列表仍然没有copy相当于贴标签。深拷贝需要copy.deepcopy()

    

  

  



# OpenCV

##### 如何任意比例调整显示框

```python
import numpy as np
import cv2

img = cv2.imread('test.jpg')
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

##### waitKey获取按键

```python
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20)&0xFF==27:
        break
cv2.destroyAllWindows()
```





# Torch

##### mish实现方式

+ 用官方实现就行，不要将公式展开算，自己写没有优化的。
+ <img src="README.assets\image-20210527102536687.png" alt="image-20210527102536687" style="zoom:67%;" />

