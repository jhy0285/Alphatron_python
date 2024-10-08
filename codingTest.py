import sys

a = int(sys.stdin.readline())

li=[0 for _ in range(a+1)]

li[0]=0
li[1]=1
li[2]=2

for i in range(3,a+1):
    li[i]=li[i-1]+li[i-2]

print(li[-1]%796796)
