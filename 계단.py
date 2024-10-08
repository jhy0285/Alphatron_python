import sys

n, total = map(int, sys.stdin.readline().rstrip().split(" "))
coin_list = []
for _ in range(n):
    coin_list.append(int(sys.stdin.readline().rstrip()))
result = 0

minn = min(coin_list)
while coin_list and minn <= total:
    maxx = max(coin_list)
    if maxx <= total:
        result += total // maxx
        total = total % maxx

    coin_list.remove(maxx)

if (total == 0):
    print(result)
else:
    print(-1)



