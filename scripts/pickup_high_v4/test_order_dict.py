from collections import OrderedDict

d = {0: OrderedDict([('action', 0), ('message', 11)]), 1: OrderedDict([('action', 4), ('message', 11)])}

for k, v in d.items():
    print("k", k)
    print("v", v[1])