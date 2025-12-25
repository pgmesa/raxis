
import numpy as np
import raxis as rx

data = [1.0, 2.0, 3.0, 4.0]
shape = (2,2)

arr = rx.NDArray(data, shape)
sum_rx = arr + arr
print(sum_rx)

arr_np = np.array(data)
sum_np = arr_np + arr_np
sum_np = sum_np.reshape(shape)
print(sum_np)