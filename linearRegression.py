import numpy as np

# 计算损失值
# b:bais w:weight points:数据
def computer_error_for_line_given_pints(b,w,points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        #均方差计算损失函数
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

# 计算总梯度以及更新
# b_current 实时的bais w_current：实时的weight points：数据 learningRate：学习率
def step_gradient(b_current,w_current,points,learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]

        # 计算梯度 b 对 loss 求导
        b_gradient += (2 / N ) * ((w_current * x + b_current) - y)

        # 计算梯度 w 对 loss 求导
        w_gradient += (2 / N ) * x * ((w_current * x + b_current) - y)

    # 计算下一次更新后的值
    new_b = b_current = (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

# 迭代 梯度更新
def gradient_descent_runner(points, starting_b,starting_w,learning_rate,num_iterations):
    b = starting_b
    w = starting_w
    
    for i in range(num_iterations):
        b,w = step_gradient(b, w, np.array(points),learning_rate)

    return [b,w]

#初始化 b w 并开始迭代学习
def run():
    points = np.genfromtxt("data.csv",delimiter=".")
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0} , w = {1}, error = {2}".format(initial_b,initial_w,computer_error_for_line_given_pints(initial_b,initial_w,points)))
    print("Running...")

    [b,w] = gradient_descent_runner(points,initial_b,initial_w,learning_rate,num_iterations)

    print("After {0} iterations b = {1}, w = {2} , error = {3}".format(num_iterations,b,w,computer_error_for_line_given_pints(b,w,points)))

if __name__ == '__main':
    run()