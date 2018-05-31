import numpy as np

# Мат ожидание - говоря простым языком, это среднеожидаемое значение при многократном повторении испытаний. 
def expected_value(xs, ps):  
    return sum(xs * ps)


def expected_value2(xs, ps):
    summ = 0
    for x, p in zip(xs,ps):
        summ += x*p
    return summ

'''--------------------------------------------------'''

# Дисперсия - это разброс значений какой то величины от его среднего (мат. ожидания) 
def dispersion(xs, ps):
    return sum(xs**2 * ps) - expected_value(xs, ps)**2
    

def main():
    xs = np.array([0,1,2,3])
    ps = ([0.2, 0.3, 0.4, 0.1])

    xl = [0,1,2,3]
    pl = [0.2, 0.3, 0.4, 0.1]

    print(expected_value(xs,ps))
    print(expected_value2(xs,ps))
    print(expected_value2(xl, pl))

    print(dispersion(xs, ps))

if __name__ == "__main__":
    main()


