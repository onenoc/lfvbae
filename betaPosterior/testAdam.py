def f(x):
    return x**2

def g(x):
    return 2*x

def iterate(theta,m,v,t):
    a = 0.01
    b_1 = 0.9
    b_2 = 0.999
    e = 10e-8
    alpha = 0.1
    g_t = g(theta)
    m = b_1*m+(1-b_1)*g_t
    v = b_2*v+(1-b_2)*(g_t**2)
    m_h = m/(1-b_1**t)
    v_h = v/(1-b_2**t)
    theta = theta-alpha*m_h/(v_h**(0.5)+e)
    return theta,m,v

if __name__=='__main__':
    m=0
    v=0
    t=1
    theta = 20
    for i in range(100):
        theta,m,v = iterate(theta,m,v,t)
        t +=1
        print theta
    
    
