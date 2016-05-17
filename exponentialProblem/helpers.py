def plot_gradients(params,num_samples,num_particles):
    for i in range(100):
        samples = sample_theta(params,num_samples)
        LB = lower_bound(params,samples,num_particles)
        g = -grad_KL(params, samples,num_particles,LB)
        all_gradients.append(g)
    plt.hist(all_gradients)
    plt.show()
