from mutual_information_estimators import smile_estimate_mutual_information

def Psi_loss(X_t0, X_t1, f_supervenient, causal_decoupling_critic, downward_causation_critic, device, clip):
    """
    params:
        X_t0: (batch_size, 6) tensor representing a batch of 6-digit bit strings at time t
        X_t1: (batch_size, 6) tensor representing a batch of 6-digit bit strings at time t+1
        f_supervenient: a network that takes in a 6-digit bit string and outputs a 
            scalar supervenient feature
        causal_decoupling_critic: a critic network that takes in a supervenient feature at t and t+1
            and outputs an 8-dimensional vector
        downward_causation_critic: a critic network that takes in a supervenient feature at time t+1
            and a bit string at time t and outputs a scalar

    returns:
        the scalar value of the Psi loss
    """

    V_t0 = f_supervenient(X_t0)
    V_t1 = f_supervenient(X_t1)

    print("V_t0")
    print(V_t0)
    print(V_t0.shape)
    print("V_t1")
    print(V_t1)
    print(V_t1.shape)
    print('--------------------------')

    # compute the MI between V_t0 and V_t1
    causal_decoupling_MI = smile_estimate_mutual_information(V_t0, V_t1, causal_decoupling_critic, device, clip=clip)

    # compute the MI between V_t1 and each individual bit of X_t0
    downward_causation_MI = 0
    for i in range(6):
        print(f"X_t0[:, {i}]")
        print(X_t0[:, i])
        downward_causation_MI += smile_estimate_mutual_information(V_t1, X_t0[:, i], downward_causation_critic, device, clip=clip)
    
    return causal_decoupling_MI - downward_causation_MI
