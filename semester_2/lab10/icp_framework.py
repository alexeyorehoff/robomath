from misc_tools import *

''' performs closest point matching of two point sets
      
    Arguments:
    x -- reference point set
    p -- point set to be matched with the reference
    
    Output:
    p_matched -- reordered p, so that the elements in p match the elements in x
'''


def closest_point_matching(x, p):

    p_matched = p

    for i in range(p.shape[1]):
        min_dist = float('inf')
        closest_idx = 0
        
        for j in range(x.shape[1]):
            dist = math.pow(p[0, i] - x[0, j], 2) + math.pow(p[1, i] - x[1, j], 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = j
                
        p_matched[:, i] = x[:, closest_idx]

    return p_matched


def icp(x, p, do_matching):
    p0 = p
    for i in range(10):
        # calculate RMSE
        rmse = 0
        for j in range(p.shape[1]):
            rmse += math.pow(p[0, j] - x[0, j], 2) + math.pow(p[1, j] - x[1, j], 2)
        rmse = math.sqrt(rmse / p.shape[1])

        # print and plot
        print("Iteration:", i, " RMSE:", rmse)
        plot_icp(x, p, p0, i, rmse)

        # data association
        if do_matching:
            p = closest_point_matching(x, p)

        # subtract center of mass
        mx = np.transpose([np.mean(x, 1)])
        mp = np.transpose([np.mean(p, 1)])
        x_prime = x - mx
        p_prime = p - mp

        # singular value decomposition
        w = np.dot(x_prime, p_prime.T)
        u, s, v = np.linalg.svd(w)

        # calculate rotation and translation
        r = np.dot(u, v.T)
        t = mx - np.dot(r, mp)

        # apply transformation
        p = np.dot(r, p) + t

    return


def main():
    x, p1, p2, p3, p4 = generate_data()

    # icp(x, p1, False)
    # icp(x, p2, False)
    icp(x, p3, True)
    # icp(x, p4, True)

    plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
