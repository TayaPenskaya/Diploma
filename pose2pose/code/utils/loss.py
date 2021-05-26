from tqdm import tqdm
import numpy as np

def calc_bones(joints):
    bones = [(0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(6,7),(7,8),(8,9),(10,11),(11,12),(12,7),(7,13),(13,14),(14,15)]
    dists = []
    for bone in bones:
        j1 = joints[bone[0]]
        j2 = joints[bone[1]]
        
        dist = ((j1[0] - j2[0])**2 + (j1[1] - j2[1])**2)**(1/2)
        dists.append(dist)
    return dists

def calc_coef(bones):
    rate = []
    for i in range(len(bones) - 1):
        for j in range(i+1, len(bones)):
            if (bones[i] == 0 or bones[j] == 0):
                rate.append(np.inf)
            else:
                r_ = bones[j]/bones[i]
                rate.append(r_)
    return rate

def calc_joints_der(joints):
    res = []
    res.append(abs(joints[3][1] - joints[6][1]))
    res.append(abs(joints[2][1] - joints[6][1]))
    res.append(abs(joints[12][1] - joints[7][1]))
    res.append(abs(joints[13][1] - joints[7][1]))
    res.append(abs(joints[8][0] - joints[7][0]))
    return res

def get_medians(dataset):
    all_coefs = []

    for i in tqdm(range(len(dataset))):
    
        joints = dataset[i][0].numpy()
        bones = calc_bones(joints)
        coefs = calc_coef(bones)
    
        coefs_ = coefs + calc_joints_der(joints)
        all_coefs.append(coefs_)
    
    all_coefs = np.array(all_coefs)
    medians = []

    for i in range(len(all_coefs[0])):
        c_ = all_coefs[:,i]
        c_ = c_[~np.isinf(c_)]
        medians.append(np.median(c_))

    return medians

def calc_bones_loss(joints, medians):
    losses = []
    for j in joints:
        loss = 0
        coefs = calc_coef(calc_bones(j))
        coefs_ = coefs + calc_joints_der(joints)
        for i in range(len(medians)):
            loss += abs(medians[i] - coefs_[i])
        losses.append(loss)

    return losses
