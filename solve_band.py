# Purpose: Read the Hamiltonian matrix and overlapping matrix of the non-orthogonal basis set (both real space and inverse space Hamiltonians can be processed), and then solve the generalized eigenvalue problem to find the energy band.
import numpy as np
import sys
from numpy.linalg import eig as SolveGeneral
from numpy.linalg import eigh as SolveHermitian
from scipy.linalg import eigh as SolveGeneralizedHermitian
import matplotlib.pyplot as plt

# ATTENTION: 
#   JKQ:
#   Chemical potential (Fermi level) here is set as VBM !
#   But in FHI-aims, it lies between VBM and CBM because of brodening of fermi distribution.
#   In general, set which energy zero is not so important, because it's the Energy Difference that really matter.
#   However, when compare with aims band structures, we should set same energy zero.


def read_overlap_matrix(path):
    # This dat is a 2D array store each line constent.
    dat = [line.split() for line in open(path, 'r').readlines()]
    print('overlap line1: ',dat[0])
    kp_frac = np.array(list(map(float, dat[0][15:])))

    print(kp_frac)

    for i in range(3):

        # NOTE: dat.pop() will return the element it deleted but not the resulting list !!!
        # Thus we should not use dat = dat.pop(), but only dat.pop()
        
        dat.pop(0)
    dim = len(dat)

    print('overlap matrix with dim: ',dim)    
    ovlp_mat = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        elements = []
        elements = np.append(elements, [complex(float(val[2*i]), float(val[2*i+1])) for val in dat])
        #print(elements)
        ovlp_mat[i,:] = elements
#    print('ovlp_mat: \n', ovlp_mat)
#    print(np.all(np.isclose(ovlp_mat, ovlp_mat.T)))
    if not np.all(np.isclose(ovlp_mat, ovlp_mat.T.conjugate())):
        raise Exception('Warning: Overlap matrix does not obey symmetry.')

    return dim, ovlp_mat, kp_frac

def read_hamiltonian_matrix(path):
    dat = [line.split() for line in open(path, 'r').readlines()]
    for i in range(3):

        # NOTE: dat.pop() will return the element it deleted but not the resulting list !!!
        # Thus we should not use dat = dat.pop(), but only dat.pop()
        
        dat.pop(0)
    dim = len(dat)
    print('Hamiltonian matrix with dim: ',dim)
    hamilton_mat = np.zeros((dim,dim),dtype=complex)
    for i in range(dim):
        elements = []
        elements = np.append(elements, [complex(float(val[2*i]), float(val[2*i+1])) for val in dat])
        #print(elements)
        hamilton_mat[i,:] = elements
#    print('ovlp_mat: \n', ovlp_mat)
#    print(np.all(np.isclose(hamilton_mat, hamilton_mat.T)))
    if not np.all(np.isclose(hamilton_mat, hamilton_mat.T.conjugate())):
        raise Exception('Warning: Hamiltonian matrix does not obey symmetry.')

    return dim, hamilton_mat


def diagonalize_GEV(H, S):
    val, vec = SolveGeneralizedHermitian(np.copy(H), b=np.copy(S))

    return val, vec

def check_ortho(dim, vecs, S=None):
    if (S is None):
        dot = np.dot( np.transpose(np.conjugate( vecs )), vecs)
    else:
        dot = np.dot( np.transpose(np.conjugate(vecs)), np.dot(S, vecs))
    for i in range(dim):
        for j in range(dim):
            if ((i==j) and (np.abs(dot[i,j] - 1.0) > 1.0e-8)):
                print('VECTOR NOT Normalized !', dot[i,j])
            if ((i!=j) and (np.abs(dot[i,j]) > 1.0e-8)):
                print("VECTOR NOT orrthogonal",dot[i,j])

def store_eval_evecs(n_basis, n_state,n_spin,n_kpoints):

    # JKQ: here evals should be real, evecs should be complex
    k_evals = np.zeros((n_state,n_spin,n_kpoints))
    k_evecs = np.zeros((n_basis,n_state,n_spin,n_kpoints),dtype='complex')
    kp_fracs = np.zeros((n_kpoints,3))

    for i in range(1,n_kpoints+1):
        print(i)
        ovlp_file = 'KS_overlap_matrix.band_1.kpt_' + str(i) + '.out'
        hamilton_file = 'KS_hamiltonian_matrix.band_1.kpt_' + str(i) + '.out'
        dim_s, ovlp_mat, kp_frac = read_overlap_matrix(ovlp_file)
        dim_h, hamilton_mat = read_hamiltonian_matrix(hamilton_file)
        if dim_s != dim_h:
            print('ERROR: Dimension of overlap matrix is not identical with hamiltonian matrix!')
        
        # Diagnoalize
        evals, evecs = diagonalize_GEV(hamilton_mat, ovlp_mat)
        # JKQ: current only for n_spin = 1
        print(evals)
        k_evals[:,0,i-1] = evals
        k_evecs[:,:,0,i-1] = evecs
        kp_fracs[i-1,:] = kp_frac
    return k_evals, k_evecs, kp_fracs    


def grep_fermi_level(aims_out):
    # Grep fermi level from aims.out file, i.e. the converged SCF fermi energy.
    scf_fermis = []
    i = 0 
    for line in open(aims_out, 'r').readlines():
        
        if ('| Chemical potential (Fermi level):' in line ):
            i += 1
#            print(i, line)
            scf_fermis = np.append(scf_fermis, line.split()[5])
        
        if ('| Number of self-consistency cycles' in line ):
            n_scf = int(line.split()[6])

    # JKQ: There are (n_scf + 1) fermi levels in aims.out, because before scf there should be an initial condition.
    if n_scf != i-1:
        print('ERROR: number of scf cycles != output fermi levels - 1')
        e_fermi = 0
    else:
        e_fermi = float(scf_fermis[-1])
    
        print('Read from aims.out: total ',n_scf, 'scf cycles, and converged fermi level is: ', e_fermi)

    return e_fermi

def reciprocal_lv(geometry_in):
    latvec = []
    fin1 = open(geometry_in,"r")
    for line in fin1:
        line = line.split("#")[0]
        words = line.split()
        if len(words) == 0:
            continue
        if words[0] == "lattice_vector":
            if len(words) != 4:
                raise Exception("geometry.in: Syntax error in line '"+line+"'")
            latvec += [ np.array(list(map(float,words[1:4]))) ]

    if len(latvec) != 3:
        raise Exception("geometry.in: Must contain exactly 3 lattice vectors")

    latvec = np.asarray(latvec)

    print("Lattice vectors:")
    for i in range(3):
        print(latvec[i,:])
    print()

    #Calculate reciprocal lattice vectors                                                                                                
    rlatvec = []
    volume = (np.dot(latvec[0,:],np.cross(latvec[1,:],latvec[2,:])))
    rlatvec.append(np.array(2*np.pi*np.cross(latvec[1,:],latvec[2,:])/volume))
    rlatvec.append(np.array(2*np.pi*np.cross(latvec[2,:],latvec[0,:])/volume))
    rlatvec.append(np.array(2*np.pi*np.cross(latvec[0,:],latvec[1,:])/volume))
    rlatvec = np.asarray(rlatvec)  # JKQ: reciprocal lattice vectors

    #rlatvec = inv(latvec) Old way to calculate lattice vectors
    print("Reciprocal lattice vectors:")
    for i in range(3):
        print(rlatvec[i,:])
    return rlatvec

def prepare_axis_plot(hartree, e_fermi, k_evals, kp_fracs, rlatvec):
    # JKQ: k_evals[ n_state, n_spin, n_kpoints ]
    # Here we only considered no-spin polarized case, thus k_evals[:,0,:] 
    y_values = (k_evals[:,0,:]*hartree - e_fermi) # y_values[n_state, n_kpoints]
    x_values = np.zeros(np.shape(y_values))

    # JKQ: convert fractional coordinate to Cartesian coordinate and calculate distance
    k_start = np.dot(rlatvec, kp_fracs[0,:])

    for i in range(np.size(x_values,1)):
        k_coord = np.dot(rlatvec, kp_fracs[i,:])
        distance = np.linalg.norm(k_coord - k_start)
        x_values[:, i] = distance
    print('eval: ', y_values)
    print('K coordinates: ', x_values)
    return x_values, y_values

def plot_band(x_values, y_values, e_fermi):
    xaxis = x_values.flatten()
    yaxis = y_values.flatten()
    fig = plt.scatter(xaxis, yaxis, marker='o', color='b',s = 8)
    plt.hlines(0,0,np.max(x_values), color = 'r', linestyles='--', linewidth=1)
    plt.xlim(0, np.max(x_values))
    plt.xticks([])
    plt.show()


# MAIN ROUTINE
# current only for one band segment !!!
hartree = 27.2113845
n_kpoints = 64
n_spin = 1
aims_out = 'aims.out'
names = locals()  # for dynamic variable names
geometry = 'geometry.in'
r_lv = reciprocal_lv(geometry)

ovlp_file = 'KS_overlap_matrix.band_1.kpt_1.out'
hamilton_file = 'KS_hamiltonian_matrix.band_1.kpt_1.out'
dim_s, ovlp_mat, kp_here = read_overlap_matrix(ovlp_file)
dim_h, hamilton_mat = read_hamiltonian_matrix(hamilton_file)
# TEST
val, vec = diagonalize_GEV(hamilton_mat, ovlp_mat)
#e_fermi = -8.72500520

print('eval: ', val)
for i in range(dim_h):
    print('eval: ', val[i])
    print('evecs: \n', vec[:,i])

e_fermi = grep_fermi_level(aims_out)
print(type(e_fermi))
print('evals eV: ',(val*hartree-e_fermi))
k_evals, k_evecs, kp_fracs = store_eval_evecs(dim_s, dim_h, n_spin, n_kpoints)
print('KP_frac: ', kp_fracs)
x_values, y_values = prepare_axis_plot(hartree, e_fermi, k_evals, kp_fracs, r_lv)
plot_band(x_values, y_values, e_fermi)
print('eval32: ', k_evals[:,0,32] )
print('K-point20: ', kp_fracs[19,:])
print('eval20: ', k_evals[:,0,19])
#print('evec1 here: ', k_evals[0,0,0],k_evecs[:,0,0,0])



# val, vec = diagonalize_GEV(hamilton_mat, ovlp_mat)

# #print('eigenvalues at 1 k: \n', val)

# ovlp_file2 = 'KS_overlap_matrix.band_1.kpt_32.out'
# hamilton_file2 = 'KS_hamiltonian_matrix.band_1.kpt_32.out'
# dim_s, ovlp_mat2 = read_overlap_matrix(ovlp_file2)
# dim_h, hamilton_mat2 = read_hamiltonian_matrix(hamilton_file2)
# val2, vec2 = diagonalize_GEV(hamilton_mat2, ovlp_mat2)
# print(np.max(val2))
# print(val2)
# fermi_level = -0.32316252
# aims_fermi = -8.72500514
# print('fermi level: ', fermi_level*hartree)
# print('eigenvalues: \n', (val-fermi_level)*hartree)

# print('eigenvalue with aims fermi level: ', val*hartree - aims_fermi)






