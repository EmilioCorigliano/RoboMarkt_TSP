import copy
import itertools
import math
import re
import time
from math import sqrt, ceil

import numpy as np
from matplotlib import pyplot as plt
from ortools.linear_solver import pywraplp

# from sklearn.cluster import KMeans, AgglomerativeClustering

### BEGIN OF CHANGEABLE PARAMETERS
"""Select the data file (path relative to the script folder)"""
filename = "minimart-I-50.dat"

"""
Select the maximum time we are given in order to search for the best solution
"""
time_limit = 8 * 60  # in seconds

"""
maximum length of the group of points for which we will search for the exact optimal solution 
len(cluster)    avg_execution_time[seconds]
9               1
10              10
11              120
"""
maxLenOptimalCluster = 10

### END OF CHANGEABLE PARAMETERS

executionTime = {9: 1, 10: 10, 11: 120}


def dist(point1, point2):
    """calculates the distance among two points"""
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def distIndex(points, i1, i2):
    """calculates the distance among two points represented as indexes"""
    return dist(points[i1], points[i2])


def cycleCost(points, cycle):
    """calculates the total distance of the cycle (from point 0, within lal the cycle till point 0)"""
    s = distIndex(points, 0, cycle[0])

    for i in range(len(cycle) - 1):
        s += distIndex(points, cycle[i], cycle[i + 1])

    s += distIndex(points, cycle[-1], 0)
    return s


def findClosestPoint(points, p, cluster):
    minDist = -1
    minPointIndex = -1
    for c in cluster:
        d = distIndex(points, p, c)
        if (minPointIndex == -1 or minDist > d):
            minPointIndex = c
            minDist = d

    return minPointIndex


def permutationsNoReflected(list):
    """
    generates only the interesting permutations (excludes the reflected ones)
    :param list: list of which we want to generate the permutations
    :return: list of generated permutations
    """
    permutations = []

    for end in itertools.combinations(list, 2):
        middle = [e for e in list if (e not in end)]
        for m in itertools.permutations(middle):
            permutations.append(tuple([end[0]]) + m + tuple([end[1]]))

    return permutations


def calculateRefurbishmentCost(points, clusters, data):
    """
    Calculates the refurbishment cost considering the ordered clusters (tracks)
    :param points: all the houses in the graph
    :param clusters: all the group of markets that the track has to refurnish
    :param data: additional data for the problem
    :return: returns the cost of the refurbishment
    """
    fc = data['Fc']
    vc = data['Vc']

    c = 0
    c += fc * len(clusters)
    for i in range(len(clusters)):
        c += cycleCost(points, clusters[i]) * vc

    return c


def calculateOpeningCost():
    """
    Calculates the cost for opening the mini-markets
    :return: cost for opening the mini-markets
    """
    global built
    c = 0
    for b in built:
        c += points[b][2]

    return c


def showResults(execution_time, f, refurbishment_cost, clusters):
    """
    calculate the total and section costs, prints the graph and writes on file the final solution
    """
    global opening_cost, fig, built

    print()
    print("Execution time: {} seconds".format(execution_time))
    with open(f, 'w') as file:
        print("Total cost: ", opening_cost + refurbishment_cost)
        file.write(str(opening_cost + refurbishment_cost) + "\n")

        print("Opening cost: ", opening_cost)
        file.write(str(opening_cost) + "\n")

        print("Refurbishment cost: ", refurbishment_cost)
        file.write(str(refurbishment_cost) + "\n")

        print("minimarkets: ", str([b + 1 for b in built])[1:-1])
        file.write(str([b + 1 for b in built])[1:-1] + "\n")

        print("Refurbishment routes:")
        for cluster in clusters:
            toPrint = [c + 1 for c in cluster]
            toPrint.append(1)
            toPrint.insert(0, 1)
            file.write(str(toPrint)[1:-1] + "\n")
            print("\t", str(toPrint)[1:-1])


def create_data_model():
    """
    reads the .DAT file and stores the data in
    :return:
    points: all the points of the problem
    data: all the other data, like n, range, capacity, Fc, Vc...
    """

    """Opening and parsing .DAT file"""
    global filename
    with open(file=filename) as file:
        content = file.read()

    # n: number of places
    n = int(re.findall(r"param n :=\s([0-9]+)", content)[0])

    # r: maximum distance among a place and the closest store
    r = int(re.findall(r"param range :=\s([0-9]+)", content)[0])

    # vc: variable cost for each km of trip
    vc = int(re.findall(r"param Vc :=\s([0-9]+)", content)[0])

    # fc: fixed cost for a trip
    fc = int(re.findall(r"param Fc :=\s([0-9]+)", content)[0])

    # capacity: max stores refurnishiable
    capacity = int(re.findall(r"param capacity :=\s([0-9]+)", content)[0])

    # cx, cy, dc, usable
    points = re.findall("\d+(?:\.\d+)?\s(\d+(?:\.\d+)?)\s(\d+(?:\.\d+)?)\s(\d+(?:\.\d+)?)\s(\d+(?:\.\d+)?)", content)

    points = [(float(points[i][0]), float(points[i][1]), float(points[i][2]), int(points[i][3])) for i in
              range(len(points))]

    """Stores the data for the problem."""
    data = {}
    data['obj_coeffs'] = [points[i][2] for i in range(len(points))]
    data['range'] = r
    data['Vc'] = vc
    data['Fc'] = fc
    data['capacity'] = capacity
    data['num_vars'] = n
    data['num_constraints'] = 2 * n

    return points, data


def solveForBuildingCost(points, data):
    """
    Optimally solves the problem of opening the mini-markets minimizing the cost and
    considering the restrictions on the max distance of the houses from a mini-market
    :return: the list of the opened mini-markets
    """
    print("starting opening markets resolution")

    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Create the variable list x
    x = {}
    for j in range(data['num_vars']):
        x[j] = solver.BoolVar('x[%i]' % j)

    # Create the constraints on the building availability
    constraint = solver.RowConstraint(1.0, 1.0, 'first')
    constraint.SetCoefficient(x[0], 1)

    for i in range(data['num_vars']):
        constraint = solver.RowConstraint(0.0, points[i][3], 'usable[%i]' % i)
        constraint.SetCoefficient(x[i], 1)

    for i in range(data['num_vars']):
        constraint = solver.RowConstraint(1.0, solver.infinity(), 'range[%i]' % i)
        for j in range(data['num_vars']):
            if (dist(points[i], points[j]) <= data['range']):
                constraint.SetCoefficient(x[j], 1)

    objective = solver.Objective()
    for j in range(data['num_vars']):
        objective.SetCoefficient(x[j], data['obj_coeffs'][j])
    objective.SetMinimization()

    solver.Solve()

    sol = [bool(x[i].solution_value()) for i in range(data['num_vars'])]

    return sol


def printPoints(points, built):
    """
    prints all the different points in different colors:
    grey: houses
    green: mini-market
    red: main branch
    """
    for i in range(len(built)):
        if i == 0:
            ax.scatter(points[i][0], points[i][1], c='red')
        elif built[i]:
            ax.scatter(points[i][0], points[i][1], c='green')
        else:
            ax.scatter(points[i][0], points[i][1], c='gainsboro')


def findMinHamiltonianCycle(points, cluster):
    """
    finds the best cycle
    ATTENTION: this method for 10-11 points is pretty slow (sped up with custom permutation generator),
    for more points is over our time limit.

    len(cluster)    avg_execution_time[seconds]
    9               1
    10              10
    11              120

    :return: cost of the best cycle and the list of ordered indexes
    """

    minCycle = tuple(cluster)
    minCost = cycleCost(points, cluster)

    for c in permutationsNoReflected(cluster):
        cost = cycleCost(points, c)
        if cost < minCost:
            minCost = cost
            minCycle = c

    return minCost, minCycle


def findMinApproximatedHamiltonianCycle(points, cluster):
    """
    finds an approximation of the best cycle (min cost from last added)
    """

    cycle = []
    clusterCopy = copy.copy(cluster)

    while len(clusterCopy) > 0:
        closest = 'x'
        if (len(cycle) == 0):
            closest = findClosestPoint(points, 0, clusterCopy)
        else:
            closest = findClosestPoint(points, cycle[-1], clusterCopy)
        cycle.append(closest)
        clusterCopy.remove(closest)

    minCost = cycleCost(points, cycle)

    return minCost, tuple(cycle)


def findMinApproximatedHamiltonianCycleChoosingFirstLast(points, cluster):
    """
    finds an approximation of the best cycle (min cost from last added and choosing first and last nodes)
    """
    cycle = []
    clusterCopy = copy.copy(cluster)
    last = -1
    while len(clusterCopy) > 0:
        closest = 'x'
        if (len(cycle) == 0):
            closest = findClosestPoint(points, 0, clusterCopy)
        else:
            closest = findClosestPoint(points, cycle[-1], clusterCopy)
        cycle.append(closest)
        clusterCopy.remove(closest)

        if (len(clusterCopy) > 1 and last == -1):
            last = findClosestPoint(points, 0, clusterCopy)
            clusterCopy.remove(last)

    cycle.append(last)

    minCost = cycleCost(points, cycle)

    return minCost, tuple(cycle)


def getMarketsOrderedByAngle():
    global points, built, data

    marketsPolar = []
    toBeChosen = copy.copy(built)
    toBeChosen.pop(0)

    for i in range(len(toBeChosen)):
        if toBeChosen[i]:
            p = points[toBeChosen[i]][0:2]
            # theta
            marketsPolar.append(np.arctan2((p[1] - points[0][1]), (p[0] - points[0][0])))

    # ordering the minimarkets in base of their angle
    return [x for _, x in sorted(zip(marketsPolar, toBeChosen))]


def getClusterGeometricalDivision(marketsOrdered, maxCapacity):
    """
    Division of the space in a geometrical way
    """
    global data

    nClusters = ceil(len(marketsOrdered) / maxCapacity)

    clusters = []

    for i in range(nClusters):
        clusters.append(marketsOrdered[i * maxCapacity:i * maxCapacity + maxCapacity])

    return clusters


def printCycle(points, cycle):
    print(cycle)
    cycle = tuple([0]) + cycle + tuple([0])
    ax.plot([points[p][0] for p in cycle], [points[p][1] for p in cycle])


# starting time
start_time = time.time()

# global data
opening_cost = 0
refurbishment_cost = 0
built = []
clusters = []

### opening of mini markets
points, data = create_data_model()
x = solveForBuildingCost(points, data)
for i in range(len(x)):
    if x[i]:
        built.append(i)

opening_cost = calculateOpeningCost()

print("opened markets: ", built)
print("opening cost: ", opening_cost)

# clustering
minRefurbishmentCost = -1
optimalClusters = []
print("\nstarting computation of clusters")

### ordering the markets by their angle
ordered = getMarketsOrderedByAngle()

maxCapacity = data['capacity']

### Taking the best clustering (with greedy algorithm)
# iterating on different starting positions
for i in range(1, len(built)):
    print("\n{:.3f} iteration {} of {}".format(time.time() - start_time, i, len(built) - 1))
    # iterating on different capacity parameters
    for capacity in range(math.floor(maxCapacity / 2), maxCapacity + 1):  # range(maxCapacity, maxCapacity+1):#
        # generating the clusters
        clusters = getClusterGeometricalDivision(ordered, capacity)
        minimized = []

        for cluster in clusters:
            # FIND MIN HAMILTONIAN CYCLE
            # available algorithms:
            # - findMinHamiltonianCycle (optimal but slow)
            # - findMinApproximatedHamiltonianCycle (pretty good and fast)
            # - findMinApproximatedHamiltonianCycleChoosingFirstLast (worst)
            cost, cycle = findMinApproximatedHamiltonianCycle(points, cluster)
            minimized.append(cycle)

        # generating results
        refurbishment_cost = calculateRefurbishmentCost(points, minimized, data)
        print("capacity {} cost: {:.3f}".format(capacity, refurbishment_cost))
        if minRefurbishmentCost == -1 or minRefurbishmentCost > refurbishment_cost:
            minRefurbishmentCost = refurbishment_cost
            cost_label = opening_cost + refurbishment_cost
            optimalClusters = minimized
            print("new optimal = {:.3f} with {}".format(cost_label, optimalClusters))

    ordered.append(ordered.pop(0))

### optimal
print("\n\nOptimal solution: ")
fig = plt.figure('optimal')
ax = fig.add_subplot()

optimal = []

printPoints(points, x)
for cluster in optimalClusters:
    # calculating the optimal minimum hamiltonian cycle only if cluster length is less than a parameter (otherwise
    # it takes too much time and resources) and if the time remained is enough for the optimal solution
    if len(cluster) <= maxLenOptimalCluster and \
            (time_limit == -1 or
             len(cluster) < 9 or (
                     len(cluster) >= 9 and time_limit > time.time() - start_time +
                     executionTime[len(cluster)]
             )):
        cost, cycle = findMinHamiltonianCycle(points, cluster)
    else:
        cost, cycle = findMinApproximatedHamiltonianCycle(points, list(cluster))
    optimal.append(cycle)
    printCycle(points, optimal[-1])

minRefurbishmentCost = calculateRefurbishmentCost(points, optimal, data)

showResults(time.time() - start_time, filename[:-4] + "-optimal.txt", minRefurbishmentCost, optimal)

cost_label = opening_cost + minRefurbishmentCost
fig.subplots_adjust(top=0.85)
fig.suptitle("{}: {:.3f}".format(filename[:-4], cost_label), fontsize=14, fontweight='bold')
plt.savefig(filename[:-4] + '-optimal.png')
plt.show()
