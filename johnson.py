import numpy as np
import math
import time
import random
import itertools
import queue
import pandas as pd

class Johnson:

    def checkCondition():
        min1 = 9999
        min2 = 9999
        condition1 = condition2 = False
        for i in range (0, n):
            min1 = min(min1, cost[i][0])
        for i in range (0, n):
            for j in range (1, m-2):
                min2 = min(min2, cost[i][j])
        if min1 >= min2:
            condition1 = True
        
        min1 = 9999

        for i in range (0, n):
            min1 = min(min1, cost[i][m-1])

        if min1 >= min2:
            condition2 = True
       
        return condition1, condition2
    
    def hypotheticalMachines():
        costHypo = []
        for i in range (0, n):
             sum1 = 0
             temp = []
             for j in range (0, m-1):
                 sum1 = sum1 + cost[i][j]
             temp.append(sum1)
             sum1 = 0
             for j in range (1, m):
                 sum1 = sum1 + cost[i][j]
             temp.append(sum1)
             costHypo.append(temp)
        return costHypo

    def generateSequence(costHypo):
        sequence = []
        complete = []
        last = n-1
        first = 0
        job = 0
        machine = 0
        for k in range (0, n):
            sequence.append(0)
        for i in range (0, n):
            minC = 9999
            for j in range (0, n):
                if j in complete:
                    continue
                for k in range (0, 2):
                    if minC > costHypo[j][k]:
                        minC = costHypo[j][k]
                        machine = k
                        job = j
            if machine == 0:
                sequence[first] = job
                first += 1
                complete.append(job)
                print(first-1)
            else:
                sequence[last] = job
                last -= 1 
                complete.append(job)
                print(last+1)
            print(sequence)
        return sequence
     
    def InOutMatrix(pop):
        avgWTime = 0.0
        avgTTime = 0.0
        avgMIdle = 0.0
        InOut = []
        total = 0
        for i in range (0, len(pop)):
            temp = []
            for j in range (0, len(cost[pop[i]])):
                if i == 0:
                    total = total + cost[pop[i]][j]
                    temp.append(total)
                elif j == 0:
                    total = InOut[i-1][j] + cost[pop[i]][j]
                    temp.append(total)
                else:
                    total = max(InOut[i-1][j], total) + cost[pop[i]][j]
                    temp.append(total)
            InOut.append(temp)
        
        for x in range (0, len(pop)):
            print(cost[pop[x]])
        print("\n")
        for x in range (0, len(pop)):
            print(InOut[x])
        
        for i in range (0, len(pop)):
             for j in range (0, len(cost[pop[i]])):
                if i == 0:
                    avgMIdle = avgMIdle + (cost[pop[i]][j] * (m-1-j))
                elif j == 0:
                    continue
                else:
                    k=j
                    while True:
                        if k == m:
                            break
                        if InOut[i-1][k] < InOut[i][j-1]:
                            avgMIdle = avgMIdle + (InOut[i][k-1] - InOut[i-1][k])  
                        k += 1 
        
        for i in range (0,len(pop)):
            sumI=0
            for j in range (0, len(cost[pop[i]])):
                sumI = sumI + cost[pop[i]][j]
            avgWTime = avgWTime + InOut[i][m-1] - sumI

        for i in range (0,len(pop)):
                avgTTime = avgTTime + InOut[i][m-1]
            
        return avgWTime/n, InOut[n-1][m-1], avgMIdle/m, avgTTime/n


dataset = input("Dataset: ")
if dataset == "0":
    optimalObjective = 82
elif dataset == "1":
    optimalObjective = 4534
elif dataset == "2":
    optimalObjective = 920
else:
    optimalObjective = 1302

filename = "data" + dataset + ".txt"
f = open(filename, 'r')
l = f.readline().split()
n = int(l[0])		#number of jobs
m = int(l[1]) 		#number of machines
cost = []		#cost matrix
    
for i in range(n):
    temp = []
    for j in range(m):
        temp.append(0)
    cost.append(temp)


for i in range(n):
    line = f.readline().split()
    for j in range(int(len(line)/2)):
        cost[i][j] = int(line[2*j+1])
f.close()
t1 = time.clock()

ob = Johnson
condition1, condition2 = ob.checkCondition()
print(condition1,"  ",condition2)
if condition1 == True or condition2 == True:
    costHypo = ob.hypotheticalMachines()
    sequence = ob.generateSequence(costHypo)
    t2 = time.clock()
    avgWTime, bestObj, avgMIdle, avgTTime = ob.InOutMatrix(sequence)

    print("Sequence:")
    print(sequence) 
    print() 

    print("MakeSpan Value:")
    print(bestObj)
    print()

    print("%Gap:")
    G = 100 * (bestObj-optimalObjective) / optimalObjective
    print("%.2f" %G)
    print()

    print("CPU Time (s)")
    timePassed = (t2-t1)
    print("%.2f" %timePassed)
    print()

    print("Average Waiting Time of Jobs:")
    print("%.2f" %avgWTime)
    print() 

    print("Average Turnaround Time :")
    print("%.2f" %avgTTime)
    print() 

    print("Average Machine Idle Time :")
    print("%.2f" %avgMIdle)
    print() 
    
