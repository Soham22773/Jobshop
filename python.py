import numpy as np
import math
import time
import random
import itertools
import queue
import pandas as pd


class flowshop:
    def __init__(self,n,m):
        self.n=n
        self.m=m
    
    def initialization(self,Npop):
        pop = []
        
        for i in range(Npop):
            p = list(np.random.permutation(self.n))
            
            while p in pop:
                p = list(np.random.permutation(self.n))
            
            pop.append(p)
        return pop

    def calculateObj(self,sol):
        m=self.m
        n=self.n
        qTime = queue.PriorityQueue()
        qMachines = []
       
        for i in range(self.m):
            qMachines.append(queue.Queue())
        
        for i in range(self.n):
            qMachines[0].put(sol[i])
        busyMachines = []
       
        for i in range(self.m):
            busyMachines.append(False)
        time = 0
        job = qMachines[0].get()
        qTime.put((time+cost[job][0], 0, job))
        busyMachines[0] = True
   
        while True:
            time, mach, job = qTime.get()
           
            if job == sol[self.n-1] and mach == self.m-1:
                break
            busyMachines[mach] = False
       
            if not qMachines[mach].empty():
                j = qMachines[mach].get()
                qTime.put((time+cost[j][mach], mach, j))
                busyMachines[mach] = True
           
            if mach < self.m-1:
                if busyMachines[mach+1] == False:
                    qTime.put((time+cost[job][mach+1], mach+1, job))
                    busyMachines[mach+1] = True
                else:
                    qMachines[mach+1].put(job)
        return time


    def selection(self,pop):
        popObj = []
        for i in range(len(pop)):
            popObj.append([self.calculateObj(pop[i]), i])
        
        popObj.sort()
        distr = []
        distrInd = []
        
        for i in range(len(pop)):
            distrInd.append(popObj[i][1])
            prob = (2*(i+1)) / (len(pop) * (len(pop)+1))
            distr.append(prob)
        
        parents = []
        for i in range(len(pop)):
            parents.append(list(np.random.choice(distrInd, 2, p=distr)))
       
        return parents

    def crossover(self,parents):
        n=self.n
        pos = list(np.random.permutation(np.arange(n-1)+1)[:2])
        if pos[0] > pos[1]:
            t = pos[0]
            pos[0] = pos[1]
            pos[1] = t
    
        child = list(parents[0])
        for i in range(pos[0], pos[1]):
            child[i] = -1
        
        p = -1
        for i in range(pos[0], pos[1]):
            while True:
                p = p + 1
                if parents[1][p] not in child:
                    child[i] = parents[1][p]
                    break
    
        return child	

    def mutation(self,sol):
        pos = list(np.random.permutation(np.arange(n))[:2])
        if pos[0] > pos[1]:
            t = pos[0]
            pos[0] = pos[1]
            pos[1] = t
        
        remJob = sol[pos[1]]    	
        for i in range(pos[1], pos[0], -1):
            sol[i] = sol[i-1]
        
        sol[pos[0]] = remJob
        return sol

    def elitistUpdate(self,oldPop, newPop):
        bestSolInd = 0
        bestSol = self.calculateObj(oldPop[0])
        
        for i in range(1, len(oldPop)):
            tempObj = self.calculateObj(oldPop[i])
            
            if tempObj < bestSol:
                bestSol = tempObj
                bestSolInd = i
        
        rndInd = random.randint(0,len(newPop)-1)
        newPop[rndInd] = oldPop[bestSolInd]
        return newPop
    
    #index number, best solution,average objective value of the given population.
    def findBestSolution(self,pop):
        bestObj = self.calculateObj(pop[0])
        avgObj = bestObj
        bestInd = 0
        
        for i in range(1, len(pop)):
            tObj = self.calculateObj(pop[i])
            avgObj = avgObj + tObj
            
            if tObj < bestObj:
                bestObj = tObj
                bestInd = i    	        
        
        return bestInd, bestObj, avgObj/len(pop)
	
    def avgTime(self,pop):
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
            
        return avgWTime/n, avgMIdle/m, avgTTime/n
	
#Reading Data


dataset = input("Dataset: ")
if dataset == "0":
    optimalObjective = 42.5	
elif dataset == 4:
    optimalObjective = 661
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

ob=flowshop(n,m)
Npop = int(input("Initial POpulation: "))	# Number of population
Pc = 1.0	# Probability of crossover
Pm = 1.0	# Probability of mutation
maxGeneration = 10000	#maximum generation
t1 = time.clock()

population = ob.initialization(Npop)
for i in range(maxGeneration):
    # Selecting parents
    parents = ob.selection(population)
    childs = []
    #crossover
    for p in parents:
        r = random.random()
        if r < Pc:
            childs.append(ob.crossover([population[p[0]], population[p[1]]]))
        else:
            if r < 0.5:
                childs.append(population[p[0]])
            else:
                childs.append(population[p[1]])
    
    #mutation
    for c in childs:
        r = random.random()
        if r < Pm:
            c = ob.mutation(c)

    # Update population
    population = ob.elitistUpdate(population, childs)
    
    #print(population)
    #print(findBestSolution(population))


t2 = time.clock()

bestSol, bestObj, avgObj = ob.findBestSolution(population)
avgWTime, avgMIdle, avgTTime = ob.avgTime(population[bestSol])

print("Population:")
print(population)
print() 

print("Sequence:")
print(population[bestSol])
print() 

print("MakeSpan Value:")
print(bestObj)
print()

print("Average MakeSpan of Population:")
print("%.2f" %avgObj)
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


