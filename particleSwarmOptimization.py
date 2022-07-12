import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#The velocities for each particle are initialized.
def initializeSpeed(dimX, dimY):
    speed=np.zeros((dimY,dimX))
    for row in range(len(speed)):
        for col in range(len(speed[0])):
            speed[row][col]=np.random.uniform(low=-.2, high=.2)
    return speed

#The particles are initialized at a random position within the search space.
def initializeParticles(components, n_particles):
    particles=np.zeros((n_particles,components))
    for particle in range(n_particles):
        for component in range(components):
            particles[particle,component]=np.random.uniform(low=-5,high=5)
    return particles

#Particle inertia and learning factors are initialized.
def inertiaAndLearningFactors():
    inertia=np.random.uniform(low=0.4,high=0.9)
    learningFactorA=np.random.uniform(low=0,high=4)
    learningFactorB=4-learningFactorA
    return inertia, learningFactorA, learningFactorB

#Fitness function
def evaluateObjectiveFuntion(X,Y):
    return X**2+Y**2

#The best overall is calculated
def calculateGbest(oldGbest,particles):
    values=evaluateObjectiveFuntion(particles[:,0],particles[:,1])
    oldValue=evaluateObjectiveFuntion(oldGbest[:,0],oldGbest[:,1])
    if min(values)<oldValue:
        return particles[np.where(values==min(values))[0]]
    else:
        return oldGbest

#The best one is calculated for each particle
def calculatePbest(oldPbest,particles):
    pbest=np.zeros((len(particles),len(particles[0])))
    oldValues=evaluateObjectiveFuntion(oldPbest[:,0],oldPbest[:,1])
    newValues=evaluateObjectiveFuntion(particles[:,0],particles[:,1])
    for particle in range(len(particles)):
        if oldValues[particle]<newValues[particle]:
            pbest[particle]=oldPbest[particle]
        else:
            pbest[particle]=particles[particle]
    return pbest

#Update Particles' position
def updatePosition(particles,speed):
    particles+=speed

#plot the particles
def plotFunction(X,Y, iteration):
    x, y = np.array(np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100)))
    z = evaluateObjectiveFuntion(x, y)
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    plt.figure(figsize=(8,6))
    plt.imshow(z, extent=[-10, 10, -10, 10], origin='lower', cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
    contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    plt.scatter(X, Y)
    plt.title(f"iteration: {iteration}")
    plt.show()

#Update Particles' speed
def updateSpeed(inertia, learningFactorA, learningFactorB, speed, pbest, particles, gbest):
    speed=inertia*speed
    aux1=inertia*speed
    aux2=(np.random.uniform()*learningFactorA*(pbest-particles))
    aux3=(np.random.uniform()*learningFactorB*(gbest-particles))
    new_speed=aux1+aux2+aux3
    return new_speed

#Main function
def main():
    information=np.zeros((20,6))
    for iteration in range(100):
        print(f"Iteracion: {iteration}")
        if iteration==0:
            particles=initializeParticles(2,20)
            speed=initializeSpeed(len(particles[0]),len(particles))
            inertia, learningFactorA, learningFactorB=inertiaAndLearningFactors()
            pbest=particles.copy()
            objetiveFuntion=evaluateObjectiveFuntion(particles[:,0],particles[:,1])
            gbest=particles[np.where(objetiveFuntion==min(objetiveFuntion))[0]]
            information[:,0:2]=particles
            information[:,2:4]=speed
            information[:,4:6]=pbest
            plotFunction(particles[:,0],particles[:,1],iteration)
            print(pd.DataFrame(information, columns=["Posicion x","Posicion y","Velocidad x","Velocidad y", "Pbest x","Pbest y"]))
            print(pd.DataFrame(gbest,columns=["Gbest x","Gbest y"]))
            speed=updateSpeed(inertia,learningFactorA,learningFactorB,speed,pbest,particles,gbest)
            updatePosition(particles,speed)
        else:
            gbest=calculateGbest(gbest.copy(),particles)
            pbest=calculatePbest(pbest.copy(),particles)
            information[:,0:2]=particles
            information[:,2:4]=speed
            information[:,4:6]=pbest
            print(pd.DataFrame(information, columns=["Posicion x","Posicion y","Velocidad x","Velocidad y", "Pbest x","Pbest y"]))
            print(pd.DataFrame(gbest,columns=["Gbest x","Gbest y"]))
            plotFunction(particles[:,0],particles[:,1],iteration)
            speed=updateSpeed(inertia,learningFactorA,learningFactorB,speed,pbest,particles,gbest)
            updatePosition(particles,speed)

main()

