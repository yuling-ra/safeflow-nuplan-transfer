import numpy as np
import matplotlib.pyplot as plt

class Robot():
    def __init__(self, lengths=[3,2.5,1,0.5], initAngles=[91,90,90,90], maxAngles=[180, 180, 170, 160], minAngles=[45, -80, -80, -80]):
        assert (len(lengths) == len(initAngles)), "Number of links(lengths passed) and angles should be equal"
        self._armLengths = np.array(lengths)
        self._angles = np.radians(np.array(initAngles))
        self._gradients = np.zeros(len(initAngles))
        self._minAngles = np.radians(np.array(minAngles))
        self._maxAngles = np.radians(np.array(maxAngles))

    @property
    def armLengths(self):
        return self._armLengths
    
    @armLengths.setter
    def armLengths(self, newArmLengths):
        if len(newArmLengths) == len(self.armLengths):
            self._armLengths = newArmLengths        
        else:
            print("Length of arm lengths list should be equal to the len of the existing arm lengths array")

    
    @property
    def angles(self):
        return self._angles
    
    @angles.setter
    def angles(self, newAngles):
        if len(newAngles) == len(self.angles):
            self._angles = newAngles
        else:
            print("Length of angles list should be equal to the lenght of the existing angles array")

    
    @property
    def gradients(self):
        return self._gradients
    
    @gradients.setter
    def gradients(self, newGradients):
        if len(newGradients) == len(self.angles):
            self._gradients = newGradients
        else:
            print("Length of gradients list should be equal to the lenght of the existing angles array")
    
    @property
    def minAngles(self):
        return self._minAngles
    
    @minAngles.setter
    def minAngles(self, newMinAngles):
        if len(newMinAngles) == len(self.angles):
            self._minAngles = newMinAngles
        else:
            print("Length of minimum angles list should be equal to the lenght of the existing angles array")
    
    @property
    def maxAngles(self):
        return self._maxAngles
    
    @maxAngles.setter
    def maxAngles(self, newMaxAngles):
        if len(newMaxAngles) == len(self.angles):
            self._maxAngles = newMaxAngles
        else:
            print("Length of maximum angles list should be equal to the lenght of the existing angles array")

    def __str__(self) -> str:
        ouput = f"Arm Lengths = {self.armLengths} \nInitial Angles(Radians) = {self.angles}"
        return ouput

    
    def forwardKinematics(self, angles = None):
          if angles is None:
              angles = self.angles
          x = np.sum(self.armLengths[:] * np.cos( angles[:]))
          y = np.sum(self.armLengths[:] * np.sin( angles[:]))

          return np.array([x,y])
              

    def poseCompute(self, angles=np.radians(np.array([90,90,90,90]))):
        assert (len(self.armLengths) == len(angles)), "Number of links and angles should be equal"
        joints = np.zeros((self.armLengths.shape[0]+1,2)) # joint positions

        for i in range(len(self.armLengths)):
            joints[i+1] = joints[i] + np.array([self.armLengths[i] * np.cos(angles[i]), 
                                                self.armLengths[i] * np.sin(angles[i])])


        return joints
        

    def distToTarget(self, r = None,  targetPoint=[1,1]):
        t = np.array(targetPoint)
        if r is None:
            r = self.forwardKinematics()

        dist = None
        ##########################################################################
        # TODO:                                                                  #
        # Implement the distance computation and return the gradient.            #
        # The returned distance should be a single value representing the        #
        # Euclidean distance between the robot's endpoint and the target point.  #
        ##########################################################################

        dist =  np.sqrt(np.sum(np.square(t-r)))

        ##########################################################################
        #                            END OF YOUR CODE                            #
        ##########################################################################

        return dist
    
    def gradient(self, targetPoint=[1,1]):
        t = np.array(targetPoint)
        r = self.forwardKinematics()

        grad = None
        ##########################################################################
        # TODO:                                                                  #
        # Implement the gradient computation and return the gradient.            #
        # The returned gradients should be a 1D numpy array that is as long as   #
        # the self.angles array.                                                 #
        ##########################################################################

        grad  =  (2 * (t[0] - r[0]) * self.armLengths[:] * np.sin(self.angles[:]) -
                       2 * (t[1] - r[1]) * self.armLengths[:] * np.cos(self.angles[:]) )
        
        ##########################################################################
        #                            END OF YOUR CODE                            #
        ##########################################################################

        return grad
    
    def gradientFDM(self, targetPoint=[1,1], h=1e-3):
        t = np.array(targetPoint)
        
        grad = np.zeros_like(self.angles)

        ##########################################################################
        # TODO:                                                                  #
        # Implement the gradient computation and return the gradient.            #
        # The returned gradients should a 1D numpy array that is as long as      #
        # the self.angles array.                                                 #
        ##########################################################################
        angles = np.copy(self.angles)

        for i in range(len(self.angles)):
            anglesNew = np.copy(angles)
            anglesNew[i] += h
            grad[i] = (self.distToTarget(self.forwardKinematics(anglesNew), targetPoint=t) 
                       - self.distToTarget(self.forwardKinematics(angles), targetPoint=t)) / h

        ##########################################################################
        #                            END OF YOUR CODE                            #
        ##########################################################################

        return grad
    
    def constrain(self):
        self.angles = np.clip(self.angles, self.minAngles, self.maxAngles)
    
    def Solve(self, targetPoint=[1,1], alpha=1e-2, distThreshold=1e-2, maxIter=1000):
        t = np.array(targetPoint)
        err = self.distToTarget(targetPoint=t)
        grads = []
        iter = 0
        self.solnAngles, self.solnError = [np.copy(self.angles)], [err]

        while err > distThreshold:
            
            grad = self.gradient(t)
            grads.append(grad)
            # Update step
            ##########################################################################
            # TODO:                                                                  #
            # Implement the explicit forward Euler step such that the array          #
            # self.angle is upated with the new angles.                              #
            ##########################################################################

            self.angles[:] = self.angles[:] - alpha * grad[:]

            ##########################################################################
            #                            END OF YOUR CODE                            #
            ##########################################################################

            self.constrain()
            err = self.distToTarget(targetPoint=t)
            self.solnAngles.append(np.copy(self.angles))
            self.solnError.append(err)
            if (np.abs(self.solnError[-1] - self.solnError[-2]) <= 1e-12):
                print("Breaking! Convergence!")
                break
            if (iter > maxIter):
                print("Breaking! MaxIter!")
                break
            iter += 1

        self.grads = grads


    def plotRobot(self):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))
        ax.set_aspect('equal')
        ax.grid()
        joints = self.poseCompute(self.angles)
        line, = ax.plot(joints[:,0], joints[:,1], 'o-', lw=2) # arms of the robot
        plt.show()
            
