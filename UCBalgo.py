import math


class UCB:
    def __init__(self):
        print("UCB Init")  # debug
        pass

    def action(self, TestBed, c=0.9):
        actionArry = [None] * len(TestBed.BanditArmsArr)
        for i in range(len(TestBed.BanditArmsArr)):
            if TestBed.BanditArmsArr[i].timesPulled > 0:
                actionArry[i] = TestBed.AverageRewardArm(i) + c * math.sqrt( (math.log(TestBed.iteration)) / TestBed.BanditArmsArr[i].timesPulled )
            else:
                actionArry[i] = 1
        action = actionArry.index(max(actionArry))
        print("actionArry: {}, armSelected: {}".format(actionArry, action))  # debug
        return action

    def updateQ(self, TestBed, index, alpha=0.1):
        return TestBed.ProbArr[index] + (1/TestBed.iteration) * (TestBed.AverageRewardArm(index) - TestBed.ProbArr[index])
