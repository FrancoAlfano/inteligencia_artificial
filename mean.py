import statistics

orders = [0,0,0,0,1,2,3,4,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
average = statistics.mean(orders)

print("The average coffee order price today is $" + str(average))
