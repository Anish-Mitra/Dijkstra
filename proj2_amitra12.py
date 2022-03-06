from cmath import inf
import math
import numpy as np
from queue import PriorityQueue
import cv2

y_flip=249

#Function to plot circle on map 
def plot_circle():
    obstacle_space=np.array([0,255,255])
    clearance=np.array([0,0,255])
    map=np.zeros((250,400,3),dtype="uint8")
    # circle 
    for y in range(250):
        for x in range(400):
            if ((y-(65))**2+(x-(300))**2)<=(45)**2:
                map[y][x]=clearance
            if ((y-(65))**2+(x-(300))**2)<=(40)**2:
                map[y][x]=obstacle_space
    return map

# Function to plot arrow on the map
def arrow_shape():
    obstacle_space=np.array([0,255,255])
    clearance=np.array([0,0,255])
    map=np.zeros((250,400,3),dtype="uint8")
    map_temp1=np.zeros((250,400,3),dtype="uint8")
    map_temp2=np.zeros((250,400,3),dtype="uint8")
    
    map1=np.zeros((250,400,3),dtype="uint8")
    map2=np.zeros((250,400,3),dtype="uint8")
    map3=np.zeros((250,400,3),dtype="uint8")
    map4=np.zeros((250,400,3),dtype="uint8")
    
    for y in range(250):
        for x in range(400):
            # Bottom left line
            if (y-((85/69)*x) <= (28)):
                map1[y][x]=clearance
            if (y-((85/69)*x) < (475/23)):
                map1[y][x]=obstacle_space
            
            # Bottom Right line
            if (y-((16/5)*x) >= (-201)):
                map2[y][x]=clearance
            if (y-((16/5)*x) > (-186)):
                map2[y][x]=obstacle_space

            # Upper right line
            if (y+((6/7)*x) <= (145)):
                    map3[y][x]=clearance
            if (y+((6/7)*x) < (970/7)):
                map3[y][x]=obstacle_space
            
            # Upper left line
            if (y+((25/79)*x) >= (71)):
                map4[y][x]=clearance
            if (y+((25/79)*x) > (6035/79)):
                map4[y][x]=obstacle_space
       
    cv2.bitwise_or(map3,map2,map_temp1,mask=None)
    # cv2.imshow("maptemp1",map_temp1)
    # cv2.waitKey(0)
    cv2.bitwise_and(map4,map1,map_temp2,mask=None)
    # cv2.imshow("maptemp2",map_temp2)
    # cv2.waitKey(0)
    cv2.bitwise_and(map_temp2,map_temp1,map,mask=None)
    # cv2.imshow('Shape',map)
    # cv2.waitKey(0)
    return map
   

# Function to plot hexagon on the map 
def hexagon():
    obstacle_space=np.array([0,255,255])
    clearance=np.array([0,0,255])
    map=np.zeros((250,400,3),dtype="uint8")

    map1=np.zeros((250,400,3),dtype="uint8")
    map2=np.zeros((250,400,3),dtype="uint8")
    map3=np.zeros((250,400,3),dtype="uint8")
    map4=np.zeros((250,400,3),dtype="uint8")
    map5=np.zeros((250,400,3),dtype="uint8")
    map6=np.zeros((250,400,3),dtype="uint8")

    for y in range(250):
        for x in range(400):
            
            if x>=160:
                map1[y][x]=clearance
            if x>=165:
                map1[y][x]=obstacle_space
            
            if (y-(23/40)*x)<=(81):
                map2[y][x]=clearance
            if (y-((4/7)*x)<=(530/7)):
                map2[y][x]=obstacle_space
            
            if (y+(23/40)*x)<=(311):
                map3[y][x]=clearance
            if (y+((4/7)*x)<=(2130/7)):
                map3[y][x]=obstacle_space
            
            if x<=240:
                map4[y][x]=clearance
            if x<=235:
                map4[y][x]=obstacle_space
            
            if (y-(23/40)*x)>=(-11):
                map5[y][x]=clearance
            if (y-((4/7)*x)>=(-30/7)):
                map5[y][x]=obstacle_space
            
            if (y+(23/40)*x)>219:
                map6[y][x]=clearance
            if (y+((4/7)*x)>=(1570/7)):
                map6[y][x]=obstacle_space

    map_temp1=np.zeros((250,400,3),dtype="uint8")
    map_temp2=np.zeros((250,400,3),dtype="uint8")
    map_temp3=np.zeros((250,400,3),dtype="uint8")
    premap=np.zeros((250,400,3),dtype="uint8")
    cv2.bitwise_and(map1,map2,map_temp1,mask=None)
    cv2.bitwise_and(map3,map4,map_temp2,mask=None)
    cv2.bitwise_and(map5,map6,map_temp3,mask=None)
    cv2.bitwise_and(map_temp1,map_temp2,premap,mask=None)
    cv2.bitwise_and(premap,map_temp3,map,mask=None)
    return map

# Function to generate the map with obstacles
def map_generation():
    clearance=np.array([0,0,255])
    obstacle_space=np.array([0,255,255])
    obstacle_nodes=[]
    free_nodes=[]
    circle_map=plot_circle()
    arrow_map=arrow_shape()
    hex_map=hexagon()
    map=np.zeros((250,400,3),dtype="uint8")
    cv2.bitwise_or(circle_map,arrow_map,map,mask=None)
    cv2.bitwise_or(hex_map,map,map,mask=None)
    row,col,depth = map.shape
    # print("Row",row,"col",col)
    for i in range(row):
        for j in range(col):
            if((map[i,j]).any()==obstacle_space.any()) or ((map[i,j]).any()==clearance.any()):
                obstacle_nodes.append((i,j))
                # map[i,j]=[255,255,255]
            else:
                free_nodes.append((i,j))
                # map[i,j]=[0,255,0]
    # cv2.imshow("Map",map)
    # cv2.waitKey(0)
    return map,obstacle_nodes,free_nodes

map,obstacle_nodes,free_nodes=map_generation()

def get_nodes():
    map, obstacle_nodes,_=map_generation()
    while(1):
        start = input('Enter start node in the form (y x): ')
        start_node = tuple(int(x) for x in start.split())
        # Flip coordinates for opencv representation
        start_node=(y_flip-start_node[0],start_node[1])
        # print("start node",start_node) 
        goal=input('Enter goal node in the form (y x): ')
        goal_node=tuple(int(x) for x in goal.split())
        # Flip coordinates for opencv representation
        goal_node=(y_flip-goal_node[0],goal_node[1])
        # print("goal node",goal_node)

        if (start_node in obstacle_nodes) or (goal_node in obstacle_nodes):
            print("Nodes in obstacle space, try again")
            continue
                        
        elif (start_node[0]<0 or start_node[0]>=250 or start_node[1]<0 or start_node[1]>=400):
            print("Start out of bounds,try again")
            continue

        elif (goal_node[0]<0 or goal_node[0]>=250 or goal_node[1]<0 or goal_node[1]>=400):
            print("Goal out of bounds, try again")
            continue

        elif (start_node==None or goal_node==None):
            print('Empty error')
            continue

        else:
            break
    return start_node,goal_node

start_node,goal_node=get_nodes()

# Class Node_det to create objects to hold details of node- coordinates, cost to come, parent node coordinates
class Node_det:
    def __init__(self, node_pos, cost, parent): 
        self.node_pos = node_pos
        self.cost = cost
        self.parent = parent

# Function to find valid nodes around the current node
def create_child_nodes(current_node): 
    i = current_node.node_pos[0]
    j = current_node.node_pos[1]

    # list containing coordinates of all 8 action set nodes around current
    adjacent_nodes = [(i, j + 1), (i + 1, j), (i - 1, j), (i, j - 1), (i + 1, j + 1), (i - 1, j - 1), (i - 1, j + 1),
    (i + 1, j - 1)]  
    valid_nodes = []
    for node_pos, path in enumerate(adjacent_nodes):
        # Check if out of bounds
        if not (path[0] >= 250 or path[0] < 0 or path[1] >= 400 or path[1] < 0):  

            # Check if obstacle space (by checking color)
            if (map[path].any())==(np.array([0,0,0]).any()):  
                if node_pos>3:
                    cost = math.sqrt(2) 
                else:
                    cost = 1
                valid_nodes.append([path, cost])
    return valid_nodes 

# Function to backtrack from goal node to start node
def backtrack(goal_node,ClosedList,map):
    # Green circle to denote start node
    cv2.circle(map,(start_node[1],start_node[0]),2,(0,255,0),-1)
    # Red circle to denote goal node
    cv2.circle(map,(goal_node[1],goal_node[0]),2,(0,0,255),-1)
    
    goal_node = ClosedList[str(goal_node)]
    parent_node = goal_node.parent  
    print("Backtracking nodes")
    # Run loop until start node is found (i.e parent node of start node is None)
    while parent_node:
        print("Node: ",parent_node.node_pos," Cost= ", parent_node.cost)
        # Change color to show backtracking path
        map[parent_node.node_pos[0], parent_node.node_pos[1],:] = np.array([255,0,0]) #using cv2.imshow to print the final path using backtracking
        out.write(map)
        cv2.waitKey(1)
        out2.write(map)
        # To show the animation of the backtracking path
        cv2.namedWindow('Backtracking',cv2.WINDOW_NORMAL)
        cv2.imshow('Backtracking', map)
        # Delay to illustrate backtracking
        cv2.waitKey(200)
        parent_node = parent_node.parent
    cv2.waitKey(0)    
        
#OpenList defined as Priority queue 
OpenList = PriorityQueue() 
# Set of visited nodes
visited = set([]) 
#Closed List Dictionary for used for backtracking 
ClosedList = {}

# VideoWriter function to show the animation of the goal traversal
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('Dijkstra.avi',fourcc,300,(400,250))
out2=cv2.VideoWriter('Backtracking.avi',fourcc,10,(400,250))


# Dictionary to contain the cost for all nodes
cost_list = {}

# Set default cost of all nodes to infinity
for i in range(0, 250):
    for j in range(0, 400):
        cost_list[str([i, j])] = inf 

# set cost of start node to zero
cost_list[str(start_node)] = 0 
visited.add(str(start_node))
current_node = Node_det(start_node, 0, None)
ClosedList[str(current_node.node_pos)] = current_node
OpenList.put([current_node.cost, current_node.node_pos])  


while not OpenList.empty():
    # Pop high priority element
    x = OpenList.get()
    current_node = ClosedList[str(x[1])]
    # Check if current node is goal node
    if x[1][0] == goal_node[0] and x[1][1] == goal_node[1]:
        print("Reached")
        ClosedList[str(goal_node)] = Node_det(goal_node, x[0],current_node)
        backtrack(goal_node,ClosedList,map)
        break
    
    # generate valid adjacent nodes to current node and iterate
    for immediate_node, cost in create_child_nodes(current_node):
        
        # update cost if necessary for previously visited nodes
        if str(immediate_node) in visited: 
            temp_cost = cost + cost_list[str(current_node.node_pos)]
            if temp_cost < cost_list[str(immediate_node)]:
                cost_list[str(immediate_node)] = temp_cost
                ClosedList[str(immediate_node)].parent = current_node
        else:
            # if not visited earlier, it is now added to list of visited nodes
            visited.add(str(immediate_node)) 
            # Calculating cost for the newly unvisited nodes and updating cost
            local_cost = cost + cost_list[str(current_node.node_pos)]
            cost_list[str(immediate_node)] = local_cost
            new_node = Node_det(immediate_node, local_cost, ClosedList[str(current_node.node_pos)])
            # Adding new node to ClosedList
            ClosedList[str(immediate_node)] = new_node
            #Added to queue for further least cost calculations
            OpenList.put([local_cost, new_node.node_pos])   
        # changing color of visited nodes to reflect traversal of Dijkstra Algorithm
        map[immediate_node[0]][immediate_node[1]]=[255,255,255]
        cv2.namedWindow("Dijkstra Implementation",cv2.WINDOW_NORMAL)
        cv2.imshow("Dijkstra Implementation",map)
        out.write(map)
        cv2.waitKey(1)

