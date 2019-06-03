import numpy as np 

class Cell(object):
	def __init__(self):
		self.position=(0,0)
		self.parent=None

		self.g = 0
		self.h = 0
		self.f = 0

	def __eq__(self, cell):
		return self.position == cell.position

	def showcell(self):
		print(self.position)


class Gridworld(object):
	
	def __init__(self, world_size=(5, 5)):
		self.w = np.zeros(world_size)
		self.world_x_limit = world_size[0]
		self.world_y_limit = world_size[1]
	
	def show(self):
		print(self.w)

	def get_neigbours(self, cell):
		neughbour_cord = [(-1, -1), (-1, 0), (-1, 1),
							(0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
		current_x = cell.position[0]
		current_y = cell.position[1]
		neighbours = []
		for n in neughbour_cord:
			x = current_x + n[0]
			y = current_y + n[1]
			if x >=0 and x < self.world_x_limit and y >=0 and y < self.world_y_limit:
				c = Cell()
				c.position = (x,y)
				c.parent = cell
				neighbours.append(c)
		return neighbours
		# print(neighbours)
		# print(self.w[cell(0)]:2)

def astar(world, start, goal):
	_open = []
	_closed = []
	_open.append(start)

	while _open:
		min_f = np.argmin([n.f for n in _open])
		current = _open[min_f]
		_closed.append(_open.pop(min_f))
		

		if current == goal:
			break

		for n in world.get_neigbours(current):
			for c in _closed:
				if c == n:
					continue
			n.g = current.g + 1
			x1, y1 = n.position
			x2, y2 = goal.position
			n.h = (y2 - y1)**2 + (x2 - x1)**2
			n.f = n.h + n.g

			for c in _open:
				if c == n and c.f < n.f:
					continue
			_open.append(n)
	path = []
	while current.parent is not None:
		path.append(current.position)
		current = current.parent
	path.append(current.position)
	path = path[::-1]
	print(path)
	return path




p = Gridworld()
start = Cell()
start.position = (0,0)
goal = Cell()
goal.position = (4,4)
print("path from {} to {} ".format(start.position, goal.position))
s=astar(p, start, goal)
for i in s:
	p.w[i] = 1
print(p.w)
# nei = p.get_neigbours(k)
# [print(i.position) for i in nei]
# p.show()