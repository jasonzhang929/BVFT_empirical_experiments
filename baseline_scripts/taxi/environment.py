import numpy as np

class taxi(object):
	n_state = 0
	n_action = 6
	def __init__(self, length):
		self.length = length
		self.x = np.random.randint(length)
		self.y = np.random.randint(length)
		self.possible_passenger_loc = [(0,0), (0,length-1), (length-1,0), (length-1, length-1)]
		self.passenger_status = np.random.randint(16)
		self.taxi_status = 4
		self.n_state = (length**2)*16*5

	def reset(self):
		length = self.length
		self.x = np.random.randint(length)
		self.y = np.random.randint(length)
		self.possible_passenger_loc = [(0,0), (0,length-1), (length-1,0), (length-1, length-1)]
		self.passenger_status = np.random.randint(16)
		self.taxi_status = 4
		return self.state_encoding()

	def state_encoding(self):
		length = self.length
		return self.taxi_status + (self.passenger_status + (self.x * length + self.y) * 16) * 5

	def state_decoding(self, state):
		length = self.length
		taxi_status = state % 5
		state = state  // 5
		passenger_status = state % 16
		state = state // 16
		y = state % length
		x = state // length
		return x,y,passenger_status,taxi_status

	def set_state(self, state):
		x, y, passenger_status, taxi_status = self.state_decoding(state)
		self.x, self.y = x, y
		self.passenger_status = passenger_status
		self.taxi_status = taxi_status

	def get_T_R(self, max_iteration_per_entry, convergence=1e-3, checkpoint=1000):
		T = np.zeros((self.n_state, self.n_action, self.n_state))
		R = np.zeros((self.n_state, self.n_action, self.n_state))
		max_iter = 0
		for state in range(self.n_state):
			if state % 10 == 0:
				print(F"state {state}")
			for action in range(self.n_action):
				old_t_states = np.zeros(self.n_state)
				t_states = np.zeros(self.n_state)
				r_states = np.zeros(self.n_state)
				for i in range(int(max_iteration_per_entry)):
					self.set_state(state)
					next_state, reward = self.step(action)
					t_states[next_state] += 1
					r_states[next_state] += reward

					if i % checkpoint == 0:
						new_t_states = t_states / np.sum(t_states)
						if np.max(np.abs(new_t_states - old_t_states)) > convergence:
							old_t_states = new_t_states
							if i > max_iter:
								print(i)
								max_iter = i
						else:
							break
				R[state, action, :] = r_states / np.maximum(t_states, 1.0)
				T[state, action, :] = t_states / np.sum(t_states)
		print(F"max_iter_{max_iter}")
		return T, R


	def render(self):
		MAP = []
		length = self.length
		for i in range(length):
			if i == 0:
				MAP.append('-'*(3*length+1))
			MAP.append('|' + '  |' * length)
			MAP.append('-'*(3*length+1))
		MAP = np.asarray(MAP, dtype = 'c')
		if self.taxi_status == 4:
			MAP[2*self.x+1, 3*self.y+2] = 'O'
		else:
			MAP[2*self.x+1, 3*self.y+2] = '@'
		for i in range(4):
			if self.passenger_status & (1<<i):
				x,y = self.possible_passenger_loc[i]
				MAP[2*x+1, 3*y+1] = 'a'
		for line in MAP:
			print (''.join(line))
		if self.taxi_status == 4:
			print ('Empty Taxi')
		else:
			x,y = self.possible_passenger_loc[self.taxi_status]
			print ('Taxi destination:({},{})'.format(x,y))

	def step(self, action):
		reward = -1

		if action == 0:
			if self.x < self.length - 1:
				self.x += 1
		elif action == 1:
			if self.y < self.length - 1:
				self.y += 1
		elif action == 2:
			if self.x > 0:
				self.x -= 1
		elif action == 3:
			if self.y > 0:
				self.y -= 1
		elif action == 4:	# Try to pick up
			for i in range(4):
				x,y = self.possible_passenger_loc[i]
				if x == self.x and y == self.y and(self.passenger_status & (1<<i)):
					# successfully pick up
					self.passenger_status -= 1<<i
					self.taxi_status = np.random.randint(4)
					while self.taxi_status == i:
						self.taxi_status = np.random.randint(4)
		elif action == 5:
			if self.taxi_status < 4:
				x,y = self.possible_passenger_loc[self.taxi_status]
				if self.x == x and self.y == y:
					reward = 20
				self.taxi_status = 4
		self.change_passenger_status()
		return self.state_encoding(), reward

	def change_passenger_status(self):
		p_generate = [0.3, 0.05, 0.1, 0.2]
		p_disappear = [0.05, 0.1, 0.1, 0.05]
		for i in range(4):
			if self.passenger_status & (1<<i):
				if np.random.rand() < p_disappear[i]:
					self.passenger_status -= 1<<i
			else:
				if np.random.rand() < p_generate[i]:
					self.passenger_status += 1<<i
	def debug(self):
		self.reset()
		while True:
			self.render()
			action = input('Action:')
			if action > 5 or action < 0:
				break
			else:
				_, reward = self.step(action)
				print (reward)


class taxi(object):
	n_state = 0
	n_action = 6
	def __init__(self, length):
		self.length = length
		self.x = np.random.randint(length)
		self.y = np.random.randint(length)
		self.possible_passenger_loc = [(0,0), (0,length-1), (length-1,0), (length-1, length-1)]
		self.passenger_status = np.random.randint(16)
		self.taxi_status = 4
		self.n_state = (length**2)*16*5

	def reset(self):
		length = self.length
		self.x = np.random.randint(length)
		self.y = np.random.randint(length)
		self.possible_passenger_loc = [(0,0), (0,length-1), (length-1,0), (length-1, length-1)]
		self.passenger_status = np.random.randint(16)
		self.taxi_status = 4
		return self.state_encoding()

	def state_encoding(self):
		length = self.length
		return self.taxi_status + (self.passenger_status + (self.x * length + self.y) * 16) * 5

	def state_decoding(self, state):
		length = self.length
		taxi_status = state % 5
		state = state  // 5
		passenger_status = state % 16
		state = state // 16
		y = state % length
		x = state // length
		return x,y,passenger_status,taxi_status

	def set_state(self, state):
		x, y, passenger_status, taxi_status = self.state_decoding(state)
		self.x, self.y = x, y
		self.passenger_status = passenger_status
		self.taxi_status = taxi_status

	def get_T_R(self, max_iteration_per_entry, convergence=1e-3, checkpoint=1000):
		T = np.zeros((self.n_state, self.n_action, self.n_state))
		R = np.zeros((self.n_state, self.n_action, self.n_state))
		max_iter = 0
		for state in range(self.n_state):
			if state % 10 == 0:
				print(F"state {state}")
			for action in range(self.n_action):
				old_t_states = np.zeros(self.n_state)
				t_states = np.zeros(self.n_state)
				r_states = np.zeros(self.n_state)
				for i in range(int(max_iteration_per_entry)):
					self.set_state(state)
					next_state, reward = self.step(action)
					t_states[next_state] += 1
					r_states[next_state] += reward

					if i % checkpoint == 0:
						new_t_states = t_states / np.sum(t_states)
						if np.max(np.abs(new_t_states - old_t_states)) > convergence:
							old_t_states = new_t_states
							if i > max_iter:
								print(i)
								max_iter = i
						else:
							break
				R[state, action, :] = r_states / np.maximum(t_states, 1.0)
				T[state, action, :] = t_states / np.sum(t_states)
		print(F"max_iter_{max_iter}")
		return T, R


	def render(self):
		MAP = []
		length = self.length
		for i in range(length):
			if i == 0:
				MAP.append('-'*(3*length+1))
			MAP.append('|' + '  |' * length)
			MAP.append('-'*(3*length+1))
		MAP = np.asarray(MAP, dtype = 'c')
		if self.taxi_status == 4:
			MAP[2*self.x+1, 3*self.y+2] = 'O'
		else:
			MAP[2*self.x+1, 3*self.y+2] = '@'
		for i in range(4):
			if self.passenger_status & (1<<i):
				x,y = self.possible_passenger_loc[i]
				MAP[2*x+1, 3*y+1] = 'a'
		for line in MAP:
			print (''.join(line))
		if self.taxi_status == 4:
			print ('Empty Taxi')
		else:
			x,y = self.possible_passenger_loc[self.taxi_status]
			print ('Taxi destination:({},{})'.format(x,y))

	def step(self, action):
		reward = -1

		if action == 0:
			if self.x < self.length - 1:
				self.x += 1
		elif action == 1:
			if self.y < self.length - 1:
				self.y += 1
		elif action == 2:
			if self.x > 0:
				self.x -= 1
		elif action == 3:
			if self.y > 0:
				self.y -= 1
		elif action == 4:	# Try to pick up
			for i in range(4):
				x,y = self.possible_passenger_loc[i]
				if x == self.x and y == self.y and(self.passenger_status & (1<<i)):
					# successfully pick up
					self.passenger_status -= 1<<i
					self.taxi_status = np.random.randint(4)
					while self.taxi_status == i:
						self.taxi_status = np.random.randint(4)
		elif action == 5:
			if self.taxi_status < 4:
				x,y = self.possible_passenger_loc[self.taxi_status]
				if self.x == x and self.y == y:
					reward = 20
				self.taxi_status = 4
		self.change_passenger_status()
		return self.state_encoding(), reward

	def change_passenger_status(self):
		p_generate = [0.3, 0.05, 0.1, 0.2]
		p_disappear = [0.05, 0.1, 0.1, 0.05]
		for i in range(4):
			if self.passenger_status & (1<<i):
				if np.random.rand() < p_disappear[i]:
					self.passenger_status -= 1<<i
			else:
				if np.random.rand() < p_generate[i]:
					self.passenger_status += 1<<i
	def debug(self):
		self.reset()
		while True:
			self.render()
			action = input('Action:')
			if action > 5 or action < 0:
				break
			else:
				_, reward = self.step(action)
				print (reward)


