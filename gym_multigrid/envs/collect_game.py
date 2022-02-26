from gym_multigrid.multigrid import *

class CollectGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        agents_index = [],
        balls_index=[],
        balls_reward=[],
        zero_sum = False,
        view_size=10

    ):
        self.num_balls = num_balls
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum
        self.running_reward = 0

        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )



    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
            for i in range(number):
                self.place_obj(Safe_Berry(self.world, index, 100))
        
        
        for i in range(50):
            self.place_obj(Poison_Berry(self.world, 0, -100))


        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)


    def _reward(self, i, rewards, reward):
        """
        Compute the reward to be given upon success
        """
        for j,a in enumerate(self.agents):
            #print(j, a.index)
            if a.index==i:
                #print("matched agent")
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward
        #print("rewarded",rewards)

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    print("agent ", i, " PICKED A:", fwd_cell.type, fwd_cell.reward)
                    if fwd_cell.type == 'safe_berry':
                        #self.grid.set(*fwd_pos, None)
                        #done = True
                        # reward += self._reward()
                        self._reward(i, rewards, fwd_cell.reward)
                        print(fwd_cell.reward)
                    elif fwd_cell.type == 'poison_berry':
                        #self.grid.set(*fwd_pos, None)
                        #done = True
                        # reward += self._reward()
                        self.agents[i].is_marked = True
                        self.agents[i].marked_step = self.step_count
                    #self._reward(i, rewards, fwd_cell.reward)
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def step(self, actions):
        
        obs, rewards, done, info, step_count = MultiGridEnv.step(self, actions)
        # print(rewards)
        self.running_reward += rewards.sum()
        print('step=%s, reward=%.2f' % (step_count, self.running_reward))
        return obs, rewards, done, info


class CollectGame4HEnv10x10N2(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[2],
        agents_index = [0],
        balls_index=[1],
        balls_reward=[1],
        zero_sum=False)

