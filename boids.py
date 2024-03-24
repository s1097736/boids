import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from scipy.spatial import KDTree

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


class Scene:

    def __init__(
        self, w, h, N, cohesion, separation, alignment, speed, interaction_radius, 
        exclusion_theta, gui=True, fps=30, draw_size=5,
    ):
        self.w = w  # width  of the window
        self.h = h  # height of the window
        self.N = N  # number of particles

        # set the simulation parameters
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.speed = speed
        self.interaction_radius = interaction_radius
        self.exclusion_cutoff = np.deg2rad(180 - exclusion_theta/2)

        # initialize swarm
        self.swarm = np.array([Particle(self.w, self.h) for _ in range(self.N)])
        self.particle_positions = np.array([particle.pos.xy for particle in self.swarm])

        # initialize gui related info
        self.gui = gui
        if gui:
            pygame.init()

            self.screen = pygame.display.set_mode([w, h])
            self.draw_size = draw_size

            self.fps = fps
            self.fps_limiter = pygame.time.Clock()

    def angle_between(self, v1, v2):
        return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def neighbours_of(self, p1):

        # find neighbours within the radius using a KDTree
        tree = KDTree(self.particle_positions)
        idx = tree.query_ball_point(np.array(p1.pos.xy),r=self.interaction_radius)
        # find the closest neighbor overall (not restricted to the
        # neighborhood; sometimes the neighborhood only contains
        # the boid itself)

        # use k=2 because the closest neighbor will always be 
        # the boid itself
        nearest_neighbour_distance = tree.query(np.array(p1.pos.xy), k=2)[0][1] 

        # exclude the neighbors within the blind spot
        seen_neighbours = []
        for i in idx:
            # if boid is evaluated against itself, keep automatically
            if p1.pos.xy == self.particle_positions[i]:
                seen_neighbours.append(i)
            else: # calculate the angle and keep boid if it can be "seen"
                angle = self.angle_between(p1.dir.xy, 
                    self.unit_vector(np.array(-p1.pos.xy + self.particle_positions[i])))
                if angle <= self.exclusion_cutoff:
                    seen_neighbours.append(i)

        seen_neighbours = np.array(seen_neighbours)
        neighbours = self.swarm[seen_neighbours]

        return neighbours, nearest_neighbour_distance

    def step(self):

        nearest_neighbour_distances = []

        for i, particle in enumerate(self.swarm):
            neighbours, nearest_neighbour_distance = self.neighbours_of(particle)
            new_position = particle.step(
                neighbours, self.cohesion, self.separation, self.alignment, self.speed)

            self.particle_positions[i] = new_position

            nearest_neighbour_distances.append(nearest_neighbour_distance)

        return nearest_neighbour_distances

    def draw(self):
        self.screen.fill((255, 255, 255))

        for particle in self.swarm:
            pygame.draw.circle(self.screen, (0, 0, 0), particle.pos, self.draw_size)

        pygame.display.flip()
        self.fps_limiter.tick(self.fps)

    def check_interrupt(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return True
        return False

    def order(self):
        return sum(
            [particle.dir for particle in self.swarm], pygame.Vector2(0, 0)
        ).magnitude() / self.N

    def run(self, num_steps=300, process_id=None, results=None):
        interrupt = False
        step = 0

        nearest_neighbour_distances = []
        orders = []

        while step < num_steps and not interrupt:
            # the user interrupted the run
            if self.gui and self.check_interrupt(): interrupt = True

            # update the scene
            nearest_neighbour_distances.append(self.step())
            orders.append(self.order())

            # draw the scene
            if self.gui: self.draw()

            step += 1

        if self.gui: pygame.quit()

        if process_id is not None:
            results['orders'][process_id] = orders
            results['neighbours'][process_id] = nearest_neighbour_distances
            return

        return orders, nearest_neighbour_distances


class Particle:

    def __init__(self, w, h):
        self.pos = pygame.Vector2(np.random.randint(w), np.random.randint(h))
        self.dir = pygame.Vector2(*(np.random.rand(2) * 2 - 1)).normalize()

        self.w = w
        self.h = h

    def wrap(self):
        # wrap the particle around to the other side of the scene
        if self.pos.x < 0 or self.pos.x > self.w: self.pos.x %= self.w
        if self.pos.y < 0 or self.pos.y > self.h: self.pos.y %= self.h

    def step(self, neighbours, cohesion, separation, alignment, speed):
        avg_pos  = pygame.Vector2(0, 0)  # average position within interaction radius
        avg_away = pygame.Vector2(0, 0)  # average direction to move away from others

        # the average direction within the interaction radius
        avg_sin = 0
        avg_cos = 0

        for neighbour in neighbours:
            avg_sin += np.sin(np.deg2rad(neighbour.dir.as_polar()[1]))
            avg_cos += np.cos(np.deg2rad(neighbour.dir.as_polar()[1]))

            avg_pos += neighbour.pos

            if neighbour != self:
                away = self.pos - neighbour.pos
                try: away /= away.magnitude_squared()  # normalize
                except ZeroDivisionError: away = pygame.Vector2(0, 0)
                avg_away += away

        # take the mean
        avg_pos  /= len(neighbours)
        avg_away /= len(neighbours)

        # alignment: move towards the average heading of the neighbours
        avg_angle = np.arctan2(avg_sin, avg_cos)  # note first y then x
        avg_angle += np.random.rand() * 0.5 - 0.25 # add some noise
        avg_dir = pygame.Vector2.from_polar((1, np.rad2deg(avg_angle)))
        self.dir = avg_dir * alignment

        # cohesion: move towards the average position of the neighbours
        self.dir += (avg_pos - self.pos) / cohesion

        # separation: move away from the neighbours to avoid crowding
        self.dir += avg_away * separation

        # normalize the current direction vector
        self.dir = self.dir.normalize()

        # update the position with a constant speed
        self.pos += self.dir * speed

        self.wrap()

        return np.array(self.pos)


def plot_order(orders, title):
    mean = np.mean(orders, axis=0)
    std = np.std(orders, axis=0)

    # plot the individual runs
    for order in orders:
        plt.plot(order, color='tab:grey', alpha=0.30, lw=0.8)

    # plot the mean and std
    plt.plot(mean, label='Mean', color='tab:red')
    plt.fill_between(
        np.arange(0, len(mean)), mean - std, mean + std,
        alpha=0.3, label='Standard deviation', color='tab:red'
    )

    plt.xlabel('Time point')
    plt.ylabel('Order parameter')
    plt.xlim([0,300])

    plt.legend()

    plt.savefig(f'results/{title}.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_nearest_neighbours(distances, title):

    mean = np.mean(distances, axis=(2,0))
    std = np.std(np.mean(distances, axis=2), axis=0)

    plt.plot(mean, label='Mean', color='tab:red')
    plt.fill_between(
        np.arange(0, len(mean)), mean - std, mean + std,
        alpha=0.3, label='Standard deviation', color='tab:red'
    )
    plt.xlim([0,300])

    plt.xlabel('Time point')
    plt.ylabel('Nearest neighbour distance')

    plt.legend()

    plt.savefig(f'results/{title}.png', bbox_inches='tight', dpi=300)
    plt.show()


def run_repeated(
    w, h, N, cohesion, separation, alignment,speed, interaction_radius,
    exclusion_theta, num_steps=300, repetitions=mp.cpu_count(),
):
    manager = mp.Manager()
    results = manager.dict()

    results['neighbours'] = manager.dict()
    results['orders'] = manager.dict()

    jobs = []

    scenes = [
        Scene(
            w=w, h=h, N=N, cohesion=cohesion, separation=separation, alignment=alignment,
            speed=speed, interaction_radius=interaction_radius, 
            exclusion_theta=exclusion_theta, gui=False
        ) for _ in range(repetitions)
    ]

    for process_id in range(repetitions):
        jobs.append(mp.Process(
            target=scenes[process_id].run,
            args=(num_steps, process_id, results)
        ))
        jobs[-1].start()

    for process in jobs:
        process.join()

    return (
        np.array(results['orders'].values()),
        np.array(results['neighbours'].values())
    )


def mutate(variables, samples):
    mutated_variables = {}
    for variable_name, variable_props in variables.items():
        mutated = np.random.normal(samples[variable_name], variable_props['sigma'])
        # in case the value is not valid, sample again
        while not variable_props['valid'](mutated):
            mutated = np.random.normal(samples[variable_name], variable_props['sigma'])
        mutated_variables[variable_name] = mutated
    return mutated_variables


def sample_priors(variables):
    return {
        variable_name: variable_function['prior'](1)[0]
        for variable_name, variable_function in variables.items()
    }


def accept(orders, epsilon, target, n=50):
    # check if |(target - mean of the last n values over all runs)| < epsilon
    return np.abs(target-np.mean(orders[:, -n:], axis=(1,0))) < epsilon


def create_population(
    epsilon, w, h, N, abc_N, variables, speed, interaction_radius, exclusion_theta, num_steps, repetitions, target
):
    accepted = []
    accepted_orders = []
    while len(accepted) < abc_N:
        variable_samples = sample_priors(variables)
        orders, _ = run_repeated(
            w, h, N, speed=speed, interaction_radius=interaction_radius, exclusion_theta=exclusion_theta, 
            repetitions=repetitions, **variable_samples,

        )

        if accept(orders, epsilon, target):
            accepted.append(variable_samples)
            accepted_orders.append(orders)

    return accepted, accepted_orders


def mutate_population(
    population, epsilon, w, h, N, abc_N, variables, speed, interaction_radius, 
    exclusion_theta, num_steps, repetitions, target
):
    accepted = []
    accepted_orders = []
    while len(accepted) < abc_N:
        variable_samples = mutate(variables, np.random.choice(population))
        orders, _ = run_repeated(
            w, h, N, speed=speed, interaction_radius=interaction_radius, 
            exclusion_theta=exclusion_theta,repetitions=repetitions, **variable_samples
        )

        if accept(orders, epsilon, target):
            accepted.append(variable_samples)
            accepted_orders.append(orders)

    return accepted, accepted_orders


def abc(
    w, h, N, abc_N, epsilons, variables, speed, interaction_radius,
    exclusion_theta, target, num_steps=300, repetitions=mp.cpu_count(),
):
    # initialize population
    print(f'abc: creating the initial population with e={epsilons[0]}')
    populations = []
    accepted_orders = []

    accepted, orders = create_population(
        epsilons[0], w, h, N, abc_N, variables,speed,
        interaction_radius, exclusion_theta, num_steps, repetitions,target)

    populations.append(accepted)
    accepted_orders.append(orders)

    # iterate over generations
    for index, epsilon in enumerate(epsilons[1:]):
        print(f'abc: mutating population with e={epsilon:.3f} ({index + 1}/{len(epsilons) - 1})')
        accepted, orders = mutate_population(
            populations[-1], epsilon, w, h, N, abc_N, variables, speed,
            interaction_radius, exclusion_theta, num_steps, repetitions,target)

        populations.append(accepted)
        accepted_orders.append(orders)

    return populations, accepted_orders

def plot_dists(populations, orders, title):

    # sort data
    gens = [0,4,9]
    cohesions = []
    separations = []
    alignments = []
    final_orders = []
    for gen in gens:
        # parameters
        cohesions.append([d['cohesion'] for d in populations[gen]])
        separations.append([d['separation'] for d in populations[gen]])
        alignments.append([d['alignment'] for d in populations[gen]])
        # order
        final_orders.append(np.mean(orders[...,-50:], axis=3)[gen])

    final_orders = np.reshape(np.array(final_orders), (3,-1))

    # plot marginal distributions for all parameters

    # cohesion
    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(8, 2.5), layout='constrained')
    ax[0].hist(cohesions[0], density=True)
    ax[1].hist(cohesions[1], density=True) 
    ax[2].hist(cohesions[2], density=True) 
    for ax in fig.get_axes():
        ax.label_outer()  
    fig.supxlabel('$C$')
    fig.supylabel('Density')
    plt.savefig(f'results/cohesion_{title}', bbox_inches='tight', dpi=300)
    plt.show()

    # separation
    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(8, 2.5), layout='constrained')
    ax[0].hist(separations[0], density=True, color='tab:blue')
    ax[1].hist(separations[1], density=True, color='tab:blue') 
    ax[2].hist(separations[2], density=True, color='tab:blue') 
    for ax in fig.get_axes():
        ax.label_outer()  
    fig.supxlabel('$S$')
    fig.supylabel('Density')
    plt.savefig(f'results/separation_{title}', bbox_inches='tight', dpi=300)    
    plt.show()

    # alignment
    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(8, 2.5), layout='constrained')
    ax[0].hist(alignments[0], density=True, color='tab:blue')
    ax[1].hist(alignments[1], density=True, color='tab:blue') 
    ax[2].hist(alignments[2], density=True, color='tab:blue') 
    for ax in fig.get_axes():
        ax.label_outer()  
    fig.supxlabel('$A$')
    fig.supylabel('Density')
    plt.savefig(f'results/alignment_{title}', bbox_inches='tight', dpi=300)    
    plt.show()

    # plot order distributions across generations 
    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(8, 2.5), layout='constrained')
    ax[0].hist(final_orders[0], density=True, color='tab:red')
    ax[1].hist(final_orders[1], density=True, color='tab:red') 
    ax[2].hist(final_orders[2], density=True, color='tab:red') 
    for ax in fig.get_axes():
        ax.label_outer()  
    fig.supxlabel('Order parameter')
    fig.supylabel('Density')
    plt.savefig(f'results/order_{title}', bbox_inches='tight', dpi=300)    
    plt.show()


    # run some simulations with 3 sets of parameters sampled from the
    # last generation, and plot
    for i in range(3):
        parameters = populations[-1][i]
        orders, nearest_neighbour_distances = run_repeated(
            w=600, h=600, N=15, speed=5, interaction_radius=200, 
            exclusion_theta=60, **parameters
        )

        plot_order(orders, f'{title}_order_{i}')

        plot_nearest_neighbours(nearest_neighbour_distances, f'{title}_neighbors_{i}')        

def main():
    np.random.seed(0)  # for reproducibility

    # simulation parameters
    w=600
    h=600
    N=15
    speed=5

    # boid parameters (where relevant)
    cohesion=100
    separation=30
    alignment=1
    interaction_radius=200
    exclusion_theta=60


    # # run multiple experiments
    # orders, nearest_neighbour_distances = run_repeated(
    #     w, h, N, cohesion, separation, 
    #     alignment, speed, interaction_radius, 
    #     exclusion_theta
    # )

    # plot_order(orders, 'initial_order')
    # plot_nearest_neighbours(nearest_neighbour_distances, 'initial_neighbors')

    # run a single trial with gui
    scene = Scene(
        w, h, N, cohesion, separation, 
        alignment, speed, interaction_radius, 
        exclusion_theta, gui=True, fps=30
    )
    scene.run(num_steps=300)


    # # run abc, with target of 1.0
    # populations, accepted_orders = abc(
    #     w, h, abc_N=20, N=15, speed=speed, interaction_radius=interaction_radius,
    #     exclusion_theta=exclusion_theta, epsilons=np.linspace(0.5, 0.05, 10), target=1.0,

    #     variables={
    #         'cohesion': {
    #             'prior': lambda n: np.random.uniform(50, 150, n),
    #             'valid': lambda x: x>0,
    #             'sigma': 10,
    #         },
    #         'separation': {
    #             'prior': lambda n: np.random.uniform(10, 50, n),
    #             'valid': lambda x: x>0,                
    #             'sigma': 10,
    #         },
    #         'alignment': {
    #             'prior': lambda n: np.random.uniform(0.1, 1.5, n),
    #             'valid': lambda x: x>0,
    #             'sigma': 0.4,
    #         },
    #     }
    # )

    # np.save('results/populations_1.npy', np.array(populations))
    # np.save('results/accepted_orders_1.npy', np.array(accepted_orders))

    # # run abc, with target of 0.6
    # populations, accepted_orders = abc(
    #     w, h, abc_N=20, N=15, speed=speed, interaction_radius=interaction_radius,
    #     exclusion_theta=exclusion_theta, epsilons=np.linspace(0.5, 0.05, 10), target=0.6,

    #     variables={
    #         'cohesion': {
    #             'prior': lambda n: np.random.uniform(50, 150, n),
    #             'valid': lambda x: x>0,
    #             'sigma': 10,
    #         },
    #         'separation': {
    #             'prior': lambda n: np.random.uniform(10, 50, n),
    #             'valid': lambda x: x>0,                
    #             'sigma': 10,
    #         },
    #         'alignment': {
    #             'prior': lambda n: np.random.uniform(0.1, 1.5, n),
    #             'valid': lambda x: x>0,
    #             'sigma': 0.4,
    #         },
    #     }
    # )

    # np.save('results/populations_06.npy', np.array(populations))
    # np.save('results/accepted_orders_06.npy', np.array(accepted_orders))    
    # for population in populations[-1:]:
    #     print(*population, sep='\n')
    #     print()

    #     for accepted in population[:3]:
    #         # run a single trial with gui
    #         scene = Scene(
    #             w=600, h=600, N=15, **accepted, speed=5, interaction_radius=100, exclusion_theta=60, gui=True, fps=30
    #         )

    #         orders, nearest_neighbour_distances = scene.run(num_steps=300)
    #         plot_order(np.array([orders]), '1_order')
    #         plot_nearest_neighbours(np.array([nearest_neighbour_distances]), '1_neighbors')


if __name__ == "__main__":

    main()

    # plot all results for ABC-SMC on target of 1.0
    # populations_1 = np.load('results/populations_1.npy', allow_pickle=True)
    # accepted_orders_1 = np.load('results/accepted_orders_1.npy')
    # plot_dists(populations_1, accepted_orders_1, '1')

    # plot all results for ABC-SMC on target of 0.6
    # populations_06 = np.load('results/populations_06.npy', allow_pickle=True)
    # accepted_orders_06 = np.load('results/accepted_orders_06.npy')
    # plot_dists(populations_06, accepted_orders_06, '06')
