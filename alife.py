import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import seaborn as sns
import argparse
import matplotlib.colors as mcolors

from collections import defaultdict

NAMES = ["Grass", "Rabbit", "Animal"]
CODES = [0, 1, 2]
SIZE = 500  # The dimensions of the field
RABBIT_OFFSPRING = 2  # Max offspring offspring when a rabbit reproduces
FOX_OFFSPRING = 1  # Max offspring offspring when a fox reproduces
FOX_NUM = 100  # Initial number of foxes and rabbits in the field
RABBIT_NUM = 50  # Initial number of foxes and rabbits in the field
GRASS_RATE = 0.05  # Probability that grass grows back at any location in the next season.
WRAP = False  # Does the field wrap around on itself when rabbits move?
K = 100  # Max generations fox can live without eating
FOX_DIST = 2  # Default fox movements
RABBIT_DIST = 1  # Default rabbit movements


class Animal:
    """ A furry creature roaming a field in search of rabbit to eat.
    Mr. Fox must eat enough to reproduce, otherwise he will starve. """

    def __init__(self, move_dist):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
        self.generation = 0
        self.last_eaten = 0
        self.dead = False
        self.move_dist = move_dist

    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         fox's eaten level is reset to zero. """
        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the fox some rabbit(food), once it has eaten, restart the count of how many years its starved """
        self.eaten += amount

    def move(self):
        """ Move up, down, left, right 0 to 2 spaces randomly """
        lst = []
        for i in range(-self.move_dist, self.move_dist + 1):
            lst.append(i)

        if WRAP:
            self.x = (self.x + rnd.choice(lst)) % SIZE
            self.y = (self.y + rnd.choice(lst)) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice(lst))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice(lst))))


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass and 0 or more foxes in search of rabbits """

    def __init__(self):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no rabbits """
        self.rabbits = []
        self.foxes = []
        self.field = np.ones(shape=(SIZE, SIZE), dtype=int)
        self.nrabbits = []
        self.nfoxes = []
        self.ngrass = []
        self.history = defaultdict(lambda: 0)

    def add_rabbit(self, rabbit):
        """ A new rabbit is added to the field """
        self.rabbits.append(rabbit)

    def add_fox(self, fox):
        """ A new fox is added to the field """
        self.foxes.append(fox)

    def rabbit_move(self):
        """ Rabbits move """
        for r in self.rabbits:
            r.move()

    def fox_move(self):
        """ Foxes move and documents what generation it is currently in"""
        for fox in self.foxes:
            fox.move()
            fox.generation += 1

    def rabbit_eat(self):
        """ Rabbits eat (if they find grass where they are) """
        for rabbit in self.rabbits:
            rabbit.eat(self.field[rabbit.x, rabbit.y])
            self.field[rabbit.x, rabbit.y] = 0

    def fox_eat(self):
        """ Foxes eat (if they encounter rabbit) """
        for fox in self.foxes:
            for rabbit in self.rabbits:
                if fox.x == rabbit.x and fox.y == rabbit.y:
                    fox.eat(1)
                    fox.generation = 0
                    self.rabbits.remove(rabbit)

    def rabbit_survive(self):
        """ Rabbits who eat some grass live to eat another day """
        self.rabbits = [r for r in self.rabbits if r.eaten > 0]

    def fox_survive(self):
        """ Foxes who's generation are still within k """
        self.foxes = [fox for fox in self.foxes if (fox.generation <= K or fox.eaten > 0)]

    def reproduce(self):
        """ Rabbits reproduce like rabbits and foxes reproduce like fox if they've eaten """
        rabbits_born = []
        for rabbit in self.rabbits:
            if rabbit.eaten > 0:
                for _ in range(rnd.randint(1, RABBIT_OFFSPRING)):
                    rabbits_born.append(rabbit.reproduce())
        self.rabbits += rabbits_born

        foxes_born = []
        for fox in self.foxes:
            if fox.eaten > 0:
                for _ in range(rnd.randint(1, FOX_OFFSPRING)):
                    foxes_born.append(fox.reproduce())
        self.foxes += foxes_born

        # Capture field state for historical tracking
        self.nrabbits.append(self.num_rabbits())
        self.nfoxes.append(self.num_foxes())
        self.ngrass.append(self.amount_of_grass())

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_rabbits(self):
        """ Get the rabbit """
        rabbits = np.zeros(shape=(SIZE, SIZE), dtype=int)
        for r in self.rabbits:
            rabbits[r.x, r.y] = 1
        return rabbits

    def get_foxes(self):
        """ Get the foxes """
        foxes = np.zeros(shape=(SIZE, SIZE), dtype=int)
        for fox in self.foxes:
            foxes[fox.x, fox.y] = 1
        return foxes

    def num_rabbits(self):
        """ How many rabbits are there in the field ? """
        return len(self.rabbits)

    def num_foxes(self):
        """ How many foxes are there in the field ? """
        return len(self.foxes)

    def amount_of_grass(self):
        return self.field.sum()

    def generation(self):
        """ Run one generation of rabbits and foxes """
        self.rabbit_move()
        self.fox_move()
        self.rabbit_eat()
        self.fox_eat()
        self.rabbit_survive()
        self.fox_survive()
        self.reproduce()
        self.grow()
        self.history[self.nrabbits] = self.rabbit_survive()
        self.history[self.nfoxes] = self.fox_survive()
        self.history[self.ngrass] = self.grow

    def history(self, showTrack=True, showPercentage=True, marker='.'):
        """ Plots the whole history of the field after 1000 generations
         showTrack (bool): track the number of populations (display line graph)
         showPercentage (bool): display the percent of animal population
         marker (str): marker (shape) on graph """

        # list of number of animals in each generation
        nrabbits = self.history[2]
        nfoxes = self.history[3]
        ngrass = self.history[1]

        # builds the plot
        plt.figure(figsize=(6, 6))
        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")

        # creates a copy of the rabbit population
        xs = nrabbits[:]

        # find the percentage of rabbit population in each cycle
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x / maxrabbit for x in xs]
            plt.xlabel("% Rabbits")

        # grass population
        ys = ngrass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [y / maxgrass for y in ys]
            plt.ylabel("% Rabbits")

        # fox population
        zs = nfoxes[:]
        if showPercentage:
            maxfox = max(zs)
            zs = [z / maxfox for z in zs]
            plt.ylabel("% Foxes")

        # sets type of graph
        if showTrack:
            plt.plot(xs, ys, marker=marker, label='Rabbits')
            plt.plot(xs, zs, marker=marker, label='Foxes')
        else:
            plt.scatter(xs, ys, marker=marker, label='Rabbits')
            plt.scatter(xs, zs, marker=marker, label='Foxes')

        plt.grid()

        # add labels to the graph
        plt.title("Population through the generations: GROW_RATE =" + str(GRASS_RATE))
        plt.legend()
        plt.savefig("Animal_growth_rate.png", bbox_inches='tight')
        plt.show()


def animate(i, field, im):
    field.generation()
    # print("AFTER: ", i, np.sum(field.field), len(field.rabbits))
    total = field.field.copy()
    for rabbit in field.rabbits:
        total[rabbit.x, rabbit.y] = 2
    for fox in field.foxes:
        total[fox.x, fox.y] = 3
    # print(total)
    im.set_array(total)
    plt.title("generation = " + str(i))
    return im,


def main():
    # include the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("grass_rate", help="The grass growth rate", type=float)
    parser.add_argument("k", help="The fox k value", type=int)
    parser.add_argument("size", help="The field size", type=int)
    parser.add_argument("fox_num", help="The number of initial foxes in the field ", type=int)
    parser.add_argument("rabbit_num", help="The number of initial rabbits in the field ", type=int)
    parser.add_argument("fox_dist", help="The max distance foxes in the field can move", type=int)
    parser.add_argument("rabbit_dist", help="The max distance rabbits in the field can move", type=int)
    args = parser.parse_args()

    message = (f'The grass growth rate will be: {args.grass_rate} \n'
               f'The fox k value will be: {args.k} \n'
               f'The field size will be: {args.size} \n'
               f'The number of initial foxes in the field: {args.fox_num} \n'
               f'The number of initial rabbits in the field: {args.rabbit_num} \n'
               f'The max distance of foxes in the field can move: {args.fox_dist} \n'
               f'The max distance of rabbits in the field can move: {args.rabbit_dist}'
               )

    print(message)

    if args.grass_rate:
        GRASS_RATE = args.grass_rate

    if args.k:
        K = args.k

    if args.size:
        SIZE = args.size

    if args.rabbit_num:
        RABBIT_NUM = args.rabbit_num

    if args.fox_num:
        FOX_NUM = args.fox_num

    if args.rabbit_dist:
        RABBIT_DIST = args.rabbit_dist

    if args.fox_dist:
        FOX_DIST = args.fox_dist

    # Create the ecosystem
    field = Field()

    for _ in range(RABBIT_NUM):
        field.add_rabbit(Animal(RABBIT_DIST))

    for _ in range(FOX_NUM):
        field.add_fox(Animal(FOX_DIST))

    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5, 5))
    cmap = mcolors.ListedColormap(['tan', 'green', 'blue', 'red'])
    bounds = [0, 1, 2, 3]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im = plt.imshow(array, cmap=cmap, interpolation='None', aspect='auto', vmin=0, vmax=3)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000, interval=1, repeat=False)
    plt.show()

    # show the plot
    field.history()


if __name__ == '__main__':
    main()
FooterNortheastern University
Northeastern University
Northeastern University
