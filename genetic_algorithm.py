import random
import numpy as np
from musicpy import *
from read_data import readBars, readMainTracks

class GeneticAlgorithm:
    def __init__(
        self,
        population,
        fitness,
        mutation_rate,
        crossover_rate,
        selection_rate,
        generations,
    ):
        self.mutation_rate = mutation_rate
        # Probability of mutation: which is the probability that a gene will change its value
        self.crossover_rate = crossover_rate
        # Probability of crossover: which is the probability that two chromosomes will exchange genetic information
        self.selection_rate = selection_rate
        # Selection rate: which is the percentage of the population that will be selected for crossover
        self.generations = generations  # Number of generations
        self.population = population
        self.fitness = fitness

    def _selection(self):
        selected = random.choices(
            self.population,
            k=int(self.selection_rate * len(self.population)),
            weights=[self.fitness(chrom) for chrom in self.population],
        )
        return selected

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.rand() * parent1.bars()
            return parent1.cut(0, point) + parent2.cut(point, parent2.bars()), \
                   parent2.cut(0, point) + parent1.cut(point, parent1.bars())
        return parent1, parent2

    def _mutate(self, chord):
        for i in range(len(chord.notes)):
            if random.random() < self.mutation_rate:
                chord.notes[i] += random.choice([-1, 1])
        return chord

    def _create_new_population(self, selected):
        new_population = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = selected[i], selected[i + 1]
            offspring1, offspring2 = self._crossover(parent1, parent2)
            new_population.append(self._mutate(offspring1))
            new_population.append(self._mutate(offspring2))
        return new_population

    def run(self):
        from tqdm import tqdm
        for generation in tqdm(range(self.generations)):
            selected = self._selection()
            self.population = self._create_new_population(selected)
            best_fitness = max(self._fitness(chrom) for chrom in self.population)
            best_genome = self.population[np.argmax([self._fitness(chrom) for chrom in self.population])]
            print(f"Epoch = {generation}; Best Fitness = {best_fitness}")
            yield best_genome, best_fitness

def _initialize_population():
    # syf
    """
    should return a list. every element is a musicpy's chord object,which looks like:
        chord(notes=[C4, E4, G4, C4, E4, G4, C4, E4, G4, E3, ...], interval=[0, 0, 1/2, 0, 0, 1/2, 0, 0, 1/2, 1/8, ...], start_time=0)
    """
    # 如果以小节作为population，用readBars；如果用主旋律做population，用readMainTracks
    '''
    chords = readMainTracks()
    return chords
    '''
    BAR_MIN_LENGTH = 5
    BAR_MAX_LENGTH = 10
    bars = readBars()
    filtered_bars = []
    for bar in bars:
        if len(bar) < BAR_MIN_LENGTH or len(bar) > BAR_MAX_LENGTH:
            continue
        filtered_bars.append(bar)
    return filtered_bars


def fitness(chord):
    # hcy
    """
    input a chord (with the form above), output a float number, which is the fitness of this sequence.
    """
    """
    计算输入和弦的适应度：
    1. 音程质量（不和谐度）。
    2. 计算调性外音符。
    3. 比较与参考和弦（初始种群）的均值和方差。
    """
    # 超参
    alpha, beta, gama = 1.0, 1.0, 1.0

    fitness_score = 0

    # 提取和弦的音程
    intervals = chord.intervalof(translate=True, cumulative=False)

    values = []
    for interval in intervals:
        if interval.num < 9:
            values.append(self.interval_values[interval.num - 1])
        else:
            values.append(5)
    mean = np.mean(values)
    variance = np.var(values)

    # 统计调性外音符
    out_of_tonality_notes_num = 0
    for note in chord.notes:
        flag = False
        for tonality_note in self.tonality_notes:
            if (note.degree - tonality_note.degree) % 12 == 0:
                flag = True
                break
        if not flag:
            out_of_tonality_notes_num += 1
    fitness_score += out_of_tonality_notes_num * gama

    # 将均值差异和方差差异加入到适应度中
    fitness_score += abs(mean - self.initial_mean) * alpha
    fitness_score += abs(variance - self.initial_variance) * beta

    # 返回适应度分数（分数越低适应度越高）
    return fitness_score


if __name__ == "__main__":
    music_piece = _initialize_population()
    ga = GeneticAlgorithm(
        population=music_piece,
        fitness=fitness,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection_rate=1.0,
        generations=100,
    )
    result = ga.run()
    with open("result.txt", "w") as f:
        for best_genome, best_fitness in result:
            f.write(f"{best_genome}\t{best_fitness}\n")
