# this version works on M1 macs and windows
# the multithreaded one doesnt work so use this instead

import unittest
import population
import simulation
import genome
import creature
import numpy as np
import time
from pathlib import Path

# default params for the GA
DEFAULT_CONFIG = {
    'pop_size': 40,
    'gene_count': 5,
    'point_mutate_rate': 0.1,
    'point_mutate_amount': 0.25,
    'shrink_mutate_rate': 0.25,
    'grow_mutate_rate': 0.1,
    'elitism_count': 1,
    'max_stagnant_generations': 75,
    'simulation_iterations': 2400,
    'peak_position': (0, 0, 5),
}


def run_ga(config=None, save_dir=None):
    """
    runs the genetic algorithm with given config
    returns dict with results
    """
    # merge config with defaults
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # setup population and sim
    pop = population.Population(pop_size=cfg['pop_size'], gene_count=cfg['gene_count'])
    sim = simulation.SimulationMountain(peak_position=cfg['peak_position'])

    # tracking stuff
    best_fitness = 0
    best_dna = None
    best_gen = 0
    stagnant = 0
    history = []

    # setup log file
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        logfile = Path(save_dir) / f"iteration_logs_{ts}.csv"
    else:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        logfile = f"iteration_logs_{ts}.csv"

    # write header
    with open(logfile, "w") as f:
        f.write("generation,best_fitness,mean_fitness,std_fitness,best_ever,mean_links,max_links\n")

    # main loop
    for gen in range(10000):
        print(f"Generation {gen} | Stagnant: {stagnant}")

        # check early stopping
        if stagnant >= cfg['max_stagnant_generations']:
            print(f"Early stopping at gen {gen} (no improvement for {cfg['max_stagnant_generations']} gens)")
            break

        # evaluate creatures
        for cr in pop.creatures:
            sim.run_creature(cr, cfg['simulation_iterations'])

        # get fitness values
        fits = [cr.get_fitness() for cr in pop.creatures]
        links = [len(cr.get_expanded_links()) for cr in pop.creatures]

        # save stats
        stats = {
            'generation': gen,
            'best_fitness': np.max(fits),
            'mean_fitness': np.mean(fits),
            'std_fitness': np.std(fits),
            'best_ever': best_fitness,
            'mean_links': np.mean(links),
            'max_links': np.max(links)
        }
        history.append(stats)

        # write to log
        with open(logfile, "a") as f:
            f.write(f"{gen},{stats['best_fitness']},{stats['mean_fitness']},{stats['std_fitness']},{stats['best_ever']},{stats['mean_links']},{stats['max_links']}\n")

        print(f"  Best: {np.max(fits):.3f} | Mean: {np.mean(fits):.3f} | Links: {np.mean(links):.1f} (max {np.max(links)})")

        # check if we found new best
        max_fit = np.max(fits)
        if max_fit > best_fitness:
            stagnant = 0
            best_fitness = max_fit
            best_gen = gen

            # save best creature
            for cr in pop.creatures:
                if cr.get_fitness() == max_fit:
                    best_dna = cr.dna.copy()
                    if save_dir:
                        genome.Genome.to_csv(best_dna, str(Path(save_dir) / "best_ever.csv"))
                    else:
                        genome.Genome.to_csv(best_dna, "best_ever.csv")
                    print(f"  NEW BEST: {best_fitness:.3f}")
                    break
        else:
            stagnant += 1

        # sort by fitness for elitism
        sorted_idx = np.argsort(fits)[::-1]

        # selection
        fit_map = population.Population.get_fitness_map(fits)
        new_pop = []

        # create offspring
        num_offspring = cfg['pop_size'] - cfg['elitism_count']
        for _ in range(num_offspring):
            p1_idx = population.Population.select_parent(fit_map)
            p2_idx = population.Population.select_parent(fit_map)
            p1 = pop.creatures[p1_idx]
            p2 = pop.creatures[p2_idx]

            # crossover and mutate
            dna = genome.Genome.crossover(p1.dna, p2.dna)
            dna = genome.Genome.point_mutate(dna, rate=cfg['point_mutate_rate'], amount=cfg['point_mutate_amount'])
            dna = genome.Genome.shrink_mutate(dna, rate=cfg['shrink_mutate_rate'])
            dna = genome.Genome.grow_mutate(dna, rate=cfg['grow_mutate_rate'])

            new_cr = creature.Creature(1)
            new_cr.update_dna(dna)
            new_pop.append(new_cr)

        # add elite creatures (best ones survive unchanged)
        for i in range(cfg['elitism_count']):
            elite_idx = sorted_idx[i]
            elite = creature.Creature(1)
            elite.update_dna(pop.creatures[elite_idx].dna.copy())
            new_pop.append(elite)

        pop.creatures = new_pop

    # return results
    return {
        'best_fitness': best_fitness,
        'final_mean_fitness': np.mean(fits) if fits else 0,
        'generations_run': gen + 1,
        'converged_at': best_gen,
        'best_dna': best_dna,
        'history': history,
        'log_file': str(logfile),
        'config': cfg
    }


class TestGA(unittest.TestCase):
    def testBasicGA(self):
        result = run_ga()
        self.assertGreater(result['best_fitness'], 0)
        print(f"\nTest done:")
        print(f"  Best fitness: {result['best_fitness']:.3f}")
        print(f"  Generations: {result['generations_run']}")
        print(f"  Converged at: {result['converged_at']}")


if __name__ == "__main__":
    unittest.main()
