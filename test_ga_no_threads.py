# If you on a Windows machine with any Python version 
# or an M1 mac with any Python version
# or an Intel Mac with Python > 3.7
# the multi-threaded version does not work
# so instead, you can use this version. 

import unittest
import population
import simulation 
import genome 
import creature 
import numpy as np
import time

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop = population.Population(pop_size=40, 
                                    gene_count=5)
        sim = simulation.Simulation()

        # tracking global best
        best_ever_fitness = 0
        best_ever_dna = None

        for iteration in range(1000):
            for cr in pop.creatures:
                sim.run_creature(cr, 2400)            
            fits = [cr.get_distance_travelled() 
                    for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) 
                    for cr in pop.creatures]
            
            #save iteration logs:
            #1 write a new log file everytime the script is called: e.g. ga_log_<timestamp>.txt
            #appen time stamop to filename to avoid overwriting

           

            #2 append to the log file the iteration number, fittest, mean fitness, mean links, max links
            if iteration == 0:
                        import time
                        timestamp = int(time.time())
                        fmt_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                        with open(f"iteration_logs_{fmt_timestamp}.csv", "w") as f:
                            f.write("iteration,fittest,mean_fitness,mean_links,max_links\n")
           
            with open(f"iteration_logs_{fmt_timestamp}.csv", "a") as f:
                f.write(f"{iteration},{np.max(fits)},{np.mean(fits)},{np.mean(links)},{np.max(links)}\n")
            
            print(iteration, "fittest:", np.round(np.max(fits), 3), 
                  "mean:", np.round(np.mean(fits), 3), "mean links", np.round(np.mean(links)), "max links", np.round(np.max(links)))       

            # updating   global best if new found
            max_fit = np.max(fits)
            if max_fit > best_ever_fitness:
                best_ever_fitness = max_fit
                for cr in pop.creatures:
                    if cr.get_distance_travelled() == max_fit:
                        best_ever_dna = cr.dna.copy()
                        genome.Genome.to_csv(best_ever_dna, "best_ever.csv")
                        print(f"  NEW BEST: {best_ever_fitness:.3f}")
                        break
            
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            for i in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)
            
            # for each generation, keep the best ever creature
            if best_ever_dna is not None:
                new_cr = creature.Creature(1)
                new_cr.update_dna(best_ever_dna)
                new_creatures[0] = new_cr
            
            pop.creatures = new_creatures
                            
        self.assertNotEqual(fits[0], 0)

unittest.main()
