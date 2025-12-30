## Set up and testing starting point

1. I created a virtual environment using anaconda (I was habving troubles installing pybullet with pip as it was failing to build the wheel).
   The environment used python 3.9 (well known for good stability and compatibility), pybullet 3.25, and matplot lib 3.9.2.

2. I tested that everithing works by loading the new mountain env and testing it with the default random creature. I was able to see what expected, I just had to add the usual block:

```code
while True:
    p.stepSimulation()
    time.sleep(1./240.)
```

to make it work on my mac m1.

3. Before now before loading any creature in the new sandbox I wanted to make sure that before climbing the creature could at least walk and try that as baseline, so I just ran the test_ga_no_threads script and see which creature I could get.

The starting script as it was from the starter code was saving a new creature for each iteration, and also the creatures fitness were regressing too much, so the first thing I changed was saving the best creature overall and implement elitism in a way that the best overall was added to each new generation:

```python
class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop = population.Population(pop_size=25,
                                    gene_count=4)
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

```

and then added a log file to track the progress of the evolution:

```python
if iteration == 0:
    import time
    timestamp = int(time.time())
    fmt_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"iteration_logs_{fmt_timestamp}.csv", "w") as f:
        f.write("iteration,fittest,mean_fitness,mean_links,max_links\n")
with open(f"iteration_logs_{fmt_timestamp}.csv", "a") as f:
    f.write(f"{iteration},{np.max(fits)},{np.mean(fits)},{np.mean(links)},{np.max(links)}\n")
```

and implemnted a penalty for huge jumps in the get_distance_travelled method of creature.py, given that in certain runs the creatures were suspiciouly going from a fitnes of ~10/11 to a fitness of ~130 in one generation and then when loading the creature in the env they were just jumping forward, probably due to some weird combination of link lengths and joint angles/amplitudes/frequencies.:

```python

    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        dist = np.linalg.norm(p1-p2)

        #penalize jumps/flying behaviors
        if abs(p2[2] - p1[2]) > 2:
            dist = dist * 0.5

        return dist
```

I tried a full run of 40 generations with population 40 and gene count 5, which
seemed to produce acceptable results (quick baseline with default parameters), with a baseline creature that was least able to walk.

```csv
iteration,fittest,mean_fitness,mean_links,max_links
0,3.8648283324293016,2.108679275898718,14.175,34
1,4.145528164228674,2.1374424819670823,11.85,34
2,4.145528164228674,2.187375028747847,9.175,33
...
991,11.002590238923963,2.742678374540616,2.725,6
992,11.063062588175942,2.6825482384751482,3.05,8 <-- last new best creature found
993,11.063062588175942,2.503953486231804,2
...
```

4. Next up I will try to load this baseline creature in the new mountain env and see how it performs there.
