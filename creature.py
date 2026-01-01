import genome 
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np

class MotorType(Enum):
    PULSE = 1
    SINE = 2

class Motor:
    def __init__(self, control_waveform, control_amp, control_freq, sensor_weight=0):
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        self.phase = 0
        self.sensor_weight = sensor_weight  # how much this motor responds to direction sensor


    def get_output(self, sensor_signal=0):
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            if self.phase < np.pi:
                output = 1
            else:
                output = -1

        if self.motor_type == MotorType.SINE:
            output = np.sin(self.phase)

        # modulate output based on sensor input (if we have any)
        # sensor_signal is like -1 to 1 (direction to peak)
        # sensor_weight controls how much the motor cares about it
        if self.sensor_weight > 0 and sensor_signal != 0:
            output = output * (1 + self.sensor_weight * sensor_signal)

        return output 

class Creature:
    def __init__(self, gene_count):
        self.spec = genome.Genome.get_gene_spec()
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count)
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        self.target_position = None

    def set_target(self, pos):
        self.target_position = pos

    def get_flat_links(self):
        if self.flat_links == None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links
    
    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links
        
        exp_links = [self.flat_links[0]]
        genome.Genome.expandLinks(self.flat_links[0], 
                                self.flat_links[0].name, 
                                self.flat_links, 
                                exp_links)
        self.exp_links = exp_links
        return self.exp_links

    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))
        first = True
        for link in self.exp_links:
            if first:# skip the root node! 
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))
        robot_tag.setAttribute("name", "pepe") #  choose a name!
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        self.get_expanded_links()
        if self.motors == None:
            motors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                # pass sensor_weight so motors can respond to direction sensor
                m = Motor(l.control_waveform, l.control_amp, l.control_freq, l.sensor_weight)
                motors.append(m)
            self.motors = motors
        return self.motors 
    
    def update_position(self, pos):
        if self.start_position == None:
            self.start_position = pos
        else:
            self.last_position = pos

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
    
    def get_fitness(self):

        if self.start_position is None or self.last_position is None:
         return 0

        # In case of no targetfall back to distance travelled (for retro compatibility)
        if self.target_position is None:
         return self.get_distance_travelled()

        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        target = np.asarray(self.target_position)

        # calculate fitness based on progress towards target
        initial_dist_to_target = np.linalg.norm(p1 - target)
        final_dist_to_target = np.linalg.norm(p2 - target)

        progress = initial_dist_to_target - final_dist_to_target

        # penaltiy for flying/jumping
        if abs(p2[2] - p1[2]) > 2:
            progress = progress * 0.5

        
        fitness = max(0, progress + initial_dist_to_target * 0.1)
        
        return fitness
    

    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None

