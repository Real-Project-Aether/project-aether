# Mechanism: how the head's odor sample reaches the motor neurons

## The chain

1. **Sample to sensory neuron.** The odor sample enters as a current in the attractant-sensing
   neuron. Its response adapts, so the cell reports the change of the sample over the swing rather
   than its level.

2. **Sensory neuron to first-layer interneurons.** The sensory neuron drives AIY and AIZ through
   chemical synapses of opposite sign, excitatory onto AIY and inhibitory onto AIZ. AIY and AIZ
   have high input resistance, so a current of a few picoamperes moves them by several millivolts;
   this gain is what lets a picoampere sample move the motor neurons by millivolts.

3. **Interneurons to the head motor neurons.** AIY and AIZ converge on RIA, whose compartments
   drive the SMD and RMD head motor neurons dorsally and ventrally. The delay along this path
   (sensory neuron, AIY/AIZ, RIA, SMD/RMD) sets the phase of the head motor pattern against the
   swing of the sample, so that the head bends towards the side where the sample was rising.

4. **Head to the cord.** The command interneuron AVB drives the B-class cord motor neurons, which
   with the A and D classes carry the wave along the body. The head signal enters the cord through
   chemical synapses from the head circuit onto AVB.

5. **Gap junctions are redundant for the swing.** Every step that carries the sample is a chemical
   synapse, so removing every gap junction leaves the head, A, B and D motor neurons swinging about
   as much as before; the gap junctions only smooth the activity within each class.

6. **Background level.** Because the sensory neuron adapts, a higher background level moves its
   operating point towards saturation, so less of a large swing passes through.

## Limitations

Proprioceptive feedback from body bending is not represented in the simulator; how the polarity of
each synapse of a multi-transmitter neuron is fixed, and how connections change during foraging,
are outside what the simulator delivers.

## Evidence

Our stimulation and wiring experiments were consistent with each step (experiment.log).
