import numpy as np
import os
from PySpice.Spice.Parser import SpiceParser
from PyLTSpice import SimCommander
from PyLTSpice import RawRead

def Create_Structure(net_name, Potential, R, input, output, parasiticResistance):
    r_sampling = 0.00000000001

    with open(net_name,'w', encoding= 'UTF-8') as f:
        
        f.write("REAL.CIR\n")

        for i in range(input):
            j = 0
            f.write("V" + str(i) + "\t W"+str(i)+"_"+str(0)+" \t 0 \t" +str(Potential[i]) +" \n")

            for j in range(output):
                # Word parasitic resistance RW0_0 W0_0 W0_1 10
                f.write("RW" + str(i)+"_" + str(j) + "\t W"+str(i)+"_" +str(j)+"\t W" +str(i)+"_" +str(j+1)+" \t" + str(parasiticResistance)+ "\n")
                
                # RRAM resistance R0_0 W0_1 B0_0 1k
                f.write("R" + str(i) +"_"+ str(j) + "\t W"+str(i)+"_" +str(j+1)+"\t B" +str(i)+"_" +str(j)+" \t" + str(R[i,j])+ "\n")

                # Bit parasitic resistance RB0_0 B0_0 B0_1 10
                f.write("RB" + str(i) +"_"+ str(j) + "\t B"+str(i)+"_" +str(j)+"\t B" +str(i+1)+"_" +str(j)+" \t " + str(parasiticResistance)+ "\n")
                
                # Current sensing
                if i == input-1:
                    f.write("RS" + str(j) + "\t B"+str(i+1)+"_" +str(j)+"\t p"+str(i+1)+"_" +str(j)+ "\t " + str(r_sampling)+ "\n")
                    f.write("VS" + str(j) + "\t p"+str(i+1)+"_" +str(j)+"\t 0 \t " + str(0)+ "\n")
        f.write(".op \n")
        f.write(".end")


def Create_NonLinear_Structure(net_name, Potential, input, output, parasiticResistance, X, S):
    r_sampling = 0.000001

    with open(net_name,'w', encoding= 'UTF-8') as f:
        
        f.write("NonLinear.sp\n")
        for i in range(input):
            j = 0
            f.write("V" + str(i) + "\t W"+str(i)+"_"+str(0)+" \t 0 \t" +str(Potential[i]) +" \n")

            for j in range(output):
                # Word parasitic resistance RW0_0 W0_0 W0_1 10
                f.write("RW" + str(i)+"_" + str(j) + "\t W"+str(i)+"_" +str(j)+"\t W" +str(i)+"_" +str(j+1)+" \t" + str(parasiticResistance)+ "\n")

                # RRAM non linear resistance X0_0 W0_1 B0_0 RRAM x=... s=...
                f.write("X" + str(i) +"_"+ str(j) + "\t W"+str(i)+"_" +str(j+1)+"\t B" +str(i)+"_" +str(j)+" \t" + "RRAM \t x=" + str(X[i,j])+ "\t s=" + str(S[i,j])+"\n")

                # Bit parasitic resistance RB0_0 B0_0 B0_1 10
                f.write("RB" + str(i) +"_"+ str(j) + "\t B"+str(i)+"_" +str(j)+"\t B" +str(i+1)+"_" +str(j)+" \t " + str(parasiticResistance)+ "\n")

                # Current sensing
                if i == input-1:
                    f.write("VS" + str(j) + "\t B"+str(i+1)+"_" +str(j)+"\t 0 \t " + str(0)+ "\n")
        f.write(".op \n")
        f.write(".end")


def ngSpice_Sim(net_name: str, folder_results, folder_model, voltage_drops, nonLinear):
    # Parse the circuit file
    parser = SpiceParser(net_name)
    circuit = parser.build_circuit()
    if nonLinear:
        circuit.include(folder_model + 'rram_model.sp')
    
    # Set up simulator
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    simulator.options(
        reltol=1e-4,      # Looser relative tolerance
        abstol=1e-14,     # Looser absolute tolerance for currents
        vntol=1e-8,       # Looser voltage tolerance
        gmin=1e-10,       # Set an initial gmin to help with stability
        gminsteps=100,    # Number of gmin steps for gradual increments
        itl1=200,         # Maximum number of iterations before stepping
        method="gear"     # Gear method for stability with nonlinear devices
    )
    analysis = simulator.operating_point()

    # Initialize NumPy matrices
    input = voltage_drops.shape[0]
    output = voltage_drops.shape[1]
    bi_j_matrix = np.zeros((input, output))
    wi_j_matrix = np.zeros((input, output))

    # Iterate through the node results and fill the respective matrices
    for node in analysis.nodes.values():
        node_name = str(node)
        voltage = float(node)
        if node_name.startswith('b'):
            i, j = map(int, node_name[1:].split('_'))
            if i < input and j < output: 
                bi_j_matrix[i, j] = voltage  # Fill the bi_j_matrix
        elif node_name.startswith('w'):
            i, j = map(int, node_name[1:].split('_'))
            if i < input and j < output: 
                wi_j_matrix[i, j] = voltage  # Fill the wi_j_matrix

    # Calculate voltage drops
    voltage_drops = wi_j_matrix - bi_j_matrix

    # Save matrices to CSV
    np.savetxt(folder_results + '/bi_j_matrix.csv', bi_j_matrix, delimiter=',')
    np.savetxt(folder_results + '/wi_j_matrix.csv', wi_j_matrix, delimiter=',')
    np.savetxt(folder_results + '/voltage_drops.csv', voltage_drops, delimiter=',')

    # Retrieve currents through resistors starting with "RS"
    currents = []
    for branch_name, current in analysis.branches.items():
        if branch_name.startswith("vs"):
            currents.append(np.abs(float(current)))  # Convert current to float and add to list

    # Save currents to CSV
    np.savetxt(folder_results + '/currents.csv', currents, delimiter=',')
    # print(np.array(currents))
    return voltage_drops, np.array(currents)  # Return voltage drops and currents as array




def LTSpice_Sim(net_name : str,  current_matrix):
    LTC = SimCommander(net_name, encoding='UTF-8', parallel_sims=6)
    LTC.run()
    LTC.wait_completion()
    
    raw_file = os.path.splitext(net_name)[0]+'_1.raw'
    if os.path.exists(raw_file):
        raw = RawRead(raw_file)
    else:
        raise FileNotFoundError(f"Raw file {raw_file} not found.")
    input, output= current_matrix.shape
    for j in range(output):
        current = (raw.get_trace("I(RS"+str(j)+")")).get_wave()

    for i in range(input):
        for j in range(output):
                current_matrix[i,j] = (raw.get_trace("I(R"+str(i)+"_"+str(j)+")")).get_wave()

    return current_matrix, current
