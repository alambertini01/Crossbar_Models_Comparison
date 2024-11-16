import os
import numpy as np
from scipy.optimize import fsolve

def create_rram_model(filepath):
    """
    Creates a SPICE model file for a filamentary RRAM device.
    
    Args:
        filepath (str): Path where the SPICE model file should be created
    """
    model_text = """* Filamentary RRAM Model Definition
.subckt RRAM Te Be
+ x=0                    ; Filament length (nm)
+ s=10.62                   ; Conductive filament area (nm^2)
+ T=303.15                  ; Temperature (K)
+ V0hrs=0.3326            ; Characteristic voltage (V)
+ rho=3000              ; Resistivity (Ω⋅nm)
+ tox=5                ; Oxide thickness (nm)
+ beta=0.199             ; Enhancement factor
+ l=0.42                ; Characteristic length (nm)
+ Ea=0.0513               ; Activation energy (eV)
+ Kb=8.617e-5           ; Boltzmann constant (eV/K)

* Calculate parameters
.param Rlrs={rho*tox/s}
.param Rbar={Rlrs * beta * (exp(x/l) - 1) * exp(Ea/(Kb*T))}
.param Rcf={rho * (tox - x) / s}

* Non-linear element using voltage-controlled current source
Grram Te n1 value={V0hrs / Rbar * sinh(V(Te, n1) / V0hrs)}

* Linear resistor
R1 n1 Be {Rcf}
.ends RRAM"""

    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Write the SPICE model file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(model_text)
        
    except Exception as e:
        print(f"Error creating SPICE model file: {str(e)}")


def resistance_array_to_x(r_array):
    """
    Convert an array of resistance values to corresponding x values
    
    Parameters:
    r_array : numpy array of resistance values (Ω), can be multi-dimensional
    
    Returns:
    numpy array of x values (nm) with the same shape as input
    """
    # Constants
    s = 10.62        # Conductive filament area (nm^2)
    T = 303.15       # Temperature (K)
    rho = 3000    # Resistivity (Ω⋅nm)
    tox = 5      # Oxide thickness (nm)
    beta = 0.199   # Enhancement factor
    l = 0.42      # Characteristic length (nm)
    Ea = 0.0513     # Activation energy (eV)
    Kb = 8.617e-5 # Boltzmann constant (eV/K)

    def get_total_resistance(x):
        """Calculate total resistance for a given x"""
        Rlrs = rho * tox / s
        Rbar = Rlrs * beta * (np.exp(x/l) - 1) * np.exp(Ea/(Kb*T))
        Rcf = rho * (tox - x) / s
        return Rbar + Rcf
    
    def find_x_for_resistance(target_r):
        """Find x value for a target resistance using numerical optimization"""
        def error_func(x):
            return get_total_resistance(x) - target_r
        
        # Try different initial guesses
        initial_guesses = [1,2,3]
        best_x = None
        best_error = float('inf')
        
        for guess in initial_guesses:
            try:
                x_solution = fsolve(error_func, guess)[0]
                if 0 <= x_solution <= tox:
                    error = abs(error_func(x_solution))
                    if error < best_error:
                        best_error = error
                        best_x = x_solution
            except:
                continue
            
        return best_x if best_x is not None else np.nan
    
    # Convert input to numpy array if it isn't already
    r_array = np.asarray(r_array)
    original_shape = r_array.shape
    
    # Flatten array, process, then reshape back
    flat_r = r_array.flatten()
    flat_x = np.array([find_x_for_resistance(r) for r in flat_r])
    x_values = flat_x.reshape(original_shape)
    
    return x_values




def calculate_resistance(x, s):
    """
    Calculate RRAM resistances for given filament lengths x and areas s.
    Returns: Total resistance matrix.
    """
    # Constants
    T = 303.15       # Temperature (K)
    rho = 3000    # Resistivity (Ω⋅nm)
    tox = 5      # Oxide thickness (nm)
    beta = 0.199   # Enhancement factor
    l = 0.42      # Characteristic length (nm)
    Ea = 0.0513     # Activation energy (eV)
    Kb = 8.617e-5 # Boltzmann constant (eV/K)

    Rlrs = rho * tox / s  # Calculate Rlrs
    Rbar = Rlrs * beta * (np.exp(x / l) - 1) * np.exp(Ea / (Kb * T))
    Rcf = rho * (tox - x) / s  # Calculate Rcf
    return Rbar + Rcf  # Return total resistance

# # check
# print(resistance_array_to_x(np.array([100000])))
# print(calculate_resistance(resistance_array_to_x(np.array([100000])),10.62))