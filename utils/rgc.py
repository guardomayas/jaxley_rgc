import jax.numpy as jnp
import jaxley as jx
import jax
# from jaxley_mech.channels.fm97 import Na, K, Leak, KA, KCa, Ca, CaNernstReversal, CaPump
from jaxley_mech.channels.benison01 import Na, Leak, KA, KCa, Kdr, CaL, CaN, CaNernstReversal,CaPumpNS 

def build_cell(
    R_Mohm = 228.7,
    C_tot_pf = 56.24
            
    ):
    cell = jx.Cell()
    cell.insert(Na())
    cell.insert(Leak())
    cell.insert(KA())
    cell.insert(KCa())
    cell.insert(Kdr())
    cell.insert(CaL())
    cell.insert(CaN())
    
    G_S = 1.0 / (R_Mohm * 1e6)          # S
    area_um2 = C_tot_pf * 100.0 ##Asuuming Cm of 1. 
    area_cm2 = area_um2 * 1e-8         # cm^2
    gLeak = G_S / area_cm2             # S/cm^2
    
    cell.set("area", area_um2) # µm^2 Ctot =58pF ## Sets global capacitance C=eA/d ==> Ctot​(pF)=Cm​(μF/cm2)×area(μm2)/100.
    cell.set("Leak_gLeak", float(gLeak))
    
    return cell

def simulate_one_trial(cell, I_one, dt_cell): 
    """ 
    Need to extend this to handle chunks/partition one trial  
    """
    cell.delete_stimuli()
    data_stimuli = None
    data_stimuli = cell.data_stimulate(current=I_one, data_stimuli=None, verbose=False)
    v = jx.integrate(cell, delta_t=dt_cell, data_stimuli=data_stimuli)
    return v[0, :-1]  # (T_cell,)

def rgc_forward_sim(I_syn, dt_cell, cell_template):
    """
    I_syn: (B, T_cell) or (T_cell,)
    returns: V_batch (B, T_cell)
    """
    # ensure batch dim
    if I_syn.ndim == 1:
        I_syn = I_syn[None, :]

    # vmap over trials
    V_batch = jax.vmap(lambda I_one: simulate_one_trial(cell_template, I_one, dt_cell))(I_syn)
    return V_batch