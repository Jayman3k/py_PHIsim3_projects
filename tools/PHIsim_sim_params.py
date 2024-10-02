from dataclasses import dataclass, replace, fields
from typing import ClassVar
from typing import Annotated

from tools.PHIsim_constants import SPEED_OF_LIGHT as c_l

@dataclass
class _PD:
    """Decorator for PHIsim_SimulationParams parameters that need to be printed to the parameters file. 
    The parameters will be printed as:
        {key}_{shortname}___   {value} # {comment}
    """
    key:       int
    shortname: str
    comment:   str
    new_section: str | None = None

class _NoPrint:
    """Decorator for PHIsim_SimulationParams parameters that are not printed to the parameters file."""	
    pass


@dataclass(kw_only=True)
class PHIsim_SimulationParams:
    # version descriptor of PHIsim, can be used indicate certain properties are (not) supported
    #################################################################################################
    PHIsim_branch:                     Annotated[str, _NoPrint] = "master"

    # simulation input files
    #################################################################################################
    params_file:                       Annotated[str, _NoPrint] # file where this will be printed to
    signal_input_file:                 Annotated[str, _NoPrint] = "signal_input.txt"
    device_file:                       Annotated[str, _NoPrint] = "device_input.txt"
    photond_file:                      Annotated[str, _NoPrint] = "photondfile.txt"
    carrier_file:                      Annotated[str, _NoPrint] = "carrierfile.txt"

    # simulation configuration parameters
    #################################################################################################

    # general parameters
    wavelength:                        Annotated[float, _PD(9000, 'wavelength',      'central wavelength in m')] 
    refractive_index:                  Annotated[float, _PD(9001, 'R_index',         'refractive index')]
    n_wavelen_segment:                 Annotated[int,   _PD(9002, 'opt_seg_len',     'optical path length of one time segment in integer nr of central wavelength')] 
    nr_cycles:                         Annotated[int,   _PD(9003, 'nr_cycles',       'Number of cycles in the simulation')]
    output_filename:                   Annotated[str,   _PD(9004, 'output_filename', 'output file name')]
    video_N:                           Annotated[int,   _PD(9005, 'video_N',         'if > 0 output data for video generated, store data for video every N time')] 
    video_start:                       Annotated[int,   _PD(9006, 'video_start',     'store the data for video from this cycle number')]
    random_seed:                       Annotated[int,   _PD(9007, 'random_seed',     'if >=0 seed value for random number generation , -1 is random seed number for random number generation')]

    # SOA parameters
    linear_gain_coefficient_amp:       Annotated[float, _PD(1000, 'aN_amplifier',   'linear gain coefficient amplifier in m2', "SOA Parameters")]
    standard_confinement_amp:          Annotated[float, _PD(1001, 'confinement_amp','standard confinement factor amp')]
    confinement_TPA_amp:               Annotated[float, _PD(1002, 'confine_amp_TPA','confinement factor 2 photon absorption amp')]
    confinement_Kerr_amp:              Annotated[float, _PD(1003, 'confine_amp_Ker','confinement factor Kerr effect amp')]
    transparency_carrier_density_amp:  Annotated[float, _PD(1004, 'Nc_tr_amp',      'transparency carrier density m-3 for the amplifier + N_min_amp')]
    minimum_carrier_density_amp:       Annotated[float, _PD(1005, 'N_min_amp',      'minimum carrier density m-3 in the amplifier')]
    epsilon1_amp:                      Annotated[float, _PD(1006, 'epsilon1_amp',   'amplifier non-linear gain coefficient espilon 1')]
    epsilon2_amp:                      Annotated[float, _PD(1007, 'epsilon2_amp',   'amplifier two photon gain/abs coefficient epsilon 2')]
    two_photon_absorption_amp:         Annotated[float, _PD(1008, 'TPA_amp',        'two photon absorption coefficient amp m/W amplifier beta 2')]
    carrier_lifetime_amp:              Annotated[float, _PD(1009, 'Tu_amp',         'carrier lifetime in the amplifier (sec)')]
    bimolecular_recombination_amp:     Annotated[float, _PD(1010, 'Brc',            'bimolecular recombination coeff. m3 * sec-1')]
    auger_recombination_amp:           Annotated[float, _PD(1011, 'Ca',             'Auger recombination coef in m6 sec-1')]
    drift_coefficient_N55_amp:         Annotated[float, _PD(1012, 'Da',             'Coefficient for N^5.5 drift coefficient')]
    active_region_height_amp:          Annotated[float, _PD(1013, 'h_act_r',        'active region hight Y direction (m) This value is consistent with the confinement, the ridge width and 500nm modeheight')]
    active_region_width_amp:           Annotated[float, _PD(1014, 'w_act_r',        'active region width X direction (m)')]
    other_loss_amp:                    Annotated[float, _PD(1015, 'other_loss_amp', 'passive other losses in the amplifier 1/m (combined with the carrier dep term this leads to a loss value for > 1kA/cm2)')]
    current_injection_efficiency_amp:  Annotated[float, _PD(1016, 'I_inj_eff',      'Current injection efficiency')]
    free_carrier_absorption_amp:       Annotated[float, _PD(1017, 'FCabsAct',       'free carrier absorption coeficient in the amplifier in m-1 per carrier per m3')]
    free_carrier_absorption_quadr_amp: Annotated[float, _PD(1018, 'FCabsAct2',      'free carrier absorption in the SOA active region quadratic term m-1 per carrier per m3 squared')]
    spontaneous_emission_coupling_amp: Annotated[float, _PD(1019, 'beta',           'spontaneous emission coupling factor to laser mode beta')]
    n2_index_amp:                      Annotated[float, _PD(1020, 'n2_index',       'non linear refractive index n2 in the amplifier')]
    carrier_linewidth_enh_factor_amp:  Annotated[float, _PD(1021, 'alpha_N_amp',    'carrier linewidth enhancement factor amp')]
    carrier_linewidth_T_enh_factor_amp:Annotated[float, _PD(1022, 'alpha_T_amp',    'carrier T linewidth enhancement factor amp')]

    # absorber parameters
    aN_absorber:                       Annotated[float, _PD(1100, 'aN_absorber',     'linear gain coefficient absorber in m2', "Absorber Parameters")]	
    confinement_abs:                   Annotated[float, _PD(1101, 'confinement_abs', 'standard confinement factor abs')] 
    confinement_TPA_abs:               Annotated[float, _PD(1102, 'confine_abs_TPA', 'confinement factor 2 photon absorption abs')]
    confinement_Kerr_abs:              Annotated[float, _PD(1103, 'confine_abs_Ker', 'confinement factor Kerr effect abs')]
    transparency_carrier_density_abs:  Annotated[float, _PD(1104, 'Nc_tr_abs',       'transparency carrier density m-3 for the absorber + N_min_abs')]
    minimum_carrier_density_abs:       Annotated[float, _PD(1105, 'N_min_abs',       'minimum carrier density m-3 in the absorber')]
    epsilon1_abs:                      Annotated[float, _PD(1106, 'epsilon1_abs',    'absorber non-linear gain coefficient epsilon 1')]
    epsilon2_abs:                      Annotated[float, _PD(1107, 'epsilon2_abs',    'absorber two photon gain/abs coefficient epsilon 2')]
    two_photon_absorption_abs:         Annotated[float, _PD(1108, 'TPA_abs',         'two photon absorption coefficient abs m/W absorber')]
    carrier_lifetime_abs:              Annotated[float, _PD(1109, 'Tu_abs',          'carrier lifetime in the absorber (sec)')]
    other_loss_abs:                    Annotated[float, _PD(1110, 'other_loss_abs',  'passive other losses in the absorber 1/m')]
    carrier_linewidth_enh_factor_abs:  Annotated[float, _PD(1111, 'alpha_N_abs',     'carrier linewidth enhancement factor abs')]
    carrier_linewidth_T_enh_factor_abs:Annotated[float, _PD(1112, 'alpha_T_abs',     'carrier T linewidth enhancement factor abs')]

    # isolation parameters
    carrier_lifetime_iso:              Annotated[float, _PD(1209, 'Tu_iso',          'carrier lifetime in a non contacted absorber isolation (sec)', "Isolation Section")]
                                                        
    # passive waveguides
    other_loss_pwg:                    Annotated[float, _PD(2000, 'other_loss_pwg',  'passive other losses in the passive waveguide 1/m', "Passive Waveguides")]

    # passive waveguide active
    confinement_wga:                   Annotated[float, _PD(2101, 'wga_confinement', 'standard confinement factor passive waveguide active (wga) (to be checked, see "derivation of equations v3 passivewg.xmcd")', "Passive Waveguides Active")]
    confinement_TPA_wga:               Annotated[float, _PD(2102, 'confine_wga_TPA', 'confinement factor 2 photon absorption passive waveguide active (wga) (to be checked, see "derivation of equations v3 passivewg.xmcd")')]
    confinement_Kerr_wga:              Annotated[float, _PD(2103, 'confine_wga_Ker', 'confinement factor Kerr effect wga (this value is also maybe 0.84?)')]
    two_photon_absorption_wga:         Annotated[float, _PD(2104, 'TPA_wga',         'two photon absorption coefficient abs m/W passive wga')]
    carrier_lifetime_wga:              Annotated[float, _PD(2105, 'Tu_wga',          'Carrier lifetime in a passive waveguide (sec) (to be checked)')]
    bimolecular_recombination_wga:     Annotated[float, _PD(2106, 'Brc_wga',         'Bimolecular recombination coeff. m3 * sec-1 for the passive waveguide core')]
    augur_recombination_wga:           Annotated[float, _PD(2107, 'Ca_wga',          'Auger recombination coef in m6 sec-1 for the passive waveguide core')]
    other_loss_wga:                    Annotated[float, _PD(2108, 'other_loss_wga',  'passive other losses in the passive waveguide active in m-1')]
    mode_surface_area_factor_wga:      Annotated[float, _PD(2109, 'mod_sur_fct_wga', 'Mode surface area factor for the passive waveguide active (e.g. to accomodate IMOS)')]
    free_carrier_absorption_wga:       Annotated[float, _PD(2110, 'FCabsWga',        'free carrier absorption coeficient in the passive waveguide active in m-1 per carrier per m3')]
    n2_index_wga:                      Annotated[float, _PD(2111, 'n2_index_wga',    'non linear refractive index n2 in passive waveguide active (wga) (Dvorak et al value)')]
    index_change_pc_wga:               Annotated[float, _PD(2112, 'i_ch_pc_wga',     'index change per free carrier per m3 in passive waveguide active (wga) (plasma effect and bandfilling, Weiming thesis )')]
    gvd_wga:                           Annotated[float, _PD(2113, 'gvd_wga',         'group velocity dispersion of the passive waveguide (sec2/m)')]

    # graphene loaded waveguide additional parameters
    alpha_non_sat_grwg:                Annotated[float, _PD(2200, 'GRwg_n_sat_loss', 'non-saturable loss of the graphene loaded passive waveguide in m-1', "Graphene Loaded Waveguides")]
    alpha_sat_grw:                     Annotated[float, _PD(2201, 'GR_wg_sat_loss',  'saturable loss of the graphene loaded passive waveguide in m-1')]
    carrier_lifetime_grw:              Annotated[float, _PD(2202, 'GR_wg_lifetime',  'recovery lifetime of the absorption in the graphene loaded waveguide')]
    N_sat_grw:                         Annotated[float, _PD(2203, 'GRwg_Nsat',       'saturated carrier population of graphene in m-2')]
    sigma_FCR_grw:                     Annotated[float, _PD(2204, 'GRwg_sigma_FCR',  'graphene-induced FCR coefficient')]
    graphene_width_grw:                Annotated[float, _PD(2205, 'GRwg_width',      'width of the graphene layer on the waveguide in m')]
    n2_index_grw:                      Annotated[float, _PD(2206, 'GRwg_n2_index',   'total non linear refractive index n2 of the graphene loaded waveguide')]

    # utility methods to convert between number of segments and units of time or length
    #################################################################################################

    def simulation_time_step(self):
        return self.n_wavelen_segment * self.wavelength / c_l
    
    def simulation_total_time(self):
        return self.simulation_time_step() * self.nr_cycles
    
    def simulation_segment_length(self, segments:int=1): ## in meters
        return segments * self.wavelength * self.n_wavelen_segment / self.refractive_index
    
    def length_to_num_segments(self, length) -> int:
        """Convert a length in meters to the number of simulation segments. Note that this introduces 
        a rounding error of at most 1/2 the simulation segment length. It's up to the user to check if
        this is a potential issue."""	
        return int(round(length / self.simulation_segment_length()))

    # list op properties that only appear in a certain version of PHIsim
    #################################################################################################
    __BUILD_VERSION_EXCLUSIVE = {
        "master" : (),
        "graphene-loaded-waveguide" : (
            # dispersion approximation
            "gvd_wga",
            # graphene-loaded waveguide properties
            "alpha_non_sat_grwg",
            "alpha_sat_grw",
            "carrier_lifetime_grw", 
            "N_sat_grw", 
            "sigma_FCR_grw",
            "graphene_width_grw",
            "n2_index_grw"
        )
    }

    @classmethod
    def __BUILD_VERSION_EXCLUSIVE_sanity_check(cls):
        # sanity check - check that all fields defined above are actually in PHIsim_SimulationParams
        allnames = [field.name for field in fields(PHIsim_SimulationParams)]
        for (version, fieldnames) in cls.__BUILD_VERSION_EXCLUSIVE.items():
            for name in fieldnames:
                if name not in allnames:
                    raise RuntimeError(f"BUILD_VERSION_EXCLUSIVE field {name} (defined for version {version}) is not in PHIsim_SimulationParams")
                
    def is_excluded(self, field):
        for (branch, exclusive_fields) in self.__BUILD_VERSION_EXCLUSIVE.items():
            for exclusive_field in exclusive_fields:
                if field == exclusive_field:
                    return branch != self.PHIsim_branch
        return False
                
    #################################################################################################

    def write_to_file(self):
        self.__BUILD_VERSION_EXCLUSIVE_sanity_check()
        """Write the simulation parameters to the configurartion file, using the annotations defined for each parameter."""	
        with open(self.params_file, 'w') as f:
            f.write("9998        #  input format indicator format 2023_04\n")
            f.write("#######################################################################################\n")
            f.write("# Control - general parameters\n")
            f.write("##############################\n")

            for field in fields(PHIsim_SimulationParams):
                annotation = field.type.__metadata__[0]
                
                if annotation == _NoPrint:
                    continue
                if self.is_excluded(field):
                    continue

                assert isinstance(annotation, _PD)
                if annotation.new_section != None:
                    f.write("##############################\n")
                    f.write(f"#### {annotation.new_section}\n")
                    f.write("##############################\n")

                f.write(f"{annotation.key}_{annotation.shortname:_<15}    {getattr(self, field.name):>14} # {annotation.comment}\n")
            f.write("\n")
            f.write("9999_END____________ 	# indicator end of input\n")

    def copy(self, **changes):
        return replace(self, **changes)

#################################################################################################

# default set of parameters provided/collected by prof. Erwin Bente
# augmented with parameters for the graphene-loaded-waveguide model
PHIsim_params_InGaAsP_ridge = PHIsim_SimulationParams(
    PHIsim_branch                      = "graphene-loaded-waveguide",

    wavelength                         = 1.547e-6,
    refractive_index                   = 3.7,
    n_wavelen_segment                  = 20,
    nr_cycles                          = 5000,
    params_file                        = "PHIsimv3_pars_InGaAsP_ridge.txt",
    output_filename                    = "PHIsimout.txt",
    video_N                            = 0,
    video_start                        = 3800,
    random_seed                        = 0,

    linear_gain_coefficient_amp        = 1.694e-19,
    standard_confinement_amp           = 0.053,
    confinement_TPA_amp                = 0.1,
    confinement_Kerr_amp               = 0.08,
    transparency_carrier_density_amp   = 0.6577e24,
    minimum_carrier_density_amp        = 0.0,
    epsilon1_amp                       = 0.2,
    epsilon2_amp                       = 200.0,
    two_photon_absorption_amp          = 3.7e-10,
    carrier_lifetime_amp               = 598e-12,
    bimolecular_recombination_amp      = 2.620e-16,
    auger_recombination_amp            = 5.269e-41,
    drift_coefficient_N55_amp          = 5.07e-102,
    active_region_height_amp           = 0.0265e-6,
    active_region_width_amp            = 2.0e-6,
    other_loss_amp                     = -1345,
    current_injection_efficiency_amp   = 0.65,
    free_carrier_absorption_amp        = 2.264e-21,
    free_carrier_absorption_quadr_amp  = -2.502e-46,
    spontaneous_emission_coupling_amp  = 1.0e-5,
    n2_index_amp                       = -3.5e-16,
    carrier_linewidth_enh_factor_amp   = 4.0,
    carrier_linewidth_T_enh_factor_amp = 2.0,

    aN_absorber                        = 1.694e-19,
    confinement_abs                    = 0.053,
    confinement_TPA_abs                = 0.1,
    confinement_Kerr_abs               = 0.08,
    transparency_carrier_density_abs   = 0.6577e24,
    minimum_carrier_density_abs        = 0.01e24,
    epsilon1_abs                       = 0.2,
    epsilon2_abs                       = 200.0,
    two_photon_absorption_abs          = 3.7e-10,
    carrier_lifetime_abs               = 15.0e-12,
    other_loss_abs                     = 100.0,
    carrier_linewidth_enh_factor_abs   = 4.0,
    carrier_linewidth_T_enh_factor_abs = 2.0,

    carrier_lifetime_iso               = 200.0e-12,

    other_loss_pwg                     = 34.5,

    confinement_wga                    = 0.75,
    confinement_TPA_wga                = 0.92,
    confinement_Kerr_wga               = 0.6,
    two_photon_absorption_wga          = 4.62e-10,
    carrier_lifetime_wga               = 100.0e-12,
    bimolecular_recombination_wga      = 2.620e-16,
    augur_recombination_wga            = 5.269e-41,
    other_loss_wga                     = 34.5,
    mode_surface_area_factor_wga       = 1.0,
    free_carrier_absorption_wga        = 7.2e-21,
    n2_index_wga                       = -1.5e-16,
    index_change_pc_wga                = 1.07e-26,
    gvd_wga                            = 0,

    alpha_non_sat_grwg                 = 1740.0,
    alpha_sat_grw                      = 1160.0,
    carrier_lifetime_grw               = 1e-12,
    N_sat_grw                          = 5e16,
    sigma_FCR_grw                      = 1e-5,
    graphene_width_grw                 = 1.5e-6,
    n2_index_grw                       = 2.6e-19) 

# if you don't want/need the graphene-loaded-waveguide
PHIsim_params_InGaAsP_ridge_master = PHIsim_params_InGaAsP_ridge.copy(
        PHIsim_branch = "master"
)


